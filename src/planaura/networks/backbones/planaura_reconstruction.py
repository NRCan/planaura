import functools
import torch
import torch.nn as nn
import numpy as np
from collections.abc import Iterable
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from einops import rearrange
from torch.utils.data import DataLoader, Dataset


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            num_frames=3,
            tubelet_size=1,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
            patch_stride=None
    ):
        super().__init__()
        if patch_stride is None:
            patch_stride = patch_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride  # patch_size[0]xpatch_size[1] for the reconstruction and 1x1 for cosine similarity
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        h_out = int((img_size[0] - 1 * (patch_size[0] - 1) - 1) / patch_stride[0] + 1)
        w_out = int((img_size[1] - 1 * (patch_size[1] - 1) - 1) / patch_stride[1] + 1)
        h_out_original = int((img_size[0] - 1 * (patch_size[0] - 1) - 1) / patch_size[0] + 1)
        w_out_original = int((img_size[1] - 1 * (patch_size[1] - 1) - 1) / patch_size[1] + 1)
        self.grid_size = (num_frames // tubelet_size, h_out, w_out)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.num_patches_original = self.grid_size[0] * h_out_original * w_out_original
        self.grid_size_original = (num_frames // tubelet_size, h_out_original, w_out_original)
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                              stride=(tubelet_size, patch_stride[0], patch_stride[1]), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # add a temporal dimension if num_frames = 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


class EmbeddingTensorDataset(Dataset):
    """Tensor dataset to parallelize the embedding attention mechanism"""
    def __init__(self, x, lis, max_q):

        self.database = []
        ra = range(len(lis))
        x[:, 0:1, :] = 0.0
        for i in ra:
            index_1_in_x = 1 + lis[i].detach().clone()
            index_1_length = index_1_in_x.shape[0]
            index_1_in_x = nn.ConstantPad1d((0, max_q - index_1_in_x.shape[0]), 0)(index_1_in_x)
            x_dums = x.detach().clone(). \
                index_select(1, index_1_in_x). \
                detach().cpu()
            for b in range(x.shape[0]):
                new_database_item = {'index_1_in_x': index_1_in_x.cpu(), 'tensor': x_dums[b, :, :],
                                     'index_1_length': index_1_length, 'index_0_in_x': b}
                self.database.append(new_database_item)

        self.database_index = self.build_index_list()

    def build_index_list(self):
        database_index = []
        for index in range(len(self.database)):
            database_index.append(index)
        return database_index

    def __len__(self):
        return len(self.database_index)

    def get_index_list(self):
        return self.database_index

    def __getitem__(self, index):
        data = self.database[index]
        return data


class PlanauraReconstructionEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 num_frames=3, tubelet_size=1,
                 in_chans=3, embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, patch_stride=16, embed_attention=True):
        super(PlanauraReconstructionEncoder, self).__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      num_frames=num_frames, tubelet_size=tubelet_size, in_chans=in_chans,
                                      embed_dim=embed_dim, patch_stride=patch_stride)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.pos_embed_original = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches_original + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        seq1 = [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)]
        self.blocks = nn.Sequential(*seq1)
        self.norm = norm_layer(embed_dim)
        self.embed_attention = embed_attention
        self.initialize_layers()
        self.embed_depth = depth

    def freeze_all(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        for feats in [self.patch_embed.proj, self.cls_token, self.pos_embed, self.pos_embed_original, self.blocks, self.norm]:
            if not isinstance(feats, Iterable):
                feats = [feats]
            for feat in feats:
                if not isinstance(feat, Iterable):
                    feat = [feat]
                for m in feat:
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Dropout):
                        m.eval()
                    try:
                        if m is not None:
                            for param in m.parameters():
                                param.requires_grad = False
                    except:
                        pass

    def initialize_layers(self):
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embed_original = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size_original,
                                                     cls_token=True)
        self.pos_embed_original.data.copy_(torch.from_numpy(pos_embed_original).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        if mask_ratio == 0.0:
            batchsize = ids_restore.shape[0]
            emlen = ids_restore.shape[1]
            ids_restore = torch.tensor([list(range(0, emlen))], device=x.device).repeat(batchsize, 1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        if mask_ratio == 0.0:
            x_masked = x

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def extract_feature_blocks(self, x, mask_ratio=0.0, intermediate_block_ids=None, norm=True, reshape=True):
        if intermediate_block_ids is None:
            intermediate_block_ids = [self.embed_depth - 1]
        if self.patch_embed.patch_stride[0] == 1:
            raise ValueError("Blocks can only be returned when stride equals the patch size")

        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio > 0:
            x, _, _ = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        features = []
        for blk_id in range(self.embed_depth):
            x = self.blocks[blk_id](x)
            if blk_id in intermediate_block_ids:
                features.append(x)

        # Remove cls token from intermediate features
        features = [feat[:, 1:, :] for feat in features]

        if norm:
            features = [self.norm(feat) for feat in features]

        # # note: each features item is "b (t h w) l" in shape
        # # if reshaping to be "b (t l) h w"
        if reshape:
            features = [rearrange(feat, 'b (t h w) l -> b (t l) h w',
                                  t=self.patch_embed.grid_size[0],
                                  h=self.patch_embed.grid_size[1],
                                  w=self.patch_embed.grid_size[2],
                                  l=self.patch_embed.embed_dim).contiguous()
                        for feat in features]

        return features

    def forward_for_neck(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, mask_ratio):
        x = self.patch_embed(x)

        if self.patch_embed.patch_stride[0] == 1:
            lt = list(range(0, self.patch_embed.grid_size[0]))
            h = self.patch_embed.grid_size[1]
            w = self.patch_embed.grid_size[2]
            lis = []
            max_q = 0
            for i in range(0, self.patch_embed.patch_size[0]):
                for j in range(0, self.patch_embed.patch_size[1]):
                    lr = list(range(i, self.patch_embed.grid_size[1], self.patch_embed.patch_size[0]))
                    lc = list(range(j, self.patch_embed.grid_size[2], self.patch_embed.patch_size[1]))
                    li = torch.LongTensor([t * h * w + r * w + c for t in lt for r in lr for c in lc]).to(x.device)
                    if li.shape[0] > max_q:
                        max_q = li.shape[0]
                    lis.append(li)

        if self.patch_embed.patch_stride[0] == 1:
            x_sub = x[:, lis[0], :].detach().clone()
            # add pos embed w/o cls token
            x_sub = x_sub + self.pos_embed_original[:, 1:, :]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if self.patch_embed.patch_stride[0] == 1:
            # masking: length -> length * mask_ratio
            x_sub, mask, ids_restore = self.random_masking(x_sub, mask_ratio)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.patch_embed.patch_stride[0] == 1:
            cls_tokens_sub = cls_token.expand(x_sub.shape[0], -1, -1)
            x_sub = torch.cat((cls_tokens_sub, x_sub), dim=1)

        if not self.embed_attention:
            x_without_embed_attention = x.detach().clone()
        else:
            x_without_embed_attention = None

        if self.patch_embed.patch_stride[0] != 1:
            x = self.blocks(x)
        else:
            if self.embed_attention:
                embed_dataset = EmbeddingTensorDataset(x, lis, max_q)
                embed_dataloader = DataLoader(embed_dataset, batch_size=4 * x.shape[0],
                                              shuffle=False, drop_last=False, pin_memory=True,
                                              num_workers=4)
                with torch.no_grad():
                    for iter_num, elem in enumerate(embed_dataloader):
                        a = self.blocks(elem['tensor'].to(device=x.device))
                        x[elem['index_0_in_x'].reshape(elem['index_1_in_x'].shape[0], 1).to(device=x.device),
                        elem['index_1_in_x'].to(device=x.device), :] = a
                # In this case, the naming is really not good; but I kept it for ease of returning
                # this x_without_embed_attention actually is the correct feature map outputted from attention blocks
                # with stride 1.
                x_without_embed_attention = x.detach().clone()

            x = self.blocks(x_sub)

        x = self.norm(x)
        if x_without_embed_attention is not None:
            x_without_embed_attention = self.norm(x_without_embed_attention[:, 1:, :])

        return x, x_without_embed_attention, mask, ids_restore


class PlanauraReconstruction(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 num_frames=3, tubelet_size=1,
                 in_chans=3, embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, patch_stride=16, embed_attention=True):
        super(PlanauraReconstruction, self).__init__()

        # --------------------------------------------------------------------------
        self.encoder = PlanauraReconstructionEncoder(img_size, patch_size,
                                                     num_frames, tubelet_size,
                                                     in_chans, embed_dim, depth, num_heads,
                                                     mlp_ratio, norm_layer, patch_stride, embed_attention)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.encoder.patch_embed.num_patches_original + 1, decoder_embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, tubelet_size * patch_size * patch_size * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_layers()

    def freeze_all(self):
        self.eval()
        self.encoder.freeze_all()
        for param in self.parameters():
            param.requires_grad = False
        for feats in [self.decoder_embed, self.mask_token, self.decoder_pos_embed, self.decoder_blocks,
                      self.decoder_norm, self.decoder_pred]:
            if not isinstance(feats, Iterable):
                feats = [feats]
            for feat in feats:
                if not isinstance(feat, Iterable):
                    feat = [feat]
                for m in feat:
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Dropout):
                        m.eval()
                    try:
                        if m is not None:
                            for param in m.parameters():
                                param.requires_grad = False
                    except:
                        pass

    def initialize_layers(self):
        self.encoder.initialize_layers()

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.encoder.patch_embed.grid_size_original,
                                                    cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.encoder.patch_embed.patch_size[0]
        tub = self.encoder.patch_embed.tubelet_size
        x = rearrange(imgs, 'b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)', tub=tub, p=p, q=p)

        return x

    def unpatchify(self, x):
        p = self.encoder.patch_embed.patch_size[0]
        num_p = self.encoder.patch_embed.img_size[0] // p
        tub = self.encoder.patch_embed.tubelet_size
        imgs = rearrange(x, 'b (t h w) (tub p q c) -> b c (t tub) (h p) (w q)', h=num_p, w=num_p, tub=tub, p=p, q=p)
        return imgs

    def forward_encoder(self, x, mask_ratio):
        x, x_without_embed_attention, mask, ids_restore = self.encoder.forward(x, mask_ratio)
        return x, x_without_embed_attention, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        x = x[:, 1:, :] #get rid of cls token

        return x

    def forward(self, imgs, mask_ratio=0.75):
        x, latent_features, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, ids_restore)
        if latent_features is not None:
            return pred, mask, latent_features
        else:
            return pred, mask, x[:, 1:, ].detach().clone()


def planaura_reconstruction(config):
    """
    Constructs a Planaura architecture
    """
    print('Planaura Model')
    img_size = config["model_params"]["img_size"]
    patch_size = config["model_params"]["patch_size"]
    num_frames = config["num_frames"]
    tubelet_size = config["model_params"]["tubelet_size"]
    in_chans = len(config["model_params"]["bands"])
    embed_dim = config["model_params"]["embed_dim"]
    depth = config["model_params"]["depth"]
    num_heads = config["model_params"]["num_heads"]
    decoder_embed_dim = config["model_params"]["decoder_embed_dim"]
    decoder_depth = config["model_params"]["decoder_depth"]
    decoder_num_heads = config["model_params"]["decoder_num_heads"]
    mlp_ratio = 4.
    norm_layer = functools.partial(torch.nn.LayerNorm, eps=1e-6)
    norm_pix_loss = False
    patch_stride = config['model_params']['patch_stride'] if 'patch_stride' in config['model_params'] else patch_size
    possible_strides = [1, patch_size]
    if patch_stride not in possible_strides:
        raise ValueError('patch stride can only be either 1 or the same as the patch size')
    embed_attention = config['model_params']['embed_attention'] if 'embed_attention' in config['model_params'] else True
    calculate_cosine_similarity = False if "change_map" not in config else config["change_map"]["return"]
    if patch_stride == 1 and not calculate_cosine_similarity:
        print('Warning: It is best to set the patch_stride to the same value as patch_size if you are not interested '
              'in creating the change intensity map!')
    model = PlanauraReconstruction(img_size, patch_size,
                                   num_frames, tubelet_size,
                                   in_chans, embed_dim, depth, num_heads,
                                   decoder_embed_dim, decoder_depth, decoder_num_heads,
                                   mlp_ratio, norm_layer, norm_pix_loss, patch_stride, embed_attention)

    return model
