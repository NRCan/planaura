from planaura.networks.loss_functions import SimpleLoss, IgnoreLoss
from planaura.networks.backbone_generator import generate_backbone
import torch
import torch.nn as nn
from einops import rearrange


class PLANAURA(nn.Module):

    def __init__(self, config):

        super(PLANAURA, self).__init__()

        self.backbone = generate_backbone(config)

        self.no_data_float = 0.0001 if "no_data_float" not in config["model_params"] else config["model_params"]["no_data_float"]

        self.loss = IgnoreLoss()

        loss_from_config = "simple" if "loss" not in config["model_params"] else config["model_params"]["loss"]
        if loss_from_config == "simple":
            self.loss = SimpleLoss(no_data=self.no_data_float)

        self.training = True
        self.infer_only = False

        self.initialize_layers()

        self.freeze_backbone = False if 'freeze_backbone' not in config["model_params"] else config["model_params"]['freeze_backbone']
        self.freeze_encoder = False if 'freeze_encoder' not in config["model_params"] else config["model_params"]['freeze_encoder']

        self.mask_ratio = config["model_params"]['mask_ratio']

        self.is_reconstruction = True
        self.patchify_mask = None

        self.return_change_config = config["change_map"]
        self.return_features_config = config["feature_maps"]

        self.freeze_some_layers()

    def prepare_to_infer(self):
        self.training = False
        self.infer_only = True
        self.mask_ratio = 0.0
        if self.is_reconstruction:
            self.patchify_mask = torch.nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=self.backbone.encoder.patch_embed.patch_size,
                stride=self.backbone.encoder.patch_embed.patch_stride, padding=0, bias=False, device=self.device_())
            self.patchify_mask.weight.data.fill_(1.0)
            self.patchify_mask.weight.requires_grad = False

    def prepare_to_evaluate(self):
        self.training = False
        self.infer_only = False

    def prepare_to_train(self):
        self.training = True
        self.infer_only = False

    def initialize_layers(self):
        self.backbone.initialize_layers()

    def freeze_some_layers(self):
        if self.freeze_backbone:
            self.backbone.freeze_all()
        if self.freeze_encoder:
            self.backbone.encoder.freeze_all()

    def device_(self):
        return next(self.parameters()).device

    def forward(self, inputs):

        if self.infer_only:
            input_img_batch = inputs
        else:
            input_img_batch = inputs
            target_batch = self.backbone.patchify(input_img_batch)
            if self.backbone.norm_pix_loss:
                mean = target_batch.mean(dim=-1, keepdim=True)
                var = target_batch.var(dim=-1, keepdim=True)
                target_batch = (target_batch - mean) / (var + 1.e-6) ** .5

        predicted_img_batch, mask_batch, latent_batch_src = self.backbone(input_img_batch, self.mask_ratio)


        if self.no_data_float is not None:
            input_im_nodata_mask = ((rearrange(input_img_batch, 'b g f h w-> b (g f) h w') == self.no_data_float) |
                                    (torch.isnan(rearrange(input_img_batch, 'b g f h w-> b (g f) h w')))).any(dim=1, keepdim=True)
        else:
            input_im_nodata_mask = (torch.isnan(rearrange(input_img_batch, 'b g f h w-> b (g f) h w'))).any(dim=1, keepdim=True)

        if self.infer_only:
            cosine_nodata_mask = self.patchify_mask(input_im_nodata_mask.float()).squeeze(1)
            cosine_nodata_mask = cosine_nodata_mask > 0

        if self.training:
            loss = self.compute_loss(predicted_img_batch, target_batch, mask_batch)
            return loss

        elif self.infer_only:
            # Create mask and prediction images (un-patchify)
            mask_img = self.backbone.unpatchify(
                mask_batch.unsqueeze(-1).repeat(1, 1, predicted_img_batch.shape[-1]))
            del mask_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            predicted_img_batch = self.backbone.unpatchify(predicted_img_batch).float()

            if self.mask_ratio == 0.0:
                rec_img = predicted_img_batch
            else:
                # mix visible and predicted patches
                rec_img = input_img_batch.clone()
                rec_img[mask_img == 1] = predicted_img_batch[mask_img == 1]  # binary mask: 0 is keep, 1 is remove

            del predicted_img_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del mask_img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Switch zeros/ones in mask images so masked patches appear darker in plots (better visualization)
            # mask_img = (~(mask_img.to(torch.bool))).to(torch.float)
            # rec_img: batch_size*channels*num_frames*img_size*img_size
            # mask_img: batch_size*channels*num_frames*img_size*img_size
            # lat_embed : batch_size*(img_size/patch_size*img_size/patch_size*num_frames+1)*embed_size
            rec_img[input_im_nodata_mask.unsqueeze(1).expand_as(rec_img)] = self.no_data_float
            cosine_map = None
            which_before = None
            feature_maps = None
            if self.return_change_config['return'] and 2 <= rec_img.shape[2] <= 3:
                latent_batch = rearrange(latent_batch_src.float(), 'b (t h w) c -> b t (h w) c',
                                         t=self.backbone.encoder.patch_embed.grid_size[0],
                                         h=self.backbone.encoder.patch_embed.grid_size[1],
                                         w=self.backbone.encoder.patch_embed.grid_size[2])

                norms1 = torch.sqrt(torch.sum(latent_batch[:, 0, :, :] * latent_batch[:, 0, :, :], dim=-1))
                norms2 = torch.sqrt(torch.sum(latent_batch[:, 1, :, :] * latent_batch[:, 1, :, :], dim=-1))
                if rec_img.shape[2] == 3:
                    norms3 = torch.sqrt(torch.sum(latent_batch[:, 2, :, :] * latent_batch[:, 2, :, :], dim=-1))
                    latent_batch[:, 0, :, 0] = torch.sum(latent_batch[:, 0, :, :] * latent_batch[:, 2, :, :],
                                                         dim=-1)
                    latent_batch[:, 1, :, 0] = torch.sum(latent_batch[:, 1, :, :] * latent_batch[:, 2, :, :],
                                                         dim=-1)
                    latent_batch[:, 0, :, 0] = latent_batch[:, 0, :, 0] / (norms1 * norms3)
                    latent_batch[:, 1, :, 0] = latent_batch[:, 1, :, 0] / (norms2 * norms3)
                    latent_batch[:, 2, :, 0] = torch.maximum(latent_batch[:, 0, :, 0], latent_batch[:, 1, :, 0])

                    cosine_map = rearrange(latent_batch[:, 2, :, 0], 'b (h w) -> b h w',
                                           h=self.backbone.encoder.patch_embed.grid_size[1],
                                           w=self.backbone.encoder.patch_embed.grid_size[2])
                else:  # rec_img.shape[2] == 2
                    latent_batch[:, 0, :, 0] = torch.sum(latent_batch[:, 0, :, :] * latent_batch[:, 1, :, :],
                                                         dim=-1)
                    latent_batch[:, 0, :, 0] = latent_batch[:, 0, :, 0] / (norms1 * norms2)

                    cosine_map = rearrange(latent_batch[:, 0, :, 0], 'b (h w) -> b h w',
                                           h=self.backbone.encoder.patch_embed.grid_size[1],
                                           w=self.backbone.encoder.patch_embed.grid_size[2])
                cosine_map = torch.nan_to_num(cosine_map, nan=-100.0, posinf=-100.0, neginf=-100.0)
                cosine_map[cosine_nodata_mask] = -100.0
                which_before = latent_batch[:, :rec_img.shape[2]-1, :, 0].argmax(dim=1)
                which_before = rearrange(which_before, 'b (h w) -> b h w',
                                         h=self.backbone.encoder.patch_embed.grid_size[1],
                                         w=self.backbone.encoder.patch_embed.grid_size[2])
                mask = (cosine_map == -100.0)
                which_before[mask] = -100
            if self.return_features_config['return']:
                feature_maps = rearrange(latent_batch_src.float(), 'b (t h w) c -> b t h w c',
                                         t=self.backbone.encoder.patch_embed.grid_size[0],
                                         h=self.backbone.encoder.patch_embed.grid_size[1],
                                         w=self.backbone.encoder.patch_embed.grid_size[2])
                feature_maps = torch.nan_to_num(feature_maps, nan=-100.0, posinf=-100.0, neginf=-100.0)
                feature_maps[cosine_nodata_mask.unsqueeze(1).unsqueeze(-1).expand_as(feature_maps)] = -100.0
            return rec_img, (cosine_map, which_before), feature_maps

        else:  # perform evaluation-only mode
            loss = self.compute_loss(predicted_img_batch, target_batch, mask_batch)
            mask_img = self.backbone.unpatchify(
                mask_batch.unsqueeze(-1).repeat(1, 1, predicted_img_batch.shape[-1]))
            pred_img = self.backbone.unpatchify(predicted_img_batch)
            # Mix visible and predicted patches
            rec_img = input_img_batch.clone()
            rec_img[mask_img == 1] = pred_img[mask_img == 1]  # binary mask: 0 is keep, 1 is remove
            if self.mask_ratio == 0.0:
                rec_img = pred_img
            return rec_img, loss


    def compute_loss(self, prediction, target, mask=None):
        if mask is None:
            return self.loss(prediction, target)
        else:
            return self.loss(prediction, target, mask)
