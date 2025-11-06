from planaura.networks.model import PLANAURA
import copy
import torch
import math
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from collections import OrderedDict


def resample_abs_pos_embed(
        posemb,
        new_size,
        old_size=None,
        num_prefix_tokens=1,
        num_frames_by_tubelet=3,
        interpolation='bilinear',
        antialias=True):
    # new_size = new_model.backbone.encoder.patch_embed.grid_size
    # posemb = origin_model.(backbone.encoder.)pos_embed
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] * new_size[2] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[1] == new_size[2]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt((num_pos_tokens - num_prefix_tokens) / num_frames_by_tubelet))
        old_size = num_frames_by_tubelet, hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens, :], posemb[:, num_prefix_tokens:, :]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(old_size[0], old_size[1], old_size[2], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=[new_size[1], new_size[2]], mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    print(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb


def generate_model(config):
    model = PLANAURA(config)
    return model


def load_state_dict(config) -> dict:
    source = config["model_params"]["load_params"]["source"]
    if source == "huggingface":
        repo_id = config["model_params"]["load_params"]["repo_id"]
        model_name = config["model_params"]["load_params"]["model_name"]
        # download from Hugging Face and get a local file path
        chkpt_path = hf_hub_download(repo_id=repo_id, filename=model_name)
    elif source == "local":
        chkpt_path = config["model_params"]["load_params"]["checkpoint_path"]
        if not chkpt_path:
            raise ValueError("chkpt_path must be provided when source='local'")
    else:
        raise ValueError("source must be 'huggingface' or 'local'")

    checkpoint = torch.load(chkpt_path, map_location=torch.device("cpu"))

    if 'model_state_dict' in checkpoint.keys():
        st = checkpoint['model_state_dict']
    else:
        if chkpt_path.endswith('.pt'):
            st = checkpoint.state_dict()
        elif chkpt_path.endswith('.pth'):
            st = checkpoint
        else:
            raise ValueError(f'model format is not recognized; could not load the checkpoint: {chkpt_path}')

    return st, chkpt_path


def load_optimizer_scheduler_epoch_and_iter_num(chkpt_path: str) -> tuple:
    checkpoint = torch.load(chkpt_path, map_location=torch.device("cpu"))
    try:
        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer_st = checkpoint['optimizer_state_dict']
        else:
            optimizer_st = None
    except:
        optimizer_st = None

    try:
        if 'scheduler_state_dict' in checkpoint.keys():
            scheduler_st = checkpoint['scheduler_state_dict']
        else:
            scheduler_st = None
    except:
        scheduler_st = None

    try:
        epoch = checkpoint['epoch']
    except:
        epoch = -1

    try:
        scheduler_iter_num = checkpoint['scheduler_iter_num']
    except:
        scheduler_iter_num = -1

    return optimizer_st, scheduler_st, epoch, scheduler_iter_num


def resume_pretrained_network(config):
    model = generate_model(config)
    load_encoder_only = False if 'resume_encoder_only' not in config['model_params'] else config['model_params']['resume_encoder_only']
    keep_pos_embed = config['model_params']['keep_pos_embedding']
    restore_weights_only = True if 'restore_weights_only' not in config['model_params'] else config['model_params']['restore_weights_only']
    st, checkpoint_path = load_state_dict(config)
    st['backbone.encoder.pos_embed_original'] = copy.deepcopy(st['backbone.encoder.pos_embed'])
    if keep_pos_embed:
        st['backbone.encoder.pos_embed_original'] = resample_abs_pos_embed(
            st['backbone.encoder.pos_embed_original'],
            model.backbone.encoder.patch_embed.grid_size_original,
            num_frames_by_tubelet=config['num_frames'])
        st['backbone.encoder.pos_embed'] = resample_abs_pos_embed(st['backbone.encoder.pos_embed'],
                                                                  model.backbone.encoder.patch_embed.grid_size,
                                                                  num_frames_by_tubelet=config['num_frames'])
        if 'backbone.decoder_pos_embed' in st.keys():
            st['backbone.decoder_pos_embed'] = resample_abs_pos_embed(st['backbone.decoder_pos_embed'],
                                                                      model.backbone.encoder.patch_embed.grid_size_original,
                                                                      num_frames_by_tubelet=config['num_frames'])
    else:
        # discard fixed pos_embedding weight
        del st['backbone.encoder.pos_embed']
        if 'backbone.decoder_pos_embed' in st.keys():
            del st['backbone.decoder_pos_embed']
        if 'backbone.encoder.pos_embed_original' in st.keys():
            del st['backbone.encoder.pos_embed_original']
    in_channels = len(config["model_params"]["bands"])
    in_channels_original = st["backbone.encoder.patch_embed.proj.weight"].shape[1]
    if in_channels != in_channels_original:
        del st["backbone.encoder.patch_embed.proj.weight"]
        if "backbone.decoder_pred.weight" in st.keys():
            del st["backbone.decoder_pred.weight"]
        if "backbone.decoder_pred.bias" in st.keys():
            del st["backbone.decoder_pred.bias"]
    if not load_encoder_only:
        model.load_state_dict(st, strict=False)
        print(f'All trained model weights were restored.')
    else:
        filtered_state_dict = OrderedDict({k: v for k, v in st.items() if "backbone.encoder" in k})
        model.load_state_dict(filtered_state_dict, strict=False)
        filtered_state_dict = None
        print(f'Only trained encoder weights were restored.')
    st = None

    if load_encoder_only or restore_weights_only:
        optimizer_st, scheduler_st, epoch, scheduler_iter_num = None, None, -1, -1
    else:
        optimizer_st, scheduler_st, epoch, scheduler_iter_num = load_optimizer_scheduler_epoch_and_iter_num(checkpoint_path)

    if optimizer_st:
        print('Optimizer was restored from the checkpoint.')

    if scheduler_st:
        print('Scheduler was restored from the checkpoint.')

    if epoch > 0:
        print(f'Checkpoint was restored from epoch {epoch + 1}')

    if scheduler_iter_num != -1:
        print(f'Scheduler was restored from iteration {scheduler_iter_num}')

    return model, optimizer_st, scheduler_st, epoch + 1, scheduler_iter_num

