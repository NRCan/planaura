from planaura.utils.inference_manager import write_reconstructed_images, fetch_model, write_feature_maps, fetch_simple_dataloader
import os
import torch
import sys
from tqdm import tqdm
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf


def setup_config():
    default_config = {
        "paths_included_in_csvs": False,
        "use_gpu": True,
        "using_multi_gpu": False,
        "num_predictions": 1,
        "autocast_float16": False,
        "save_reconstructed_images": False,
        "minimum_valid_percentage": 0.9,
        "use_xarray": False,
        "num_frames": 2,
        "tif_compression": "NONE",
        "change_map":
            {
                "return": True,
                "upsample_cosine_map": True
            },
        "feature_maps":
            {
                "return": True,
                "write_as_image": True,
                "embeddings": None
            },
        "model_params": {
            "load_params":
                {
                    "source": "local",
                    "checkpoint_path": "",
                    "repo_id": "",
                    "model_name": ""
                },
            "freeze_backbone": False,
            "freeze_encoder": False,
            "resume_encoder_only": False,
            "keep_pos_embedding": True,
            "restore_weights_only": True,
            "ignore_index": 255,
            "loss": "simple",
            "backbone": "planaura_reconstruction",
            "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
            "img_size": 512,
            "depth": 12,
            "decoder_depth": 8,
            "patch_size": 16,
            "patch_stride": 16,
            "embed_attention": True,
            "embed_dim": 768,
            "decoder_embed_dim": 512,
            "num_heads": 12,
            "decoder_num_heads": 16,
            "mask_ratio": 0.75,
            "tubelet_size": 1,
            "no_data": -9999,
            "no_data_float": 0.0001
        }
    }
    return default_config


def infer_simple(config):
    using_gpu = config["use_gpu"]
    if using_gpu and not torch.cuda.is_available():
        using_gpu = False
    using_multi_gpu = config["use_multi_gpu"]
    if using_multi_gpu and not torch.cuda.is_available():
        using_multi_gpu = False
    using_autocast_float16 = config["autocast_float16"]
    calculate_cosine_similarity = config["change_map"]["return"]
    return_feature_maps = config["feature_maps"]["return"]
    write_as_im_feature_maps = config["feature_maps"]["write_as_image"]
    upsample_cosine_map = config["change_map"]["upsample_cosine_map"]
    patch_stride = config['model_params']['patch_stride'] if 'patch_stride' in config['model_params'] else \
        config['model_params']['patch_size']
    upsample_cosine_map_factor = -1.0
    if upsample_cosine_map and patch_stride > 1:
        upsample_cosine_map_factor = patch_stride

    for fr in range(config['num_frames']):
        os.makedirs(config['inference_save_folder_frame_' + str(fr)], exist_ok=True)

    infer_dataloader = fetch_simple_dataloader(config, using_gpu)

    model = fetch_model(config)
    device = 'cpu'
    if using_gpu:
        model = model.cuda()
        device = 'cuda'
    if using_multi_gpu:
        model_multi_gpu = torch.nn.DataParallel(model).cuda()
        model = model_multi_gpu.module
    print(f'device={device}')
    print(f'using_multi_gpu={using_multi_gpu}')
    model.prepare_to_infer()

    if not using_multi_gpu:
        model.eval()
    else:
        model_multi_gpu.eval()

    with torch.no_grad():
        for iter_num, data in tqdm(enumerate(infer_dataloader)):
            model_device = model.device_()
            input_img_batch = data['img_input']
            batch_image_names = [data['input_file_0']]
            for fr in range(1, config['num_frames']):
                batch_image_names.append(data['input_file_' + str(fr)])
            if model.is_reconstruction:
                if using_autocast_float16:
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        if not using_multi_gpu:
                            predicted_img_batch, cosine_maps, feat_maps = model(input_img_batch.to(device=model_device).float())
                        else:
                            predicted_img_batch, cosine_maps, feat_maps = model_multi_gpu(input_img_batch.cuda().float())
                else:
                    if not using_multi_gpu:
                        predicted_img_batch, cosine_maps, feat_maps = model(input_img_batch.to(device=model_device).float())
                    else:
                        predicted_img_batch, cosine_maps, feat_maps = model_multi_gpu(input_img_batch.cuda().float())

                write_reconstructed_images(config, predicted_img_batch, batch_image_names,
                                           cosine_maps=cosine_maps[0],
                                           calculate_cosine_similarity=calculate_cosine_similarity,
                                           which_before_epochs=None,
                                           upsample_cosine_map_factor=upsample_cosine_map_factor)
                if return_feature_maps:
                    write_feature_maps(config, feat_maps, batch_image_names, write_as_im_feature_maps,
                                       upsample_feature_map_factor=upsample_cosine_map_factor)
                del cosine_maps, predicted_img_batch, input_img_batch, feat_maps
            else:
                print("nothing implemented yet for when model is not is_reconstruction")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":

    argv = sys.argv
    dict_config = {}
    conf_path = os.path.split(argv[1])[0]
    conf_name = os.path.splitext(os.path.split(argv[1])[1])[0]
    if os.path.isabs(argv[1]):
        with initialize_config_dir(version_base=None, config_dir=conf_path, job_name="experiment_infer_photo"):
            conf = compose(config_name=conf_name)
    else:
        with initialize(version_base=None, config_path=conf_path, job_name="experiment_infer_photo"):
            conf = compose(config_name=conf_name)

    dict_config = OmegaConf.to_container(conf, resolve=True)
    def_config = setup_config()
    dict_config = OmegaConf.merge(def_config, dict_config)

    infer_simple(dict_config)
