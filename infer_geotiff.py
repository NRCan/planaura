from planaura.utils.inference_manager import planaura_infer_geotiff, planaura_mosaic_geotiff, determine_mosaicable_files
from planaura.utils.log_generator import setup_file_logging
import os
import sys
import shutil
import time
from datetime import datetime
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf


def setup_config():
    default_config = {
        "task_list": ["infer", "mosaic"],
        "use_gpu": True,
        "using_multi_gpu": False,
        "num_predictions": 1,
        "autocast_float16": False,
        "save_reconstructed_images": False,
        "concatenate_char": "!",
        "minimum_valid_percentage": 0.9,
        "use_xarray": False,
        "mosaic_params":
            {
                "target_crs": 'EPSG:3979',
                "target_resolution": 30,
                "delete_residues": False,
                "mosaic_save_postfix": "dummy"
            },
        "paths_included_in_csvs": True,
        "reference_prefix": None,
        "ignore_prefixes": None,
        "num_frames": 2,
        "tif_compression": "NONE",
        "change_map":
            {
                "return": True,
                "upsample_cosine_map": True,
                "save_dates_layer": False,
                "save_fmask_layer": False,
                "fmask_in_bit": False,
                "date_regex": None,
            },
        "feature_maps":
            {
                "return": True,
                "write_as_csv": True,
                "write_as_image": False,
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


def infer_geo(config):
    config['csv_inference_file_geotiffs'] = config['csv_inference_file']
    config['csv_inference_file'] = os.path.join(os.path.split(config['csv_inference_file'])[0],
                                                     'temp_inference.csv')
    num_frames_ = config['num_frames']
    for _fr in range(num_frames_):
        _sfr = str(_fr)
        if 'inference_input_folder_frame_' + _sfr in config:
            config['inference_input_folder_geotiff_frame_' + _sfr] = config[
                'inference_input_folder_frame_' + _sfr]
        else:
            config['inference_input_folder_geotiff_frame_' + _sfr] = ''

        if config['inference_input_folder_geotiff_frame_' + _sfr]:
            os.makedirs(config['inference_input_folder_geotiff_frame_' + _sfr], exist_ok=True)

        config['inference_input_folder_frame_' + _sfr] = os.path.join(
            config['inference_save_folder_frame_' + _sfr], 'temp_input_inference')
        os.makedirs(config['inference_input_folder_frame_' + _sfr], exist_ok=True)

        config['inference_save_folder_geotiff_frame_' + _sfr] = config['inference_save_folder_frame_' + _sfr]
        os.makedirs(config['inference_save_folder_geotiff_frame_' + _sfr], exist_ok=True)

        config['inference_save_folder_frame_' + _sfr] = os.path.join(
            config['inference_save_folder_frame_' + _sfr], 'temp_save_inference')
        os.makedirs(config['inference_save_folder_frame_' + _sfr], exist_ok=True)

    config["use_xarray"] = False
    task_list = config["task_list"]
    reference_prefix = config["reference_prefix"]
    ignore_prefixes = config["ignore_prefixes"]
    concat_char = config["concatenate_char"]

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_file_logging(os.path.join(config['inference_save_folder_geotiff_frame_0'],
                                    'log_' + timestamp_str + '.log'))
    print("----- CONFIGS -----")
    print(config)
    if "infer" in task_list:
        _start_time = time.time()
        planaura_infer_geotiff(config)
        print(f"infer time (sec): {time.time() - _start_time}")
    if "mosaic" in task_list:
        for _fr in range(num_frames_):
            _sfr = str(_fr)
            try:
                shutil.rmtree(config['inference_input_folder_frame_' + _sfr])
                shutil.rmtree(config['inference_save_folder_frame_' + _sfr])
            except:
                pass
        _start_time = time.time()
        files_found, prefixes_found = determine_mosaicable_files(config)

        if files_found:
            planaura_mosaic_geotiff(config,
                                    reference_prefix=reference_prefix,
                                    ignore_prefixes=ignore_prefixes)
        else:
            print("We couldn't find any files satisfying the formatting requirements. "
                  "Ensure your files exist, rename them with the formatting rules as below, and re-run!")
            print("The followings are allowed naming styles for the files that need merging:")
            print(f"    cosine_map{concat_char}BEFOREINPUTFILENAME{concat_char}AFTERINPUTFILENAME.tif")
            print(f"    after_date{concat_char}BEFOREINPUTFILENAME{concat_char}AFTERINPUTFILENAME.tif")
            print(f"    before_date{concat_char}BEFOREINPUTFILENAME{concat_char}AFTERINPUTFILENAME.tif")
            print(f"    quality_fmask{concat_char}BEFOREINPUTFILENAME{concat_char}AFTERINPUTFILENAME.tif")
            print(f"    infer_0{concat_char}BEFOREINPUTFILENAME{concat_char}AFTERINPUTFILENAME.tif")
            print(f"    infer_1{concat_char}BEFOREINPUTFILENAME{concat_char}AFTERINPUTFILENAME.tif")
            print(f"    feature_maps_0{concat_char}BEFOREINPUTFILENAME{concat_char}AFTERINPUTFILENAME.tif")
            print(f"    feature_maps_1{concat_char}BEFOREINPUTFILENAME{concat_char}AFTERINPUTFILENAME.tif")
            print("Additionally, for your assembled mosaic_before_input and mosaic_after_input files to be creatable, "
                  "you need to make sure that BEFOREINPUTFILENAME and AFTERINPUTFILENAME filenames "
                  "do have a matching date_regex as identified in your Configs.")
        print(f"mosaic time (sec): {time.time() - _start_time}")


if __name__ == "__main__":

    argv = sys.argv
    dict_config = {}
    conf_path = os.path.split(argv[1])[0]
    conf_name = os.path.splitext(os.path.split(argv[1])[1])[0]
    if os.path.isabs(argv[1]):
        with initialize_config_dir(version_base=None, config_dir=conf_path, job_name="experiment_infer_geotiff"):
            conf = compose(config_name=conf_name)
    else:
        with initialize(version_base=None, config_path=conf_path, job_name="experiment_infer_geotiff"):
            conf = compose(config_name=conf_name)

    dict_config = OmegaConf.to_container(conf, resolve=True)
    def_config = setup_config()
    dict_config = OmegaConf.merge(def_config, dict_config)

    infer_geo(dict_config)
