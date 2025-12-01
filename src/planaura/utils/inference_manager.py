from planaura.networks.network_generator import resume_pretrained_network
from planaura.utils.data.dataset_generator import CSVDataset, CustomCollation
from planaura.utils.data.batch_sampler import Sampler_NoShuffle
from planaura.utils.data.sample_creator import DataSampleCreator
from planaura.utils.raster_management.read_geotiff import read_geotiff
from planaura.utils.data.data_folder_parser_generator import fetch_parser_geotiff
from planaura.utils.raster_management.tiling_geotiff import tile_geotiffs
from planaura.utils.raster_management.write_geotiff import write_geotiff
from planaura.utils.raster_management.merge_geotiff import merge_geotiffs
from planaura.utils.data.sample_normalizer import SampleImageUnNormalizer
import gc
import re
import glob
import os
import torch
import shutil
import rasterio
import copy
import time
import numpy as np
import pandas as pd
import torch.nn.functional as Func
from datetime import datetime, timedelta
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.transform import from_origin
from scipy.stats import mode as array_mode
from scipy.ndimage import zoom
from osgeo import gdal
from numpy.linalg import norm


def fetch_simple_dataloader(config, use_gpu):
    infer_dataset = CSVDataset(mode='inference', data_file=config['csv_inference_file'], config=config,
                               transforms=transforms.Compose([
                                   DataSampleCreator(config)
                               ]), sampler=Sampler_NoShuffle())
    print('Total number of inference images: {}'.format(len(infer_dataset)))
    custom_collation = CustomCollation(config)
    n_workers = 24 if "data_loader_num_workers" not in config else config["data_loader_num_workers"]
    infer_dataloader = DataLoader(infer_dataset, num_workers=n_workers, collate_fn=custom_collation,
                                  batch_size=config['batch_size'], shuffle=False, drop_last=False, pin_memory=use_gpu)

    return infer_dataloader

def write_unnormalized_image(save_folder, image_name, img, no_data, compression):
    image_name_wo_extension = os.path.splitext(os.path.split(image_name)[1])[0]
    write_geotiff('unknown', None, '', img, save_folder, image_name_wo_extension, alpha_im=None,
                  data_type=gdal.GDT_Float32, save_tfw=False,
                  no_data_values=[no_data] * img.shape[2], compression=compression)


def cosine_sim(A, B):
    norm_product = norm(A) * norm(B)
    if norm_product != 0:
        cosine = np.dot(A, B) / norm_product
    else:
        cosine = -100
    return cosine


def resample_cosines_features(arr, upsample_scale_factor, clip=True):
    mask = (arr != -100.0).astype(arr.dtype)
    data_weighted = np.where(mask > 0, arr, 0.0).astype(arr.dtype)
    zoom_factors = (upsample_scale_factor, upsample_scale_factor)
    if arr.ndim == 2:
        num = zoom(data_weighted, zoom_factors, order=3, mode="nearest", grid_mode=True)
        den = zoom(mask, zoom_factors, order=3, mode="nearest", grid_mode=True)
    elif arr.ndim == 3:
        num = np.stack([
            zoom(data_weighted[..., c], zoom_factors, order=3, mode="nearest", grid_mode=True)
            for c in range(arr.shape[2])
        ], axis=-1)
        den = np.stack([
            zoom(mask[..., c], zoom_factors, order=3, mode="nearest", grid_mode=True)
            for c in range(arr.shape[2])
        ], axis=-1)
    else:
        raise ValueError('cannot handle more than 3 dimensions')
    out = np.full_like(num, -100.0, dtype=arr.dtype)
    good = den > 0.9
    if clip:
        out[good] = np.clip(num[good], -1.0, 1.0)
    else:
        out[good] = num[good]
    return out


def resample_epochs(arr, upsample_scale_factor, num_frames):
    mask = (arr != -100).astype(arr.dtype)
    zoom_factors = (upsample_scale_factor, upsample_scale_factor)
    num = zoom(arr, zoom_factors, order=0, mode="nearest", grid_mode=True)
    den = zoom(mask, zoom_factors, order=0, mode="nearest", grid_mode=True)
    out = np.full_like(num, -100, dtype=arr.dtype)
    good = den > 0.9
    out[good] = np.clip(num[good], 0, num_frames-2)
    return out


def transform_xy_csv_with_dict(offsets_scales_dict, x, y, fp_wo):
    M = offsets_scales_dict.get(fp_wo)
    return M[2] * x + M[0], M[3] * y + M[1]


def merge_feat_csv_files(sfr, images_path, offsets_scales_dict, full_save_path, x_start, y_start, x_len, y_len):
    files_wo_extension = [
        os.path.splitext(f)[0]
        for f in os.listdir(images_path)
        if f.startswith('feats_' + sfr) and f.endswith(".csv")
    ]
    first = True
    x_col = 'x'
    y_col = 'y'
    with open(full_save_path, "w", newline="") as out_f:
        for fp_wo in files_wo_extension:
            for chunk in pd.read_csv(os.path.join(images_path, fp_wo + '.csv'), chunksize=262144):
                if x_col not in chunk.columns or y_col not in chunk.columns:
                    continue
                mask = pd.Series(True, index=chunk.index)
                mask &= chunk[x_col].between(x_start, x_start+x_len-1, inclusive='both')
                mask &= chunk[y_col].between(y_start, y_start+y_len-1, inclusive='both')
                if not mask.any():
                    continue
                chunk = chunk.loc[mask]
                x_new, y_new = transform_xy_csv_with_dict(offsets_scales_dict, chunk[x_col], chunk[y_col], fp_wo)
                chunk[x_col] = x_new
                chunk[y_col] = y_new

                if first:
                    expected_cols = list(chunk.columns)
                else:
                    chunk = chunk.reindex(columns=expected_cols)

                chunk.to_csv(out_f, index=False, header=first)
                if first:
                    first = False


def write_feature_maps(config, predicted_features, batch_image_names, write_as_im, upsample_feature_map_factor):
    if predicted_features is not None:
        embeddings = config["feature_maps"]["embeddings"]
        if not embeddings:
            embeddings = list(range(0,predicted_features.shape[4]))
        predicted_features = predicted_features.cpu().detach().numpy()
        patch_size = config["model_params"]["patch_size"]
        patch_stride = config["model_params"]["patch_stride"]
        compression_type = config["tif_compression"]
        if patch_stride == 1:
            strt_index = int(patch_size / 2 - 1)
        else:
            strt_index = 0

        num_frames = predicted_features.shape[1]
        for fr in range(num_frames):
            save_folder = config['inference_save_folder_frame_' + str(fr)]
            for b in range(predicted_features.shape[0]):
                csv_name = 'feats_' + str(fr) + '_' + os.path.splitext(os.path.split(batch_image_names[fr][b])[1])[0]
                A = predicted_features[b, fr, ...]
                H, W, C = A.shape

                ys = strt_index + np.arange(H) * patch_stride
                xs = strt_index + np.arange(W) * patch_stride

                Y, X = np.meshgrid(ys, xs, indexing='ij')  # both H x W

                coords = np.column_stack([X.reshape(-1), Y.reshape(-1)])  # (H*W) x 2, columns: x, y
                feats = A.reshape(-1, C)  # (H*W) x C
                data = np.hstack([coords, feats])  # (H*W) x (2+C)
                columns = ["x", "y"] + [f"f{k}" for k in range(C)]
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(os.path.join(save_folder, csv_name + '.csv'), index=False)
                if write_as_im:
                    A= A[:,:,embeddings]
                    if upsample_feature_map_factor > 0:
                        A = resample_cosines_features(A, upsample_feature_map_factor, clip=False)
                    write_geotiff('unknown', None, '', A, save_folder,
                                  csv_name, alpha_im=None,
                                  data_type=gdal.GDT_Float32, save_tfw=False,
                                  no_data_values=[-100.0]*A.shape[2], compression=compression_type,
                                  band_descriptions=[f"f{k}" for k in embeddings])


def write_reconstructed_images(config, predicted_img_batch, batch_image_names, cosine_maps=None,
                               calculate_cosine_similarity=False, which_before_epochs=None,
                               upsample_cosine_map_factor=-1):
    un_normalizer = SampleImageUnNormalizer(config)
    num_frames = config["num_frames"]
    no_data = config["model_params"]["no_data"]
    img_size = config["model_params"]["img_size"]
    patch_size = config["model_params"]["patch_size"]
    embed_dim = config["model_params"]["embed_dim"]
    embed_w = img_size // patch_size
    embed_length = embed_w * embed_w
    compression_type = config["tif_compression"]
    do_write_unnormalized = True if "save_reconstructed_images" not in config else config["save_reconstructed_images"]

    for b in range(predicted_img_batch.shape[0]):
        if calculate_cosine_similarity and cosine_maps is not None:
            cosine_map = cosine_maps[b, :, :].cpu().detach().numpy()
            if upsample_cosine_map_factor > 0:
                cosine_map = resample_cosines_features(cosine_map, upsample_cosine_map_factor)
            image_name_wo_extension = 'cosines_' + os.path.splitext(os.path.split(batch_image_names[0][b])[1])[0]
            write_geotiff('unknown', None, '', cosine_map, config['inference_save_folder_frame_' + str(0)],
                          image_name_wo_extension, alpha_im=None,
                          data_type=gdal.GDT_Float32, save_tfw=False,
                          no_data_values=[-100.0], compression=compression_type)
            if which_before_epochs is not None:
                which_before_epoch = which_before_epochs[b, :, :].cpu().detach().numpy()
                if upsample_cosine_map_factor > 0:
                    which_before_epoch = resample_epochs(which_before_epoch, upsample_cosine_map_factor, num_frames)
                image_name_wo_extension = 'before_epochs_' + \
                                          os.path.splitext(os.path.split(batch_image_names[0][b])[1])[0]
                write_geotiff('unknown', None, '', which_before_epoch, config['inference_save_folder_frame_' + str(0)],
                              image_name_wo_extension, alpha_im=None,
                              data_type=gdal.GDT_Int32, save_tfw=False,
                              no_data_values=[-100], compression=compression_type)

        if do_write_unnormalized:
            # predicted_img_batch: batch_size*channels*num_frames*img_size*img_size
            img = predicted_img_batch[b, :, :, :, :].permute(1, 2, 3, 0).cpu().detach().numpy()
            # num_frames*height*row*channels
            for fr in range(num_frames):
                imgeg = un_normalizer(img[fr, :, :, :])
                write_unnormalized_image(config['inference_save_folder_frame_' + str(fr)],
                                         batch_image_names[fr][b], imgeg, no_data, compression_type)


def convert_yeardoy_to_string(yeardoy, date_regex):
    year = yeardoy // 1000
    doy = yeardoy % 1000
    # find the (\d{n}) part
    match = re.search(r'(.*)\(\\d\{(\d+)\}\)(.*)', date_regex)
    if not match:
        raise ValueError(f"Invalid regex pattern {date_regex}")

    prefix = match.group(1).replace('\\', '')
    num_digits = int(match.group(2))
    suffix = match.group(3).replace('\\', '')

    if num_digits == 7:
        # HLS-style: YYYYDOY
        number_part = f"{year}{doy:03d}"
    elif num_digits == 8:
        # S2-style: YYYYMMDD
        date_obj = datetime(int(year), 1, 1) + timedelta(days=int(doy - 1))
        number_part = date_obj.strftime("%Y%m%d")
    else:
        raise ValueError(f"Unsupported number of digits {num_digits} for a date string!")

    return f"{prefix}{number_part}{suffix}"


def extract_dates_as_int(file_names, date_regex):
    extracted_dates = []
    print('Image names:', file_names)
    for i, file_name in enumerate(file_names):
        match = re.search(date_regex, file_name)
        yyyy_doy = i
        if match:
            date_str = match.group(1)
            try:
                if len(date_str) == 7 and date_str.isdigit():
                    yyyy_doy = int(date_str)
                elif len(date_str) == 8 and date_str.isdigit():
                    date_obj = datetime.strptime(date_str, "%Y%m%d")
                    year = date_obj.year
                    doy = date_obj.timetuple().tm_yday
                    date_obj = datetime.strptime(f"{year}{doy:03d}", "%Y%j")
                    yyyy_doy = int(date_obj.strftime("%Y%j"))
            except ValueError:
                print(f'No correctly formatted date string found in {file_name}')
        extracted_dates.append(yyyy_doy)
    print('Image dates:', extracted_dates)
    return extracted_dates


def fetch_model(config):
    model, _, _, _, _ = resume_pretrained_network(config=config)
    return model


def fetch_datasets(config):
    dataset_infer = CSVDataset(mode='inference', data_file=config['csv_inference_file'], config=config,
                               transforms=transforms.Compose([
                                   DataSampleCreator(config)
                               ]), sampler=Sampler_NoShuffle())

    return dataset_infer


def fetch_datasets_geotiff(config):
    csv_parser = fetch_parser_geotiff(config, 'inference')
    csv_file = config['csv_inference_file_geotiffs']
    paths_included_in_csvs = False if "paths_included_in_csvs" not in config else config["paths_included_in_csvs"]
    if not os.path.exists(csv_file):
        raise ValueError("For this kind of inference, you need to create csv file yourself!")
    csv_parser.read_data(csv_file, paths_included_in_csvs)
    infer_dataset = csv_parser.data_base
    return infer_dataset


def fetch_dataloader(config, dataset_infer, use_gpu):
    custom_collation = CustomCollation(config)
    n_workers = 12 if "data_loader_num_workers" not in config else config["data_loader_num_workers"]
    infer_dataloader = DataLoader(dataset_infer, num_workers=n_workers, collate_fn=custom_collation,
                                  batch_size=config['batch_size'], shuffle=False, drop_last=False, pin_memory=use_gpu)
    return infer_dataloader


def get_stack_arg(file_ids, key, h, w, arr_type, arr_default, target_transform, window, target_crs, direction):
    arr_stack = []
    resampling_method = WarpResampling.nearest

    for fid, paths in file_ids.items():
        with rasterio.open(paths[key]) as src:
            src_data = np.ones((h, w), dtype=arr_type) * arr_default
            reproject(
                source=rasterio.band(src, 1),
                destination=src_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=window_transform(window, target_transform),
                dst_crs=target_crs,
                resampling=resampling_method,
                src_nodata=arr_default,
                dst_nodata=arr_default,
                init_dest_nodata=False
            )
            arr_stack.append(src_data)

    arr_stack = np.stack(arr_stack, axis=0)
    if direction == 'max':
        arg_indices = np.argmax(arr_stack, axis=0)
    elif direction == 'min':
        arg_indices = np.argmin(arr_stack, axis=0)
    elif direction == 'majority':
        maj_values, _ = array_mode(arr_stack, axis=0, keepdims=False)
        matches = (arr_stack == maj_values)
        majority_first_idx = np.argmax(matches, axis=0)  # index of first match
        maj_is_ignore = (maj_values == arr_default)
        nonmaj_mask = ~matches
        first_nonmaj_idx = np.argmax(nonmaj_mask, axis=0)
        arg_indices = np.where(maj_is_ignore, first_nonmaj_idx, majority_first_idx)
    else:
        raise ValueError(f"Unknown direction {direction}!")
    valid_mask = (arr_stack != arr_default)
    count_valid = valid_mask.sum(axis=0)
    return arg_indices, count_valid


def find_unique_inputs_from_csv(csv_path, file_ids, concatenate_char, dict_input_paths=None):
    if dict_input_paths is None:
        dict_input_paths = {}
    df = pd.read_csv(csv_path)
    input_cols = sorted(
        (col for col in df.columns if col.startswith('input_file_')), key=lambda x: int(x.split('_')[-1]))
    input_cols_before = input_cols[:-1]
    input_cols_after = input_cols[-1:]

    def catch_files(colums):
        tmp = set()
        for col in colums:
            for filepath in df[col].dropna():  # ignore NaNs
                filepath = filepath.strip()
                if filepath and filepath.endswith('.tif'):
                    if os.path.isfile(filepath):
                        tmp.add(filepath)
                    else:
                        if dict_input_paths:
                            tmp.add(os.path.join(dict_input_paths[col], filepath))
        return tmp

    file_paths_before = catch_files(input_cols_before)
    file_paths_after = catch_files(input_cols_after)

    def catch_names(filepaths):
        tmp = {}
        for fid in file_ids.keys():
            tmp[fid] = []
            fid_parts = fid.split(concatenate_char)
            for part in fid_parts:
                matches = [path for path in filepaths if part and part in path]
                if matches:
                    tmp[fid].append(matches[0])

        return tmp

    file_ids_inputs_before = catch_names(file_paths_before)
    file_ids_inputs_after = catch_names(file_paths_after)
    file_ids_inputs = {'before_input': file_ids_inputs_before, 'after_input': file_ids_inputs_after}
    return file_ids_inputs


def find_yeardoy_in_filenames(yeardoy, image_file_list, date_regex):
    yeardoy_str = convert_yeardoy_to_string(yeardoy, date_regex)
    for file_path in image_file_list:
        filename = os.path.basename(file_path)
        if yeardoy_str in filename:
            return file_path
    return None


def remove_all_file_versions(tif_file):
    base_name = os.path.splitext(tif_file)[0]
    matching_files = glob.glob(base_name + ".*")
    for f in matching_files:
        if not f.lower().endswith('.csv'):
            try:
                os.unlink(f)
            except FileNotFoundError:
                print(f"File not found (already deleted?): {f}")
            except Exception as e:
                print(f"Error deleting {f}: {e}")


def convert_fmask_bitwise2int(fmask_array):
    fmask_bitfield = [1, 2, 3, 4, 6]
    # source: https://hrodmn.dev/posts/hls/#cloudshadow-masking
    result = np.zeros_like(fmask_array, dtype=np.uint8)
    for fid in fmask_bitfield:
        if fid != 6:
            mask = ((fmask_array >> fid) & 1).astype(bool)
        else:
            mask = (((fmask_array >> fid) & 3) == 3).astype(bool)
        result[mask] = fid
    return result


def save_sub_fmask_from_input(fr, tiles_dir_input, file_name_without_extension, geotiff_database_infer_index, fmask_in_bit):
    sfr = str(fr)
    master_image_path = os.path.split(geotiff_database_infer_index['input_file_' + sfr])[0]
    master_image_name = os.path.split(geotiff_database_infer_index['input_file_' + sfr])[1]
    master_image_name = "fmask_" + master_image_name
    file_name_without_extension_mask = "quality_fmasks_" + file_name_without_extension
    if os.path.exists(os.path.join(master_image_path, master_image_name)):
        with (rasterio.open(os.path.join(tiles_dir_input[fr], file_name_without_extension + '.tif')) as src,
              rasterio.open(os.path.join(master_image_path, master_image_name)) as msk):
            dst_crs = src.crs
            dst_transform = src.transform
            dst_height = src.height
            dst_width = src.width

            master_data = msk.read(1)
            if not fmask_in_bit:
                dst_dtype = msk.dtypes[0]
                dst_nodata = msk.nodata
            else:
                dst_dtype = 'uint8'
                dst_nodata = 255
                master_data = convert_fmask_bitwise2int(master_data)

            mask_dst = np.full((dst_height, dst_width), dst_nodata, dtype=dst_dtype)
            reproject(
                source=master_data,
                destination=mask_dst,
                src_transform=msk.transform,
                src_crs=msk.crs,
                src_nodata=msk.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=dst_nodata,
                resampling=WarpResampling.nearest
            )
            profile = src.profile.copy()
            profile.update(
                count=1,
                dtype=dst_dtype,
                nodata=dst_nodata
            )

            with rasterio.open(os.path.join(tiles_dir_input[fr], file_name_without_extension_mask + '.tif'),
                               "w", **profile) as dst:
                dst.write(mask_dst, 1)
    else:
        # print(f"no explicit quality fmask file was found for {master_image_name}; it will be assumed the image is clean")
        with rasterio.open(os.path.join(tiles_dir_input[fr], file_name_without_extension + '.tif')) as src:
            profile = src.profile.copy()
            profile.update(
                dtype="uint8",
                driver="GTiff",
                count=1,
                nodata=255
            )
            with rasterio.open(os.path.join(tiles_dir_input[fr], file_name_without_extension_mask + '.tif'),
                               "w", **profile) as dst:
                mask_dst = np.full((src.height, src.width), 0, dtype=np.uint8) # set this to 0 as if there is no cloud!
                dst.write(mask_dst, 1)


def create_sub_geot_from_tfw(file_path, patch_size, patch_stride, upsample_cosine_map_factor=-1):
    with open(file_path, 'r') as file:
        geot = file.readlines()
    x_res = float(geot[0].replace("\n", ""))
    y_res = float(geot[3].replace("\n", ""))
    x_org = float(geot[4].replace("\n", "")) - x_res / 2.0
    y_org = float(geot[5].replace("\n", "")) - y_res / 2.0
    x_res_cos = x_res * patch_stride
    y_res_cos = y_res * patch_stride
    if upsample_cosine_map_factor > 0:
        x_res_cos = x_res_cos / upsample_cosine_map_factor
        y_res_cos = y_res_cos / upsample_cosine_map_factor
    if patch_stride == 1:
        x_org_cos = int(patch_size / 2 - 1) * x_res + x_org
        y_org_cos = int(patch_size / 2 - 1) * y_res + y_org
    else:
        x_org_cos = x_org
        y_org_cos = y_org
    geot[0] = str(x_res_cos) + '\n'
    geot[3] = str(y_res_cos) + '\n'
    geot[4] = str(x_org_cos + x_res_cos / 2.0) + '\n'
    geot[5] = str(y_org_cos + y_res_cos / 2.0) + '\n'
    return geot, x_org, y_org, x_res, y_res


def create_sub_infer_dataset(config, column_name):
    using_gpu = True if "use_gpu" not in config else config["use_gpu"]
    num_frames = config['num_frames']
    columns_to_save = {}
    for fr in range(num_frames):
        columns_to_save.update(
            {'input_file_' + str(fr): [], 'target_file_' + str(fr): []})  # The name of the columns

    for i in range(len(column_name['input_file_0'])):
        f0 = column_name['input_file_0'][i]
        pt = [i for i, letter in enumerate(f0) if letter == '_']
        p0 = f0[pt[-2]:]
        count_exists = 1
        names = [None] * num_frames
        names[0] = f0
        for fr in range(1, num_frames):
            sfr = str(fr)
            for j in range(len(column_name['input_file_' + sfr])):
                fj = column_name['input_file_' + sfr][j]
                pt = [i for i, letter in enumerate(fj) if letter == '_']
                pj = fj[pt[-2]:]
                if p0 == pj:
                    count_exists += 1
                    names[fr] = fj
                    break

        if count_exists == num_frames:
            for fr in range(num_frames):
                sfr = str(fr)
                columns_to_save['input_file_' + sfr].append(names[fr])
                columns_to_save['target_file_' + sfr].append(None)

    df = pd.DataFrame.from_dict(columns_to_save)
    df.to_csv(config['csv_inference_file'])

    dataset_infer = fetch_datasets(config)
    dataloader_infer = fetch_dataloader(config, dataset_infer, using_gpu)
    print('Total number of inference tiles: {}'.format(len(dataset_infer)))
    return dataloader_infer, dataset_infer


def save_sub_fmask_pooled(config, batch_size, batch_image_names, keep_as_is=False, upsample_cosine_map_factor=-1):
    patch_stride = config['model_params']['patch_stride'] if 'patch_stride' in config['model_params'] else \
        config['model_params']['patch_size']
    patch_size = config['model_params']['patch_size']
    num_frames = config['num_frames']
    compression_type = config["tif_compression"]
    for b in range(batch_size):
        for fr in range(num_frames):
            fmask_filepath = os.path.split(batch_image_names[fr][b])[0]
            fmask_filename = "quality_fmasks_" + \
                             os.path.splitext(os.path.split(batch_image_names[fr][b])[1])[0]
            fmask, _, _, _ = read_geotiff(os.path.join(fmask_filepath, fmask_filename + '.tif'))
            if not keep_as_is:
                fmask = Func.max_pool2d(torch.from_numpy(fmask).float().unsqueeze(0).unsqueeze(0),
                                        kernel_size=patch_size, stride=patch_stride)
                if upsample_cosine_map_factor > 0:
                    fmask = Func.interpolate(fmask, scale_factor=upsample_cosine_map_factor, mode='nearest')
                write_geotiff('unknown', None, '', fmask.squeeze(0).squeeze(0).numpy(),
                              config['inference_save_folder_frame_' + str(fr)],
                              fmask_filename, alpha_im=None,
                              data_type=gdal.GDT_Byte, save_tfw=False,
                              no_data_values=[255], compression=compression_type)
            else:
                write_geotiff('unknown', None, '', fmask,
                              config['inference_save_folder_frame_' + str(fr)],
                              fmask_filename, alpha_im=None,
                              data_type=gdal.GDT_Byte, save_tfw=False,
                              no_data_values=[255], compression=compression_type)


def merge_fmasks_pooled(config, inputs_for_ints, which_before_epoch):
    num_frames = config['num_frames']

    fmasks = []
    for subfr in range(num_frames):
        fmpath = config['inference_save_folder_geotiff_frame_' + str(subfr)]
        fmname = 'fmask_' + inputs_for_ints[subfr] + '.tif'
        fmask_path = os.path.join(fmpath, fmname)
        fmask, _, _, _ = read_geotiff(fmask_path)
        remove_all_file_versions(fmask_path)
        fmasks.append(fmask)
    fmasks = np.stack(fmasks, axis=0)
    result = np.ones(which_before_epoch.shape, dtype=np.uint8) * 255
    valid_mask = (which_before_epoch != -100)
    result[valid_mask] = fmasks[:-1, :, :][which_before_epoch[valid_mask],
    np.where(valid_mask)[0],
    np.where(valid_mask)[1]]
    result = np.maximum(result, fmasks[-1, :, :])
    del fmasks
    return result


def merge_shifted_results(config, files_list, save_path, name_attachment, given_prefix_list):
    num_frames = config["num_frames"]
    tile_size = 1024
    files_delete = True
    image_nodata = config["model_params"]["no_data"]
    mask_nodata = config["model_params"]["ignore_index"]
    image_band_counts = len(config["model_params"]["bands"])
    embed_counts_total = config["model_params"]["embed_dim"]
    embeddings = config["feature_maps"]["embeddings"]
    if not embeddings:
        embed_counts = embed_counts_total
    else:
        embed_counts = len(embeddings)
    compression_type = config["tif_compression"]

    pre_fixes = ['cosine_map', 'before_date', 'after_date', 'quality_fmask', 'predicted_mask_0', 'infer_0', 'feature_maps_0']
    pref_fixes_nodatas = [-100.0, -100, -100, 255, mask_nodata, image_nodata, -100.0]
    pre_fixes_counts = [1, 1, 1, 1, 1, image_band_counts, embed_counts]
    pref_fixes_dtypes = ['float32', 'int32', 'int32', 'uint8', 'int16', 'float32', 'float32']
    for fr in range(1, num_frames):
        pre_fixes.append('predicted_mask_' + str(fr))
        pref_fixes_nodatas.append(mask_nodata)
        pre_fixes_counts.append(1)
        pref_fixes_dtypes.append('int16')
        pre_fixes.append('infer_' + str(fr))
        pref_fixes_nodatas.append(image_nodata)
        pre_fixes_counts.append(image_band_counts)
        pref_fixes_dtypes.append('float32')
        pre_fixes.append('feature_maps_' + str(fr))
        pref_fixes_nodatas.append(-100.0)
        pre_fixes_counts.append(embed_counts)
        pref_fixes_dtypes.append('float32')

    prefix_found = {}
    for prfx in pre_fixes:
        prefix_found[prfx] = False
    for prfx in given_prefix_list:
        prefix_found[prfx] = True
    ref_prefix = given_prefix_list[0]

    file_ids = {}
    for file_item in files_list:
        fname = os.path.split(file_item)[1]
        fid = None
        for prfx in pre_fixes:
            if fname.startswith(prfx) and prefix_found[prfx]:
                kind = prfx
                parts = fname.replace(prfx, '')
                fid = parts.replace('.tif', '')
                break
        if fid is not None:
            if fid not in file_ids:
                file_ids[fid] = {}
            file_ids[fid][kind] = file_item

    ref_files = [v[ref_prefix] for v in file_ids.values()]
    with rasterio.open(ref_files[0]) as src:
        target_crs = src.crs
        xres, yres = src.res
        target_res = abs(xres)

    bounds_list = []
    for cosine_path in ref_files:
        with rasterio.open(cosine_path) as src:
            bounds = src.bounds
            src_crs = src.crs
            dst_bounds = rasterio.warp.transform_bounds(src_crs, target_crs, *bounds)
            bounds_list.append(dst_bounds)
    minxs, minys, maxxs, maxys = zip(*bounds_list)
    full_bounds = (min(minxs), min(minys), max(maxxs), max(maxys))

    target_transform = from_origin(full_bounds[0], full_bounds[3], target_res, target_res)
    target_width = int((full_bounds[2] - full_bounds[0]) / target_res) + tile_size
    target_width_original = int((full_bounds[2] - full_bounds[0]) / target_res) + 1
    target_height = int((full_bounds[3] - full_bounds[1]) / target_res) + tile_size
    target_height_original = int((full_bounds[2] - full_bounds[0]) / target_res) + 1

    out_dict = {}
    for ipfx, prfx in enumerate(pre_fixes):
        if prefix_found[prfx]:
            out_dict[prfx] = [rasterio.open(os.path.join(save_path, prfx + name_attachment + '.tif'),
                                            'w', driver='GTiff',
                                            height=target_height_original, width=target_width_original,
                                            count=pre_fixes_counts[ipfx], dtype=pref_fixes_dtypes[ipfx],
                                            crs=target_crs, transform=target_transform,
                                            nodata=pref_fixes_nodatas[ipfx], compress=compression_type, tiled=True,
                                            blockxsize=tile_size, blockysize=tile_size, BIGTIFF="YES"),
                              pre_fixes_counts[ipfx]]

    #TODO: make these nested loops parallel or use rasterio VRT creation to avoid for loops all together
    for row_start in tqdm(range(0, target_height, tile_size)):
        if row_start >= target_height_original:
            continue
        for col_start in range(0, target_width, tile_size):
            if col_start >= target_width_original:
                continue
            w = min(tile_size, target_width_original - col_start)
            h = min(tile_size, target_height_original - row_start)
            window = Window(col_start, row_start, w, h)

            if ref_prefix == 'cosine_map':
                arg_indices, persistence_data = get_stack_arg(file_ids, ref_prefix, h, w, 'float32', -100.0,
                                                              target_transform, window, target_crs, 'max')
            elif ref_prefix ==  'quality_fmask':
                arg_indices, persistence_data = get_stack_arg(file_ids, ref_prefix, h, w, 'uint8', 255,
                                                              target_transform, window, target_crs, 'min')
            elif 'predicted_mask' in ref_prefix:
                arg_indices, persistence_data = get_stack_arg(file_ids, ref_prefix, h, w, 'int16', mask_nodata,
                                                              target_transform, window, target_crs, 'majority')
            elif 'infer' in ref_prefix:
                arg_indices, persistence_data = get_stack_arg(file_ids, ref_prefix, h, w, 'float32', image_nodata,
                                                              target_transform, window, target_crs, 'max')
            elif 'feature_maps' in ref_prefix:
                arg_indices, persistence_data = get_stack_arg(file_ids, ref_prefix, h, w, 'float32', -100.0,
                                                              target_transform, window, target_crs, 'min')

            to_go_dict = {}
            traversable_keys = []
            for ipfx, prfx in enumerate(pre_fixes):
                if prefix_found[prfx]:
                    traversable_keys.append(prfx)
                    ones_shape = (h, w)
                    if pre_fixes_counts[ipfx] > 1:
                        ones_shape = (pre_fixes_counts[ipfx], h, w)
                    to_go_dict[prfx] = [np.ones(ones_shape, dtype=pref_fixes_dtypes[ipfx]) * pref_fixes_nodatas[ipfx],
                                        pref_fixes_dtypes[ipfx], pref_fixes_nodatas[ipfx]]

            for idx, fid in enumerate(file_ids.keys()):
                mask = (arg_indices == idx)
                if not np.any(mask):
                    continue
                for layer_type in traversable_keys:
                    if any(x in layer_type for x in ("cosine", "input", "infer", "feature_maps")):
                        resampling_method = WarpResampling.cubic
                    else:
                        resampling_method = WarpResampling.nearest
                    output_mosaic, arr_dtype, arr_default = to_go_dict[layer_type]
                    with rasterio.open(file_ids[fid][layer_type]) as src:
                        band_count = src.count
                        if band_count == 1:
                            src_data = np.ones((h, w), dtype=arr_dtype) * arr_default
                            reproject(
                                source=rasterio.band(src, 1),
                                destination=src_data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=window_transform(window, target_transform),
                                dst_crs=target_crs,
                                resampling=resampling_method,
                                src_nodata=arr_default,
                                dst_nodata=arr_default
                            )
                            output_mosaic[mask] = src_data[mask]
                        else:
                            src_data = np.ones((band_count, h, w), dtype=arr_dtype) * arr_default
                            for b in range(1, band_count + 1):  # bands are 1-indexed in rasterio
                                reproject(
                                    source=rasterio.band(src, b),
                                    destination=src_data[b - 1],  # src_data is 0-indexed
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=window_transform(window, target_transform),
                                    dst_crs=target_crs,
                                    resampling=resampling_method,
                                    src_nodata=arr_default,
                                    dst_nodata=arr_default
                                )
                            output_mosaic[:, mask] = src_data[:, mask]

            for key in out_dict.keys():
                if out_dict[key][1] == 1:
                    out_dict[key][0].write(to_go_dict[key][0], 1, window=window)
                else:
                    out_dict[key][0].write(to_go_dict[key][0], window=window)

    for key in out_dict.keys():
        out_dict[key][0].build_overviews([2, 4, 8, 16], WarpResampling.average)
        out_dict[key][0].update_tags(ns="rio_overview", resampling="average")
        out_dict[key][0].close()

    if files_delete:
        for fid in file_ids.keys():
            for kind in file_ids[fid].keys():
                tif_file = file_ids[fid][kind]
                remove_all_file_versions(tif_file)


def determine_mosaicable_files(config, ignore_prefixes=None):
    num_frames = config["num_frames"]
    out_folders = [config['inference_save_folder_geotiff_frame_' + str(fr)] for fr in range(num_frames)]

    pre_fixes = ['cosine_map', 'before_date', 'after_date', 'quality_fmask', 'before_input', 'after_input',
                 'predicted_mask_0', 'infer_0', 'feature_maps_0']
    for fr in range(1, num_frames):
        pre_fixes.append('predicted_mask_' + str(fr))
        pre_fixes.append('infer_' + str(fr))
        pre_fixes.append('feature_maps_' + str(fr))

    if ignore_prefixes is None:
        ignore_prefixes = []

    prefix_found = {}
    for prfx in pre_fixes:
        prefix_found[prfx] = False

    list_of_files = []
    for fr in range(num_frames):
        for fname in os.listdir(out_folders[fr]):
            if fname.endswith('.tif'):
                fid = None
                for prfx in pre_fixes:
                    if prfx not in ignore_prefixes:
                        if fname.startswith(prfx):
                            prefix_found[prfx] = True
                            parts = fname.replace(prfx, '')
                            fid = parts.replace('.tif', '')
                            break
                if fid is not None:
                    list_of_files.append(os.path.join(out_folders[fr], fname))
    return list_of_files, prefix_found


def planaura_mosaic_geotiff(config, reference_prefix=None, ignore_prefixes=None):
    num_frames = config["num_frames"]
    out_postfix = '' if "mosaic_save_postfix" not in config["mosaic_params"] else config["mosaic_params"]["mosaic_save_postfix"]
    if out_postfix:
        out_postfix = '_' + out_postfix
    out_folders = [config['inference_save_folder_geotiff_frame_' + str(fr)] for fr in range(num_frames)]
    tile_size = 1024
    target_crs = config["mosaic_params"]["target_crs"]
    target_res = config["mosaic_params"]["target_resolution"]
    files_delete = config["mosaic_params"]["delete_residues"]
    image_nodata = config["model_params"]["no_data"]
    mask_nodata = config["model_params"]["ignore_index"]
    embed_counts_total = config["model_params"]["embed_dim"]
    embeddings = config["feature_maps"]["embeddings"]
    if not embeddings:
        embed_counts = embed_counts_total
    else:
        embed_counts = len(embeddings)
    image_band_counts = len(config["model_params"]["bands"])
    date_regex = config["change_map"]["date_regex"]
    concatenate_char = config["concatenate_char"]
    compression_type = config["tif_compression"]

    pre_fixes = ['cosine_map', 'before_date', 'after_date', 'quality_fmask', 'before_input', 'after_input',
                 'predicted_mask_0', 'infer_0', 'feature_maps_0']
    pref_fixes_nodatas = [-100.0, -100, -100, 255, image_nodata, image_nodata, mask_nodata, image_nodata, -100.0]
    pre_fixes_counts = [1, 1, 1, 1, image_band_counts, image_band_counts, 1, image_band_counts, embed_counts]
    pref_fixes_dtypes = ['float32', 'int32', 'int32', 'uint8', 'int16', 'int16', 'int16', 'float32', 'float32']
    for fr in range(1, num_frames):
        pre_fixes.append('predicted_mask_' + str(fr))
        pref_fixes_nodatas.append(mask_nodata)
        pre_fixes_counts.append(1)
        pref_fixes_dtypes.append('int16')
        pre_fixes.append('infer_' + str(fr))
        pref_fixes_nodatas.append(image_nodata)
        pre_fixes_counts.append(image_band_counts)
        pref_fixes_dtypes.append('float32')
        pre_fixes.append('feature_maps_' + str(fr))
        pref_fixes_nodatas.append(-100.0)
        pre_fixes_counts.append(embed_counts)
        pref_fixes_dtypes.append('float32')

    if ignore_prefixes is None:
        ignore_prefixes = []

    prefix_found = {}
    for prfx in pre_fixes:
        prefix_found[prfx] = False
    file_ids = {}
    for fr in range(num_frames):
        for fname in os.listdir(out_folders[fr]):
            if fname.endswith('.tif'):
                fid = None
                for prfx in pre_fixes:
                    if prfx not in ignore_prefixes:
                        if fname.startswith(prfx):
                            kind = prfx
                            prefix_found[prfx] = True
                            parts = fname.replace(prfx, '')
                            fid = parts.replace('.tif', '')
                            break
                if fid is not None:
                    if fid not in file_ids:
                        file_ids[fid] = {}
                    file_ids[fid][kind] = os.path.join(out_folders[fr], fname)

    if not file_ids:
        print("No relevant file was found for mosaic creation! Exiting ... ")
        return []

    dict_input_paths = {}
    if not config["paths_included_in_csvs"]:
        for fr in range(num_frames):
            dict_input_paths['input_file_' + str(fr)] = config['inference_input_folder_geotiff_frame_' + str(fr)]
    file_ids_inputs = find_unique_inputs_from_csv(config['csv_inference_file_geotiffs'], file_ids,
                                                  concatenate_char, dict_input_paths=dict_input_paths)

    if reference_prefix is not None:
        if not prefix_found[reference_prefix]:
            reference_prefix = None

    ordered_potential_refs = ['cosine_map', 'quality_fmask']
    special_postfix_char = {'cosine_map': '_c', 'quality_fmask': '_f'}
    for fr in range(num_frames):
        ordered_potential_refs.append('predicted_mask_' + str(fr))
        special_postfix_char['predicted_mask_' + str(fr)] = '_p' + str(fr)
        ordered_potential_refs.append('infer_' + str(fr))
        special_postfix_char['infer_' + str(fr)] = '_i' + str(fr)
        ordered_potential_refs.append('feature_maps_' + str(fr))
        special_postfix_char['feature_maps_' + str(fr)] = '_m' + str(fr)

    if reference_prefix is None:
        for ordered_ref in ordered_potential_refs:
            if prefix_found[ordered_ref]:
                ref_files = [v[ordered_ref] for v in file_ids.values()]
                reference_prefix = ordered_ref
                break
        if reference_prefix is None:
            print("There is no reference map available!")
            return []
    else:
        ref_files = [v[reference_prefix] for v in file_ids.values()]

    print(f"Creating mosaics while optimizing {reference_prefix} !")
    print(f"The file names for these mosaics will end with  {special_postfix_char[reference_prefix]}.tif")
    out_postfix += special_postfix_char[reference_prefix]

    bounds_list = []
    for cosine_path in ref_files:
        with rasterio.open(cosine_path) as src:
            bounds = src.bounds
            src_crs = src.crs
            dst_bounds = rasterio.warp.transform_bounds(src_crs, target_crs, *bounds)
            bounds_list.append(dst_bounds)
    minxs, minys, maxxs, maxys = zip(*bounds_list)
    full_bounds = (min(minxs), min(minys), max(maxxs), max(maxys))

    target_transform = from_origin(full_bounds[0], full_bounds[3], target_res, target_res)
    target_width = int((full_bounds[2] - full_bounds[0]) / target_res) + tile_size
    target_width_original = int((full_bounds[2] - full_bounds[0]) / target_res) + 1
    target_height = int((full_bounds[3] - full_bounds[1]) / target_res) + tile_size
    target_height_original = int((full_bounds[3] - full_bounds[1]) / target_res) + 1

    list_of_created_mosaics = []
    out_dict = {}
    for ipfx, prfx in enumerate(pre_fixes):
        if (prefix_found[prfx] or 'input' in prfx) and prfx not in ignore_prefixes:
            out_dict[prfx] = [rasterio.open(os.path.join(out_folders[0], 'mosaic_' + prfx + out_postfix + '.tif'),
                                            'w', driver='GTiff',
                                            height=target_height_original, width=target_width_original,
                                            count=pre_fixes_counts[ipfx], dtype=pref_fixes_dtypes[ipfx],
                                            crs=target_crs, transform=target_transform,
                                            nodata=pref_fixes_nodatas[ipfx], compress=compression_type, tiled=True,
                                            blockxsize=tile_size, blockysize=tile_size, BIGTIFF="YES"),
                              pre_fixes_counts[ipfx]]
            list_of_created_mosaics.append(prfx)

    out_persistence = rasterio.open(os.path.join(out_folders[0], 'mosaic_persistence' + out_postfix + '.tif'),
                                    'w', driver='GTiff', height=target_height_original, width=target_width_original,
                                    count=1, dtype='uint16', crs=target_crs, transform=target_transform, nodata=0,
                                    compress=compression_type, tiled=True,
                                    blockxsize=tile_size, blockysize=tile_size, BIGTIFF = "YES")

    # TODO: make these nested loops parallel or use rasterio VRT creation to avoid for loops all together
    date_input_found = False
    for row_start in tqdm(range(0, target_height, tile_size)):
        if row_start >= target_height_original:
            continue
        for col_start in range(0, target_width, tile_size):
            if col_start >= target_width_original:
                continue
            w = min(tile_size, target_width_original - col_start)
            h = min(tile_size, target_height_original - row_start)
            window = Window(col_start, row_start, w, h)

            if reference_prefix == 'cosine_map':
                arg_indices, persistence_data = get_stack_arg(file_ids, 'cosine_map', h, w, 'float32', -100.0,
                                                              target_transform, window, target_crs, 'max')
            elif reference_prefix == 'quality_fmask':
                arg_indices, persistence_data = get_stack_arg(file_ids, 'quality_fmask', h, w, 'uint8', 255,
                                                              target_transform, window, target_crs, 'min')
            elif 'predicted_mask' in reference_prefix:
                arg_indices, persistence_data = get_stack_arg(file_ids, reference_prefix, h, w, 'int16', mask_nodata,
                                                              target_transform, window, target_crs, 'majority')
            elif 'infer' in reference_prefix:
                arg_indices, persistence_data = get_stack_arg(file_ids, reference_prefix, h, w, 'float32', image_nodata,
                                                              target_transform, window, target_crs, 'max')
            elif 'feature_maps' in reference_prefix:
                arg_indices, persistence_data = get_stack_arg(file_ids, reference_prefix, h, w, 'float32', -100.0,
                                                              target_transform, window, target_crs, 'min')
            else:
                print("Could not find a reference map!")
                return []

            to_go_dict = {}
            traversable_keys = []
            for ipfx, prfx in enumerate(pre_fixes):
                if (prefix_found[prfx] or 'input' in prfx) and prfx not in ignore_prefixes:
                    if 'input' not in prfx:
                        traversable_keys.append(prfx)
                    ones_shape = (h, w)
                    if pre_fixes_counts[ipfx] > 1:
                        ones_shape = (pre_fixes_counts[ipfx], h, w)
                    to_go_dict[prfx] = [np.ones(ones_shape, dtype=pref_fixes_dtypes[ipfx]) * pref_fixes_nodatas[ipfx],
                                        pref_fixes_dtypes[ipfx], pref_fixes_nodatas[ipfx]]

            for idx, fid in enumerate(file_ids.keys()):
                mask = (arg_indices == idx)
                if not np.any(mask):
                    continue
                for layer_type in traversable_keys:
                    if any(x in layer_type for x in ("cosine", "input", "infer", "feature_maps")):
                        resampling_method = WarpResampling.cubic
                    else:
                        resampling_method = WarpResampling.nearest
                    output_mosaic, arr_dtype, arr_default = to_go_dict[layer_type]
                    with rasterio.open(file_ids[fid][layer_type]) as src:
                        band_count = src.count
                        if band_count == 1:
                            src_data = np.ones((h, w), dtype=arr_dtype) * arr_default
                            reproject(
                                source=rasterio.band(src, 1),
                                destination=src_data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=window_transform(window, target_transform),
                                dst_crs=target_crs,
                                resampling=resampling_method,
                                src_nodata=arr_default,
                                dst_nodata=arr_default
                            )
                            output_mosaic[mask] = src_data[mask]
                            if "date" in layer_type:
                                input_layer_type = layer_type.replace("date", "input")
                                if input_layer_type in ignore_prefixes:
                                    continue
                                output_mosaic_input, arr_dtype_input, arr_default_input = to_go_dict[input_layer_type]
                                unique_yeardoys = np.unique(src_data[mask])
                                unique_yeardoys = unique_yeardoys[unique_yeardoys != arr_default]
                                for yeardoy in unique_yeardoys:
                                    tif_file = find_yeardoy_in_filenames(yeardoy,
                                                                         file_ids_inputs[input_layer_type][fid],
                                                                         date_regex)
                                    if tif_file is None:
                                        print(f"Warning: No matching file for yeardoy {yeardoy}")
                                    else:
                                        date_input_found = True
                                        with rasterio.open(tif_file) as src_img:
                                            matching_pixels = (src_data == yeardoy) & mask
                                            src_image_data = np.ones((image_band_counts, h, w),
                                                                     dtype=arr_dtype_input) * arr_default_input
                                            for b in range(1, image_band_counts + 1):
                                                reproject(
                                                    source=rasterio.band(src_img, b),
                                                    destination=src_image_data[b - 1],
                                                    src_transform=src_img.transform,
                                                    src_crs=src_img.crs,
                                                    dst_transform=window_transform(window, target_transform),
                                                    dst_crs=target_crs,
                                                    resampling=WarpResampling.cubic,
                                                    src_nodata=arr_default,
                                                    dst_nodata=arr_default
                                                )
                                            output_mosaic_input[:, matching_pixels] = src_image_data[:, matching_pixels]
                        else:
                            src_data = np.ones((band_count, h, w), dtype=arr_dtype) * arr_default
                            for b in range(1, band_count + 1):  # bands are 1-indexed in rasterio
                                reproject(
                                    source=rasterio.band(src, b),
                                    destination=src_data[b - 1],  # src_data is 0-indexed
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=window_transform(window, target_transform),
                                    dst_crs=target_crs,
                                    resampling=resampling_method,
                                    src_nodata=arr_default,
                                    dst_nodata=arr_default
                                )
                            output_mosaic[:, mask] = src_data[:, mask]

            for key in out_dict.keys():
                if 'input' in key:
                    if not date_input_found:
                        continue
                if out_dict[key][1] == 1:
                    out_dict[key][0].write(to_go_dict[key][0], 1, window=window)
                else:
                    out_dict[key][0].write(to_go_dict[key][0], window=window)
            out_persistence.write(persistence_data, 1, window=window)

    for key in out_dict.keys():
        out_dict[key][0].build_overviews([2, 4, 8, 16], WarpResampling.average)
        out_dict[key][0].update_tags(ns="rio_overview", resampling="average")
        out_dict[key][0].close()

    for key in out_dict.keys():
        if 'input' in key:
            if not date_input_found:
                removing_path = os.path.join(out_folders[0], 'mosaic_' + key + out_postfix + '.tif')
                list_of_created_mosaics.remove(key)
                if os.path.exists(removing_path):
                    os.remove(removing_path)

    out_persistence.build_overviews([2, 4, 8, 16], WarpResampling.average)
    out_persistence.update_tags(ns="rio_overview", resampling="average")
    out_persistence.close()

    if files_delete:
        for fid in file_ids.keys():
            for kind in file_ids[fid].keys():
                tif_file = file_ids[fid][kind]
                remove_all_file_versions(tif_file)

    return list_of_created_mosaics


def delete_temp_inferences(num_frames, out_folders, tiles_dir_input):
    for fr in range(num_frames):
        # remove all temporary files
        for filename in os.listdir(out_folders[fr]):
            file_path = os.path.join(out_folders[fr], filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

        for filename in os.listdir(tiles_dir_input[fr]):
            file_path = os.path.join(tiles_dir_input[fr], filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

def planaura_infer_geotiff(config):
    using_gpu = config["use_gpu"]
    if using_gpu and not torch.cuda.is_available():
        using_gpu = False
    using_multi_gpu = config["use_multi_gpu"]
    if using_multi_gpu and not torch.cuda.is_available():
        using_multi_gpu = False
    config["use_xarray"] = False
    num_inferences = config["num_predictions"]
    calculating_cosine_similarity = config["change_map"]["return"]
    output_feature_maps = config["feature_maps"]["return"]
    write_as_im_feature_maps = config["feature_maps"]["write_as_image"]
    device = 'cuda' if torch.cuda.is_available() and using_gpu else 'cpu'
    if device == 'cpu':
        using_autocast_float16 = False
    else:
        using_autocast_float16 = config["autocast_float16"]
    compression_type = config["tif_compression"]
    saving_reconstructed_images = config["save_reconstructed_images"]
    saving_dates = config["change_map"]["save_dates_layer"]
    date_regex = config["change_map"]["date_regex"]
    saving_fmask = config["change_map"]["save_fmask_layer"]
    fmask_in_bit = config["change_map"]["fmask_in_bit"]
    if saving_fmask:
        saving_dates = True
    concatenate_char = config["concatenate_char"]

    model = fetch_model(config)
    if using_gpu:
        model = model.cuda()
    if using_multi_gpu:
        model_multi_gpu = torch.nn.DataParallel(model).cuda()
        model = model_multi_gpu.module

    geotiff_database_infer = fetch_datasets_geotiff(config)
    original_paths_included_in_csvs = copy.deepcopy(config["paths_included_in_csvs"])
    config["paths_included_in_csvs"] = False  # do this after the geotiff dataset is read, for the temporary ones!

    patch_stride = config['model_params']['patch_stride'] if 'patch_stride' in config['model_params'] else \
        config['model_params']['patch_size']
    upsample_cosine_map = config["change_map"]["upsample_cosine_map"]
    upsample_cosine_map_factor = -1.0
    if upsample_cosine_map and patch_stride > 1:
        upsample_cosine_map_factor = patch_stride
    merge_stride = patch_stride
    if upsample_cosine_map_factor > 0:
        merge_stride = 1
    print(f"merge_stride: {merge_stride}")
    print(f"upsample_cosine_map_factor: {upsample_cosine_map_factor}")
    patch_size = config['model_params']['patch_size']
    num_frames = config['num_frames']
    crop_size_width_base = config['model_params']['img_size']
    crop_size_height_base = config['model_params']['img_size']
    crop_overlap_percentages = [50]
    down_size_factors = [1]
    middle_percentages = [100]
    random_choice_prob = 0.2
    cut_data_portion = config['minimum_valid_percentage']
    resolution_threshold = np.inf
    discard_beyond_three_bands = False
    replace_unknown_nodata_zero = True
    save_tfw = True

    INPUT_TARGET = False
    single_image = False

    for index in range(len(geotiff_database_infer)):
        start_time_index = time.time()
        prediction_shifts = [64 * n for n in range(num_inferences)]
        geotiff_database_infer_index = geotiff_database_infer[index]
        name_attachment_wo_shift = ""
        for subfr in range(num_frames):
            name_attachment_wo_shift += concatenate_char
            name_attachment_wo_shift += \
                os.path.splitext(
                    os.path.split(geotiff_database_infer_index['input_file_' + str(subfr)])[1])[0]
        list_of_files_to_mode = []
        for prediction_shift in prediction_shifts:
            images_path_input = []
            images_path_target = [None] * num_frames
            image_names_input = []
            for i in range(num_frames):
                image_names_input.append([])
            image_names_target = [] * num_frames
            for i in range(num_frames):
                image_names_target.append([None])
            tiles_dir_input = []
            tiles_dir_target = [None] * num_frames
            for fr in range(num_frames):
                sfr = str(fr)
                input_folder = config['inference_input_folder_frame_' + sfr]
                for filename in os.listdir(input_folder):
                    file_path = os.path.join(input_folder, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                images_path_input.append(os.path.split(geotiff_database_infer_index['input_file_' + sfr])[0])
                image_names_input[fr].append(os.path.split(geotiff_database_infer_index['input_file_' + sfr])[1])
                tiles_dir_input.append(input_folder)

            success_tile = tile_geotiffs(num_frames, random_choice_prob, cut_data_portion, resolution_threshold,
                                         discard_beyond_three_bands,
                                         replace_unknown_nodata_zero, save_tfw, INPUT_TARGET, single_image, crop_size_width_base,
                                         crop_size_height_base,
                                         crop_overlap_percentages=crop_overlap_percentages, down_size_factors=down_size_factors,
                                         middle_percentages=middle_percentages,
                                         tiles_dir_input=tiles_dir_input, tiles_dir_target=tiles_dir_target,
                                         images_path_input=images_path_input,
                                         images_path_target=images_path_target, image_names_input=image_names_input,
                                         image_names_target=image_names_target, ensure_all_image=True,
                                         start_point_shift=prediction_shift)
            if not success_tile:
                continue
            if os.path.exists(config['csv_inference_file']):
                os.unlink(config['csv_inference_file'])

            out_folders = []
            column_name = {}
            offsets_scales_dict = {}
            for fr in range(num_frames):
                column_name.update({'input_file_' + str(fr): []})  # The name of the columns
            for fr in range(num_frames):
                sfr = str(fr)
                output_folder = config['inference_save_folder_frame_' + sfr]
                out_folders.append(output_folder)
                for filename in os.listdir(output_folder):
                    file_path = os.path.join(output_folder, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)

                for filename in os.listdir(tiles_dir_input[fr]):
                    file_path = os.path.join(tiles_dir_input[fr], filename)
                    file_name_without_extension = os.path.splitext(filename)[0]
                    if os.path.isfile(file_path):
                        if file_path.lower().endswith('tfw'):
                            column_name['input_file_' + sfr].append(file_name_without_extension + '.tif')
                            shutil.copy(file_path, os.path.join(output_folder, filename))
                            if saving_fmask:
                                save_sub_fmask_from_input(fr, tiles_dir_input, file_name_without_extension,
                                                          geotiff_database_infer_index, fmask_in_bit)
                            geot, x_org_feat, y_org_feat, x_res_feat, y_res_feat = create_sub_geot_from_tfw(file_path,
                                                                                                            patch_size,
                                                                                                            patch_stride,
                                                                                                            upsample_cosine_map_factor)

                            if output_feature_maps:
                                offsets_scales_dict['feats_' + str(fr) + '_' + file_name_without_extension] = [x_org_feat,
                                                                                                               y_org_feat,
                                                                                                               x_res_feat,
                                                                                                               y_res_feat]
                                if write_as_im_feature_maps:
                                    file_name_feat = 'feats_' + str(fr) + '_' + filename
                                    with open(os.path.join(output_folder, file_name_feat), 'w') as file:
                                        file.writelines(geot)
                            if calculating_cosine_similarity:
                                file_name_cosine = 'cosines_' + filename
                                if saving_dates:
                                    file_name_before = 'before_epochs_' + filename
                                if fr == 0:
                                    with open(os.path.join(output_folder, file_name_cosine), 'w') as file:
                                        file.writelines(geot)
                                    if saving_dates:
                                        with open(os.path.join(output_folder, file_name_before), 'w') as file:
                                            file.writelines(geot)
                                if saving_fmask:
                                    file_name_fmask = 'quality_fmasks_' + filename
                                    with open(os.path.join(output_folder, file_name_fmask), 'w') as file:
                                        file.writelines(geot)

            dataloader_infer, dataset_infer = create_sub_infer_dataset(config, column_name)
            if not dataset_infer:
                print('... Moving on to next geotiffs!')
                delete_temp_inferences(num_frames, out_folders, tiles_dir_input)
                continue
            model.prepare_to_infer()

            if not using_multi_gpu:
                model.eval()
            else:
                model_multi_gpu.eval()

            mega_master_file = None
            with torch.no_grad():
                for iter_num, data in enumerate(dataloader_infer):
                    model_device = model.device_()
                    input_img_batch = data['img_input']
                    batch_size = input_img_batch.shape[0]
                    batch_image_names = [data['input_file_0']]
                    if iter_num == 0:
                        mega_master_file = data['input_file_0'][0]
                    for fr in range(1, num_frames):
                        batch_image_names.append(data['input_file_' + str(fr)])

                    if saving_fmask:
                        if calculating_cosine_similarity and model.is_reconstruction:
                            save_sub_fmask_pooled(config, batch_size, batch_image_names, keep_as_is=False,
                                                  upsample_cosine_map_factor=upsample_cosine_map_factor)

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
                    if model.is_reconstruction:
                        write_reconstructed_images(config, predicted_img_batch, batch_image_names,
                                                   cosine_maps=cosine_maps[0],
                                                   calculate_cosine_similarity=calculating_cosine_similarity,
                                                   which_before_epochs=cosine_maps[1] if saving_dates else None,
                                                   upsample_cosine_map_factor=upsample_cosine_map_factor)
                        if output_feature_maps and prediction_shift == 0:
                            write_feature_maps(config, feat_maps, batch_image_names,
                                               write_as_im_feature_maps, upsample_cosine_map_factor)
                    else:
                        print("nothing implemented yet for when model is not is_reconstruction")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    del predicted_img_batch, input_img_batch, cosine_maps

            fifty_overlap = True
            crop_width = config['model_params']['img_size']
            recrop_width = int(crop_width / 2)
            c_start = int(0.25 * crop_width)
            crop_height = config['model_params']['img_size']
            recrop_height = int(crop_height / 2)
            r_start = int(0.25 * crop_height)

            if crop_width % 4 != 0 or crop_height % 4 != 0:
                print('Warning, merging result might be incorrect! Your image sizes should be divisible by 4')

            name_attachment = ""
            for subfr in range(num_frames):
                name_attachment += concatenate_char
                name_attachment += \
                    os.path.splitext(
                        os.path.split(geotiff_database_infer_index['input_file_' + str(subfr)])[1])[0]
            if num_inferences > 1:
                name_attachment += concatenate_char
                name_attachment += str(prediction_shift)
            if model.is_reconstruction:
                for fr in range(num_frames):
                    sfr = str(fr)
                    images_path = out_folders[fr]
                    if saving_reconstructed_images:
                        infer_filename = 'infer_' + sfr + name_attachment + '.tif'
                        save_full_path = os.path.join(config['inference_save_folder_geotiff_frame_' + sfr], infer_filename)
                        list_of_files_to_mode.append(save_full_path)
                        merge_geotiffs(fifty_overlap, recrop_height, recrop_width, r_start, c_start,
                                       mega_master_file,
                                       images_path, save_full_path,
                                       ignore_prefix=['cosines_', 'before_epochs_', 'quality_fmasks_', 'feats_'],
                                       keep_border=True, compression=compression_type)
                    if calculating_cosine_similarity and saving_fmask:
                        save_full_path = os.path.join(config['inference_save_folder_geotiff_frame_' + sfr],
                                                      'fmask_' +
                                                      os.path.split(geotiff_database_infer_index['input_file_' + sfr])[1])
                        merge_geotiffs(True, recrop_height // merge_stride,
                                       recrop_width // merge_stride,
                                       r_start // merge_stride,
                                       c_start // merge_stride,
                                       mega_master_file,
                                       images_path, save_full_path,
                                       insist_prefix='quality_fmasks_',
                                       keep_border=True, compression=compression_type)

                    if output_feature_maps:
                        if prediction_shift == 0:
                            feat_filename = 'feature_maps_' + sfr + name_attachment_wo_shift + '.csv'
                            save_full_path = os.path.join(config['inference_save_folder_geotiff_frame_' + sfr],
                                                          feat_filename)
                            merge_feat_csv_files(sfr, images_path, offsets_scales_dict, save_full_path,
                                                 c_start, r_start,
                                                 recrop_width, recrop_height)
                        if write_as_im_feature_maps:
                            feat_filename = 'feature_maps_' + sfr + name_attachment + '.tif'
                            save_full_path = os.path.join(config['inference_save_folder_geotiff_frame_' + sfr],
                                                          feat_filename)
                            list_of_files_to_mode.append(save_full_path)
                            merge_geotiffs(True, recrop_height // merge_stride,
                                           recrop_width // merge_stride,
                                           r_start // merge_stride,
                                           c_start // merge_stride, mega_master_file,
                                           images_path, save_full_path, insist_prefix='feats_' + sfr,
                                           keep_border=True, compression=compression_type)

                    if fr == 0 and calculating_cosine_similarity:
                        cos_filename = 'cosine_map' + name_attachment + '.tif'
                        save_full_path = os.path.join(config['inference_save_folder_geotiff_frame_' + sfr], cos_filename)
                        list_of_files_to_mode.append(save_full_path)
                        merge_geotiffs(True, recrop_height // merge_stride,
                                       recrop_width // merge_stride,
                                       r_start // merge_stride,
                                       c_start // merge_stride, mega_master_file,
                                       images_path, save_full_path, insist_prefix='cosines_',
                                       keep_border=True, compression=compression_type)
                        if saving_dates:
                            before_filename = 'before_date' + name_attachment + '.tif'
                            save_full_path = os.path.join(config['inference_save_folder_geotiff_frame_' + sfr],
                                                          before_filename)
                            list_of_files_to_mode.append(save_full_path)
                            merge_geotiffs(True, recrop_height // merge_stride,
                                           recrop_width // merge_stride,
                                           r_start // merge_stride,
                                           c_start // merge_stride, mega_master_file,
                                           images_path, save_full_path, insist_prefix='before_epochs_',
                                           keep_border=True, compression=compression_type)

                if calculating_cosine_similarity and saving_dates:
                    sfr = '0'
                    inputs_for_ints = [os.path.splitext(os.path.split(
                        geotiff_database_infer_index['input_file_' + str(subfr)])[1])[0]
                                       for subfr in range(num_frames)]
                    if date_regex is not None:
                        input_dates_as_ints = extract_dates_as_int(inputs_for_ints, date_regex)
                    else:
                        input_dates_as_ints = [i for i in range(len(inputs_for_ints))]
                    before_filename = 'before_date' + name_attachment
                    before_full_path = os.path.join(config['inference_save_folder_geotiff_frame_' + sfr],
                                                    before_filename + '.tif')
                    which_before_epoch, ds_which_before_epoch, _, _ = read_geotiff(before_full_path)
                    ds_geot = copy.deepcopy(list(ds_which_before_epoch.GetGeoTransform()))
                    ds_prj = copy.deepcopy(ds_which_before_epoch.GetProjection())
                    ds_which_before_epoch = None

                    if saving_fmask:
                        result = merge_fmasks_pooled(config, inputs_for_ints, which_before_epoch)
                        fmask_filename = 'quality_fmask' + name_attachment
                        list_of_files_to_mode.append(os.path.join(config['inference_save_folder_geotiff_frame_' + sfr],
                                                                  fmask_filename + '.tif'))
                        write_geotiff('unknown', ds_geot, ds_prj, result,
                                      config['inference_save_folder_geotiff_frame_' + sfr], fmask_filename,
                                      alpha_im=None,
                                      data_type=gdal.GDT_Byte, save_tfw=True,
                                      no_data_values=[255], compression=compression_type)

                    for i, value in enumerate(input_dates_as_ints[:-1]):
                        which_before_epoch[which_before_epoch == i] = value
                    remove_all_file_versions(before_full_path)
                    write_geotiff('unknown', ds_geot,
                                  ds_prj, which_before_epoch,
                                  config['inference_save_folder_geotiff_frame_' + sfr], before_filename,
                                  alpha_im=None,
                                  data_type=gdal.GDT_Int32, save_tfw=True,
                                  no_data_values=[-100], compression=compression_type)
                    which_after_epoch = np.ones(which_before_epoch.shape) * input_dates_as_ints[-1]
                    which_after_epoch[which_before_epoch == -100] = -100
                    after_filename = 'after_date' + name_attachment
                    list_of_files_to_mode.append(os.path.join(config['inference_save_folder_geotiff_frame_' + sfr],
                                                              after_filename + '.tif'))
                    write_geotiff('unknown', ds_geot,
                                  ds_prj, which_after_epoch,
                                  config['inference_save_folder_geotiff_frame_' + sfr], after_filename,
                                  alpha_im=None,
                                  data_type=gdal.GDT_Int32, save_tfw=True,
                                  no_data_values=[-100], compression=compression_type)

            for dataset_item in dataset_infer.database:
                for key, ds in dataset_item.items():
                    if 'xr_' in key:
                        if ds is not None:
                            try:
                                ds.close()
                            except Exception as e:
                                print(f"Warning: could not close xarray dataset {key}: {e}")
                        dataset_item[key] = None
            del dataset_infer, dataloader_infer
            gc.collect()
            delete_temp_inferences(num_frames, out_folders, tiles_dir_input)

        if num_inferences > 1:
            all_prefix_lists = []
            save_paths_list = []
            if model.is_reconstruction:
                if calculating_cosine_similarity:
                    pre_fixes_list = ['cosine_map']
                    if saving_dates:
                        pre_fixes_list.append('before_date')
                        pre_fixes_list.append('after_date')
                    if saving_fmask:
                        pre_fixes_list.append('quality_fmask')
                    all_prefix_lists.append(pre_fixes_list)
                    save_paths_list.append(config['inference_save_folder_geotiff_frame_0'])
                if saving_reconstructed_images:
                    for fr in range(num_frames):
                        sfr = str(fr)
                        pre_fixes_list = ['infer_' + sfr]
                        all_prefix_lists.append(pre_fixes_list)
                        save_paths_list.append(config['inference_save_folder_geotiff_frame_' + sfr])
                if output_feature_maps and write_as_im_feature_maps:
                    for fr in range(num_frames):
                        sfr = str(fr)
                        pre_fixes_list = ['feature_maps_' + sfr]
                        all_prefix_lists.append(pre_fixes_list)
                        save_paths_list.append(config['inference_save_folder_geotiff_frame_' + sfr])
            for list_itr, pre_fixes_list in enumerate(all_prefix_lists):
                files_list = []
                save_path = save_paths_list[list_itr]
                for prfx in pre_fixes_list:
                    for file_item in list_of_files_to_mode:
                        filename = os.path.split(file_item)[1]
                        if filename.startswith(prfx) and filename.endswith('.tif'):
                            files_list.append(file_item)
                if files_list:
                    merge_shifted_results(config, files_list, save_path, name_attachment_wo_shift, pre_fixes_list)

        print(f"completed inference at index {index} in {time.time() - start_time_index} seconds")

    for fr in range(num_frames):
        sfr = str(fr)
        if os.path.exists(config['csv_inference_file']):
            os.unlink(config['csv_inference_file'])
        try:
            os.rmdir(config['inference_input_folder_frame_' + sfr])
            os.rmdir(config['inference_save_folder_frame_' + sfr])
        except:
            pass

    config["paths_included_in_csvs"] = original_paths_included_in_csvs
