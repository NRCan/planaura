from planaura.utils.raster_management.read_geotiff import read_geotiff
from planaura.utils.raster_management.write_geotiff import write_geotiff
import copy
import os
import cv2
import random
import rasterio
import numpy as np
from copy import deepcopy
from osgeo import gdal, osr
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject, Resampling as WarpResampling


# set the steps along rows and columns
def set_rows_columns_steps(middle_percentage, height_im, width_im, crop_size_height, crop_size_width,
                           crop_overlap_percentage, ensure_all_image=False, start_point_shift=0):
    crop_overlap_width = int(crop_overlap_percentage / 100.0 * crop_size_width)
    crop_overlap_height = int(crop_overlap_percentage / 100.0 * crop_size_height)

    if middle_percentage == 100:
        along_row = list(range(start_point_shift, height_im + 1 - crop_size_height, crop_size_height - crop_overlap_height))
        along_col = list(range(start_point_shift, width_im + 1 - crop_size_width, crop_size_width - crop_overlap_width))
        if ensure_all_image and along_row and along_col:
            if along_col[-1] + crop_size_width < width_im:
                along_col.append(width_im - crop_size_width)
            if along_row[-1] + crop_size_height < height_im:
                along_row.append(height_im - crop_size_height)
    else:
        st_h = max(0, int(height_im / 2 - height_im / 2.0 * middle_percentage / 100.0 - crop_size_height / 2.0 - 1))
        en_h = min(height_im + 1 - crop_size_height,
                   int(height_im / 2 + height_im / 2.0 * middle_percentage / 100.0 + crop_size_height / 2.0 + 1))
        st_w = max(0, int(width_im / 2 - width_im / 2.0 * middle_percentage / 100.0 - crop_size_width / 2.0 - 1))
        en_w = min(width_im + 1 - crop_size_width,
                   int(width_im / 2 + width_im / 2.0 * middle_percentage / 100.0 + crop_size_width / 2.0 + 1))
        along_row = list(range(st_h, en_h, crop_size_height - crop_overlap_height))
        along_col = list(range(st_w, en_w, crop_size_width - crop_overlap_width))
    return along_row, along_col


# Read the geographic information of the image
def read_gdal_ds(image_fullpath, middle_percentage, crop_size_height_base, crop_size_width_base, down_size_factor,
                 resolution_threshold, down_size_factor_tuple=None, base_resolution=None):
    image_valid = True
    try:
        ds_im = gdal.Open(image_fullpath)
        height_im = ds_im.RasterYSize
        width_im = ds_im.RasterXSize
        prj_im = ds_im.GetProjection()
        geotrans_im = deepcopy(list(ds_im.GetGeoTransform()))
        this_resolution = abs(geotrans_im[1])
        if base_resolution is None:
            base_resolution = this_resolution
        if geotrans_im[2] != 0 or geotrans_im[4] != 0:
            print(image_fullpath, 'has an atypical geo-transformation!')

        if ((height_im * this_resolution * middle_percentage / 100.0) < (
                crop_size_height_base * base_resolution * down_size_factor) or
                (width_im * this_resolution *middle_percentage / 100.0) < (
                crop_size_width_base * base_resolution * down_size_factor)):
            print('width or height of the image is too small; trying to decrease the patch size ...')

            if (height_im  * this_resolution * middle_percentage / 100.0 < (crop_size_height_base * base_resolution) or
                    width_im  * this_resolution * middle_percentage / 100.0 < crop_size_width_base * base_resolution):
                print('width or height of the image is too small')
                image_valid = False
            else:
                if down_size_factor_tuple is not None:
                    if 1 in down_size_factor_tuple:
                        down_size_factor = 1
                    else:
                        image_valid = False
                else:
                    image_valid = False

        if down_size_factor > 1:
            sx = abs(down_size_factor * geotrans_im[1])
            sy = abs(down_size_factor * geotrans_im[5])
            if sx > resolution_threshold or sy > resolution_threshold:
                if down_size_factor_tuple is not None:
                    if 1 in down_size_factor_tuple:
                        down_size_factor = 1
                    else:
                        image_valid = False
                else:
                    image_valid = False

        ds_im = None
        return image_valid, height_im, width_im, prj_im, geotrans_im, down_size_factor
    except:
        image_valid = False
        print(image_fullpath, " could not be opened!")
        return image_valid, None, None, None, None, None


def read_write_sub_rasterio(image_fullpath, target_crs, target_transform,
                            r_start_first, c_start_first, crop_size_height_first, crop_size_width_first,
                            crop_size_height_base, crop_size_width_base, down_size_factor,
                            lead_zeros, im_name, prj_im, r_count, c_count, tiles_dir, cut_data_portion,
                            discard_beyond_three_bands=False, replace_unknown_nodata_zero=False, save_tfw=True):
    window = Window(c_start_first, r_start_first, crop_size_width_first, crop_size_height_first)

    ds = gdal.Open(image_fullpath)
    gdal_type = ds.GetRasterBand(1).DataType
    num_bands = 0
    no_data_vals = []
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i + 1)
        if band.GetColorInterpretation() != gdal.GCI_AlphaBand:
            num_bands += 1
            no_data_vals.append(band.GetNoDataValue())
        band = None
    ds = None

    if num_bands == 1:
        bands_sub = 'gray'
    elif num_bands == 3:
        bands_sub = 'rgb'
    else:
        bands_sub = 'unknown'

    if replace_unknown_nodata_zero:
        for i in range(len(no_data_vals)):
            if no_data_vals[i] is None:
                no_data_vals[i] = 0

    if discard_beyond_three_bands:
        if num_bands > 3:
            num_bands = 3
            no_data_vals = no_data_vals[0:3]

    with rasterio.open(image_fullpath) as src:
        dtype = src.dtypes[0]
        np_dtype = np.dtype(dtype)
        im_sub = np.ones((num_bands, crop_size_height_first, crop_size_width_first), dtype=dtype)
        dst_transform = window_transform(window, target_transform)
        for b in range(1, num_bands + 1):  # bands are 1-indexed in rasterio
            im_sub[b-1] *= np_dtype.type(no_data_vals[b-1])
            reproject(
                source=rasterio.band(src, b),
                destination=im_sub[b - 1],  # im_sub is 0-indexed
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform= dst_transform,
                dst_crs=target_crs,
                resampling=WarpResampling.cubic
            )
        geoTransformation = dst_transform.to_gdal()

    im_sub = np.transpose(im_sub, (1, 2, 0)) #should become channel-last to work with gdal
    save_name_wo_extension = os.path.splitext(im_name)[0] + '_' + str(r_count).zfill(
        lead_zeros) + '_' + str(c_count).zfill(lead_zeros)

    if down_size_factor != 1:
        im_sub = cv2.resize(im_sub, (crop_size_width_base, crop_size_height_base),
                            interpolation=cv2.INTER_CUBIC)
        geoTransformation[1] *= down_size_factor
        geoTransformation[5] *= down_size_factor

    keep_tile = True
    if cut_data_portion > 0:
        no_mask = np.ones((im_sub.shape[0], im_sub.shape[1])) * True
        if num_bands == 1:
            no_mask = im_sub == no_data_vals[0]
            no_data_count = float(np.sum(no_mask))
        else:
            no_masks = []
            for i in range(im_sub.shape[2]):
                no_masks.append(float(np.sum(np.logical_and(no_mask, im_sub[:, :, i] == no_data_vals[i]))))
            no_data_count = np.amax(np.array(no_masks))
        no_portion = no_data_count / float(im_sub.shape[0] * im_sub.shape[1])
        if no_portion > (1 - cut_data_portion):
            keep_tile = False

    if keep_tile:
        write_geotiff(bands_sub, geoTransformation, prj_im, im_sub, tiles_dir,
                      save_name_wo_extension, data_type=gdal_type, no_data_values=no_data_vals, save_tfw=save_tfw)


def read_write_sub(image_fullpath, r_start, c_start, crop_size_height, crop_size_width,
                   crop_size_height_base, crop_size_width_base, down_size_factor, geotrans_im,
                   lead_zeros, im_name, prj_im, r_count, c_count, tiles_dir, cut_data_portion,
                   discard_beyond_three_bands=False, replace_unknown_nodata_zero=False, save_tfw=True):
    im_sub, ds_sub, bands_sub, no_data_vals = read_geotiff(
        image_fullpath, strow=int(r_start), stcol=int(c_start),
        rowlen=int(crop_size_height),
        collen=int(crop_size_width))

    if replace_unknown_nodata_zero:
        for i in range(len(no_data_vals)):
            if no_data_vals[i] is None:
                no_data_vals[i] = 0

    if discard_beyond_three_bands:
        if bands_sub != 'gray':
            if im_sub.shape[2] > 3:
                im_sub = im_sub[:, :, 0:3]
                bands_sub = 'rgb'
                no_data_vals = no_data_vals[0:3]

    gdal_type = ds_sub.GetRasterBand(1).DataType
    geoTransformation = deepcopy(list(geotrans_im))

    x_start = geotrans_im[0] + c_start * geotrans_im[1] + r_start * geotrans_im[2]
    y_start = geotrans_im[3] + c_start * geotrans_im[4] + r_start * geotrans_im[5]

    geoTransformation[0] = x_start
    geoTransformation[3] = y_start

    if down_size_factor != 1:
        im_sub = cv2.resize(im_sub, (crop_size_width_base, crop_size_height_base),
                            interpolation=cv2.INTER_CUBIC)
        geoTransformation[1] *= down_size_factor
        geoTransformation[5] *= down_size_factor

    save_name_wo_extension = os.path.splitext(im_name)[0] + '_' + str(r_count).zfill(
        lead_zeros) + '_' + str(c_count).zfill(lead_zeros)

    keep_tile = True
    if cut_data_portion > 0:
        no_mask = np.ones((im_sub.shape[0], im_sub.shape[1])) * True
        if bands_sub == 'gray':
            no_mask = im_sub == no_data_vals[0]
            no_data_count = float(np.sum(no_mask))
        else:
            no_masks = []
            for i in range(im_sub.shape[2]):
                no_masks.append(float(np.sum(np.logical_and(no_mask, im_sub[:, :, i] == no_data_vals[i]))))
            no_data_count = np.amax(np.array(no_masks))
        no_portion = no_data_count / float(im_sub.shape[0] * im_sub.shape[1])
        if no_portion > (1 - cut_data_portion):
            keep_tile = False

    if keep_tile:
        write_geotiff(bands_sub, geoTransformation, prj_im, im_sub, tiles_dir,
                      save_name_wo_extension, data_type=gdal_type, no_data_values=no_data_vals, save_tfw=save_tfw)

    return x_start, y_start


def tile_geotiffs(num_frames: int, random_choice_prob, cut_data_portion, resolution_threshold, discard_beyond_three_bands,
                  replace_unknown_nodata_zero, save_tfw, INPUT_TARGET, single_image, crop_size_width_base, crop_size_height_base,
                  crop_overlap_percentages=None, down_size_factors=None, middle_percentages=None,
                  tiles_dir_input=None, tiles_dir_target=None, images_path_input=None, images_path_target=None,
                  image_names_input=None, image_names_target=None,
                  ensure_all_image=False, start_point_shift=0):
    if middle_percentages is None:
        middle_percentages = [100]

    if down_size_factors is None:
        down_size_factors = [1]

    if crop_overlap_percentages is None:
        crop_overlap_percentages = [50]

    along_row_at_least_once = []
    along_col_at_least_once = []
    for dataset_index in range(len(image_names_input[0])):

        random_prob = random.uniform(0.0, 1.0)
        if random_prob < random_choice_prob:
            crop_overlap_percentage = crop_overlap_percentages[0]
            down_size_factor = down_size_factors[0]
            middle_percentage = middle_percentages[0]
        else:
            crop_overlap_percentage = random.choice(crop_overlap_percentages)
            down_size_factor = random.choice(down_size_factors)
            middle_percentage = random.choice(middle_percentages)

        print(
            'selected configurations before any adjustment: overlap = {0}%, resize factor = {1}, image coverage = {2}%'.format(
                crop_overlap_percentage, down_size_factor, middle_percentage))

        valid_im = [False] * (num_frames * 2)
        height_im = [0] * (num_frames * 2)
        width_im = [0] * (num_frames * 2)
        prj_im = [''] * (num_frames * 2)
        geotrans_im = [None] * (num_frames * 2)
        name_im = [''] * (num_frames * 2)
        fullpath_im = [''] * (num_frames * 2)
        tiledir_im = [''] * (num_frames * 2)
        base_resolution = None
        if image_names_input[0][dataset_index] is not None:
            image_fullpath_input = os.path.join(images_path_input[0], image_names_input[0][dataset_index])
            valid_im_input, height_im_input, width_im_input, prj_im_input, geotrans_im_input, down_size_factor = \
                read_gdal_ds(image_fullpath_input, middle_percentage, crop_size_height_base,
                             crop_size_width_base, down_size_factor, resolution_threshold, tuple(down_size_factors))
            base_resolution = abs(geotrans_im_input[1])
            if not valid_im_input:
                continue
        elif image_names_target[0][dataset_index] is not None:
            image_fullpath_target = os.path.join(images_path_target[0], image_names_target[0][dataset_index])
            valid_im_target, height_im_target, width_im_target, prj_im_target, geotrans_im_target, down_size_factor = \
                read_gdal_ds(image_fullpath_target, middle_percentage, crop_size_height_base,
                             crop_size_width_base, down_size_factor, resolution_threshold, tuple(down_size_factors))
            base_resolution = abs(geotrans_im_target[1])
            if not valid_im_target:
                continue
        else:
            continue

        crop_size_width = crop_size_width_base * down_size_factor
        crop_size_height = crop_size_height_base * down_size_factor

        print(
            'selected configurations after any adjustment: overlap = {0}%, resize factor = {1}, image coverage = {2}%'.format(
                crop_overlap_percentage, down_size_factor, middle_percentage))

        for fr in range(num_frames):
            if image_names_input[fr][dataset_index] is not None:
                image_fullpath = os.path.join(images_path_input[fr], image_names_input[fr][dataset_index])
                fullpath_im[fr] = image_fullpath
                tiledir_im[fr] = tiles_dir_input[fr]
                name_im[fr] = image_names_input[fr][dataset_index]
                valid_im[fr], height_im[fr], width_im[fr], prj_im[fr], geotrans_im[fr], _ = \
                    read_gdal_ds(image_fullpath, middle_percentage, crop_size_height_base,
                                 crop_size_width_base, down_size_factor, resolution_threshold,
                                 tuple(down_size_factors), base_resolution=base_resolution)

        for fr in range(num_frames):
            if image_names_target[fr][dataset_index] is not None:
                image_fullpath = os.path.join(images_path_target[fr], image_names_target[fr][dataset_index])
                fullpath_im[num_frames + fr] = image_fullpath
                tiledir_im[num_frames + fr] = tiles_dir_target[fr]
                name_im[num_frames + fr] = image_names_target[fr][dataset_index]
                valid_im[num_frames + fr], height_im[num_frames + fr], width_im[num_frames + fr], \
                    prj_im[num_frames + fr], geotrans_im[num_frames + fr], _ = \
                    read_gdal_ds(image_fullpath, middle_percentage, crop_size_height_base,
                                 crop_size_width_base, down_size_factor, resolution_threshold,
                                 tuple(down_size_factors), base_resolution=base_resolution)

        along_row = []
        along_col = []
        first_valid_index = 0
        for i in range(2 * num_frames):
            if valid_im[i]:
                along_row, along_col = set_rows_columns_steps(middle_percentage, height_im[i], width_im[i],
                                                              crop_size_height, crop_size_width,
                                                              crop_overlap_percentage, ensure_all_image,
                                                              start_point_shift)
                if along_row and along_col:
                    first_valid_index = i
                    break

        lead_zeros = max(len(str(len(along_row))), len(str(len(along_col))))

        if along_row and along_col:
            along_row_at_least_once.extend(copy.deepcopy(along_row))
            along_col_at_least_once.extend(copy.deepcopy(along_col))
            dest_prj = osr.SpatialReference(wkt=prj_im[first_valid_index])
            for indx in range(2 * num_frames):
                if indx != first_valid_index:
                    if valid_im[indx]:
                        source_prj = osr.SpatialReference(wkt=prj_im[indx])
                        if not dest_prj.IsSame(source_prj) or abs(geotrans_im[indx][1]) != base_resolution:
                            valid_im[indx] = 9999
                            print(
                                'Warning! The CRS or Resolution of image {} is different from the base;'
                                'a re-projection will happen which might affect the quality of the data'
                                '!'.format(name_im[indx]))

        with rasterio.open(fullpath_im[first_valid_index]) as src:
            target_first_crs = src.crs
            xres, _ = src.res
            target_first_transform = src.transform

        for r_count, r_start_first in enumerate(along_row):
            for c_count, c_start_first in enumerate(along_col):
                if not (c_start_first + crop_size_width > width_im[first_valid_index] or
                        r_start_first + crop_size_height > height_im[first_valid_index] or
                        r_start_first < 0 or c_start_first < 0):
                    x_start_first, y_start_first = read_write_sub(fullpath_im[first_valid_index], r_start_first,
                                                                  c_start_first,
                                                                  crop_size_height, crop_size_width,
                                                                  crop_size_height_base, crop_size_width_base,
                                                                  down_size_factor, geotrans_im[first_valid_index],
                                                                  lead_zeros, name_im[first_valid_index],
                                                                  prj_im[first_valid_index], r_count, c_count,
                                                                  tiledir_im[first_valid_index], cut_data_portion,
                                                                  discard_beyond_three_bands=discard_beyond_three_bands,
                                                                  replace_unknown_nodata_zero=replace_unknown_nodata_zero,
                                                                  save_tfw=save_tfw)

                    for indx in range(2 * num_frames):
                        if indx != first_valid_index:
                            if valid_im[indx] == True: # the case that crs and resolution are the same
                                cr = np.matmul(np.linalg.inv(
                                    np.array(
                                        [[geotrans_im[indx][1], geotrans_im[indx][2]],
                                         [geotrans_im[indx][4], geotrans_im[indx][5]]])),
                                    np.array(
                                        [[x_start_first - geotrans_im[indx][0]],
                                         [y_start_first - geotrans_im[indx][3]]]))
                                c_start = int(cr[0, 0])
                                r_start = int(cr[1, 0])
                                c_end = c_start + crop_size_width
                                r_end = r_start + crop_size_height

                                if not (c_end > width_im[indx] or r_end > height_im[
                                    indx] or r_start < 0 or c_start < 0):
                                    _, _ = read_write_sub(fullpath_im[indx], r_start,
                                                          c_start,
                                                          crop_size_height, crop_size_width,
                                                          crop_size_height_base, crop_size_width_base,
                                                          down_size_factor,
                                                          geotrans_im[indx],
                                                          lead_zeros, name_im[indx],
                                                          prj_im[indx], r_count, c_count,
                                                          tiledir_im[indx],
                                                          cut_data_portion,
                                                          discard_beyond_three_bands=discard_beyond_three_bands,
                                                          replace_unknown_nodata_zero=replace_unknown_nodata_zero,
                                                          save_tfw=save_tfw)
                            elif valid_im[indx] == 9999:
                                read_write_sub_rasterio(fullpath_im[indx], target_first_crs, target_first_transform,
                                                        r_start_first, c_start_first,
                                                        crop_size_height, crop_size_width,
                                                        crop_size_height_base, crop_size_width_base,
                                                        down_size_factor,
                                                        lead_zeros, name_im[indx], prj_im[first_valid_index],
                                                        r_count, c_count, tiledir_im[indx],
                                                        cut_data_portion,
                                                        discard_beyond_three_bands=discard_beyond_three_bands,
                                                        replace_unknown_nodata_zero=replace_unknown_nodata_zero,
                                                        save_tfw=save_tfw)


    if along_col_at_least_once and along_row_at_least_once:
        return True
    else:
        return False
