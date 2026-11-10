
import copy
import os
from copy import deepcopy
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import islice, product
from threading import local

import cv2
import rasterio
import numpy as np
from osgeo import gdal, osr, ogr
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.warp import transform_bounds
from rasterio.crs import CRS

from planaura.utils.raster_management.read_geotiff import read_geotiff
from planaura.utils.raster_management.write_geotiff import write_geotiff


def get_intersect(bbox: ogr.Geometry, vector_ds: ogr.DataSource) -> bool:
    """
    Check if a bounding box intersects with any feature in a vector dataset.

    :param bbox: An OGR Geometry representing the bounding box.
    :param vector_ds: An OGR DataSource containing vector features.
    :return: True if there is an intersection, False otherwise.
    """
    for layer_index in range(vector_ds.GetLayerCount()):
        layer = vector_ds.GetLayerByIndex(layer_index)
        layer.SetSpatialFilter(bbox)
        layer.ResetReading()
        feature = layer.GetNextFeature()
        if feature is not None:
            return True

    return False


def get_vector_patch(geotrans, r_start, c_start, crop_size_height, crop_size_width, transform=None) -> ogr.Geometry:
    """Build the tile footprint, including possible affine rotation/skew."""
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for col, row in (
        (c_start, r_start),
        (c_start + crop_size_width, r_start),
        (c_start + crop_size_width, r_start + crop_size_height),
        (c_start, r_start + crop_size_height),
        (c_start, r_start),
    ):
        x = geotrans[0] + col * geotrans[1] + row * geotrans[2]
        y = geotrans[3] + col * geotrans[4] + row * geotrans[5]
        ring.AddPoint(x, y)
    bbox = ogr.Geometry(ogr.wkbPolygon)
    bbox.AddGeometry(ring)
    if transform is not None:
        bbox.Transform(transform)
    return bbox


def _float32_tile_if_needed(image, gdal_type, no_data_vals):
    """Only downcast Float64 inputs; preserve smaller integer and Float32 rasters."""
    if gdal_type != gdal.GDT_Float64:
        return image, gdal_type, no_data_vals
    image = image.astype(np.float32, copy=False)
    no_data_vals = [None if v is None else float(np.float32(v)) for v in no_data_vals]
    return image, gdal.GDT_Float32, no_data_vals


# set the steps along rows and columns
def set_rows_columns_steps(raster_path, middle_percentage, height_im, width_im, crop_size_height, crop_size_width,
                           crop_overlap_percentage, ensure_all_image=False, start_point_shift=0, bbox=None, bbox_crs=None):
    crop_overlap_width = int(crop_overlap_percentage / 100.0 * crop_size_width)
    crop_overlap_height = int(crop_overlap_percentage / 100.0 * crop_size_height)

    row_min = 0
    col_min = 0
    if bbox is not None:
        middle_percentage = 100
        minx, miny, maxx, maxy = bbox
        with rasterio.open(raster_path) as ds:
            raster_crs = ds.crs
            if bbox_crs is not None:
                if str(raster_crs) != bbox_crs:
                    minx, miny, maxx, maxy = transform_bounds(bbox_crs, raster_crs, minx, miny, maxx, maxy)
            row_min, col_min = ds.index(minx, maxy)
            height_im, width_im = ds.index(maxx, miny)

    if middle_percentage == 100:
        along_row = list(range(start_point_shift + row_min, height_im + 1 - crop_size_height, crop_size_height - crop_overlap_height))
        along_col = list(range(start_point_shift + col_min, width_im + 1 - crop_size_width, crop_size_width - crop_overlap_width))
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
        dtype = "float32" if gdal_type == gdal.GDT_Float64 else src.dtypes[0]
        np_dtype = np.dtype(dtype)
        if gdal_type == gdal.GDT_Float64:
            gdal_type = gdal.GDT_Float32
            no_data_vals = [None if v is None else float(np.float32(v)) for v in no_data_vals]
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
        geoTransformation = list(dst_transform.to_gdal())

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
    ds_sub = None  # Release the GDAL dataset before parallel tile writing.
    im_sub, gdal_type, no_data_vals = _float32_tile_if_needed(im_sub, gdal_type, no_data_vals)
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
                  ensure_all_image=False, start_point_shift=0, bbox=None, bbox_crs=None,
                  conditioned_on=None, tiling_num_workers=4):
    # The cropping bounding box in form of (minx, miny, maxx, maxy), in the same CRS as the images if bbox_crs is None
    # otherwise in the bbox_crs as "EPSG:XXXX".
    if not isinstance(tiling_num_workers, int) or tiling_num_workers < 1:
        raise ValueError("tiling_num_workers must be a positive integer")

    if middle_percentages is None:
        middle_percentages = [100]

    if down_size_factors is None:
        down_size_factors = [1]

    if crop_overlap_percentages is None:
        crop_overlap_percentages = [50]

    along_row_at_least_once = []
    along_col_at_least_once = []
    epsg_codes = {}
    processed_tile_count = 0
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
                along_row, along_col = set_rows_columns_steps(raster_path=fullpath_im[i],
                                                              middle_percentage=middle_percentage,
                                                              height_im=height_im[i],
                                                              width_im=width_im[i],
                                                              crop_size_height=crop_size_height,
                                                              crop_size_width=crop_size_width,
                                                              crop_overlap_percentage=crop_overlap_percentage,
                                                              ensure_all_image=ensure_all_image,
                                                              start_point_shift=start_point_shift,
                                                              bbox=bbox, bbox_crs=bbox_crs)
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

        if not along_row or not along_col:
            continue

        with rasterio.open(fullpath_im[first_valid_index]) as src:
            target_first_crs = src.crs
            target_first_transform = src.transform
            epsg_codes[dataset_index] = target_first_crs.to_string()

        # OGR DataSource, Layer spatial filters, and CoordinateTransformation
        # cannot safely be shared between workers. Every thread owns its own.
        thread_data = local()
        if conditioned_on:
            check_ds = ogr.Open(conditioned_on, 0)
            if check_ds is None or check_ds.GetLayerCount() == 0:
                raise RuntimeError(f"Cannot open conditioning vector dataset: {conditioned_on}")
            vector_srs = check_ds.GetLayerByIndex(0).GetSpatialRef()
            if vector_srs is None:
                raise ValueError(f"Conditioning vector dataset has no CRS: {conditioned_on}")
            vector_wkt = vector_srs.ExportToWkt()
            vector_crs = CRS.from_wkt(vector_wkt)
            needs_transform = vector_crs != target_first_crs
            if needs_transform:
                print(f"Warning: conditioning vector {conditioned_on} will be reprojected for spatial queries")
            check_ds = None

        # Affine inverse is the same for all tiles within this image pair.
        inverse_transforms = {}
        for indx in range(2 * num_frames):
            if indx != first_valid_index and valid_im[indx] is True:
                matrix = np.array([[geotrans_im[indx][1], geotrans_im[indx][2]],
                                   [geotrans_im[indx][4], geotrans_im[indx][5]]])
                inverse_transforms[indx] = np.linalg.inv(matrix)

        def process_tile(tile):
            (r_count, r_start_first), (c_count, c_start_first) = tile
            if (c_start_first + crop_size_width > width_im[first_valid_index] or
                    r_start_first + crop_size_height > height_im[first_valid_index] or
                    r_start_first < 0 or c_start_first < 0):
                return False

            if conditioned_on:
                if not hasattr(thread_data, "vector_ds"):
                    thread_data.vector_ds = ogr.Open(conditioned_on, 0)
                    if thread_data.vector_ds is None:
                        raise RuntimeError(f"Cannot open conditioning vector dataset: {conditioned_on}")
                    thread_data.transform = None
                    if needs_transform:
                        src_srs = osr.SpatialReference(wkt=target_first_crs.to_wkt())
                        dst_srs = osr.SpatialReference(wkt=vector_wkt)
                        if hasattr(src_srs, "SetAxisMappingStrategy"):
                            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                            dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                        thread_data.transform = osr.CoordinateTransformation(src_srs, dst_srs)
                vector_patch_bbox = get_vector_patch(
                    geotrans_im[first_valid_index], r_start_first, c_start_first,
                    crop_size_height, crop_size_width, thread_data.transform)
                if not get_intersect(vector_patch_bbox, thread_data.vector_ds):
                    return False

            x_start_first, y_start_first = read_write_sub(
                fullpath_im[first_valid_index], r_start_first, c_start_first,
                crop_size_height, crop_size_width,
                crop_size_height_base, crop_size_width_base, down_size_factor,
                geotrans_im[first_valid_index], lead_zeros, name_im[first_valid_index],
                prj_im[first_valid_index], r_count, c_count, tiledir_im[first_valid_index],
                cut_data_portion, discard_beyond_three_bands=discard_beyond_three_bands,
                replace_unknown_nodata_zero=replace_unknown_nodata_zero, save_tfw=save_tfw)

            for indx in range(2 * num_frames):
                if indx == first_valid_index:
                    continue
                if valid_im[indx] is True:
                    cr = inverse_transforms[indx] @ np.array([
                        x_start_first - geotrans_im[indx][0],
                        y_start_first - geotrans_im[indx][3]])
                    c_start, r_start = int(cr[0]), int(cr[1])
                    if (c_start + crop_size_width > width_im[indx] or
                            r_start + crop_size_height > height_im[indx] or
                            r_start < 0 or c_start < 0):
                        continue
                    read_write_sub(
                        fullpath_im[indx], r_start, c_start, crop_size_height,
                        crop_size_width, crop_size_height_base, crop_size_width_base,
                        down_size_factor, geotrans_im[indx], lead_zeros, name_im[indx],
                        prj_im[indx], r_count, c_count, tiledir_im[indx],
                        cut_data_portion, discard_beyond_three_bands=discard_beyond_three_bands,
                        replace_unknown_nodata_zero=replace_unknown_nodata_zero, save_tfw=save_tfw)
                elif valid_im[indx] == 9999:
                    read_write_sub_rasterio(
                        fullpath_im[indx], target_first_crs, target_first_transform,
                        r_start_first, c_start_first, crop_size_height, crop_size_width,
                        crop_size_height_base, crop_size_width_base, down_size_factor,
                        lead_zeros, name_im[indx], prj_im[first_valid_index], r_count,
                        c_count, tiledir_im[indx], cut_data_portion,
                        discard_beyond_three_bands=discard_beyond_three_bands,
                        replace_unknown_nodata_zero=replace_unknown_nodata_zero, save_tfw=save_tfw)
            return True

        tile_positions = product(enumerate(along_row), enumerate(along_col))
        if tiling_num_workers == 1:
            for tile in tile_positions:
                processed_tile_count += bool(process_tile(tile))
        else:
            # Keep in-flight work bounded; executor.map() on an entire large grid
            # would enqueue a future for every tile in Python 3.9.
            with ThreadPoolExecutor(max_workers=tiling_num_workers) as executor:
                while True:
                    batch = list(islice(tile_positions, 2 * tiling_num_workers))
                    if not batch:
                        break
                    processed_tile_count += sum(executor.map(process_tile, batch))


    if processed_tile_count:
        return True, epsg_codes
    else:
        return False, None
