from planaura.utils.raster_management.gdal_merge import gdal_merge
import glob
import os
from osgeo import gdal
import sys
from planaura.utils.raster_management.read_geotiff import read_geotiff
from planaura.utils.raster_management.write_geotiff import write_geotiff
from copy import deepcopy


def get_partial_tiff(im_path, im_name, r_start, c_start,
                     crop_size_height, crop_size_width):
    im_sub, ds_sub, bands_sub, no_data_vals = read_geotiff(
        os.path.join(im_path, im_name), strow=int(r_start), stcol=int(c_start),
        rowlen=int(crop_size_height),
        collen=int(crop_size_width))

    geotrans_im = ds_sub.GetGeoTransform()
    prj_im = ds_sub.GetProjection()
    geoTransformation = deepcopy(list(geotrans_im))

    x_start = geotrans_im[0] + c_start * geotrans_im[1] + r_start * geotrans_im[2]
    y_start = geotrans_im[3] + c_start * geotrans_im[4] + r_start * geotrans_im[5]

    geoTransformation[0] = x_start
    geoTransformation[3] = y_start

    save_name_wo_extension = os.path.splitext(im_name)[0] + '__mmidtemp'
    write_geotiff(bands_sub, geoTransformation, prj_im, im_sub, im_path,
                  save_name_wo_extension, data_type=ds_sub.GetRasterBand(1).DataType, no_data_values=no_data_vals)
    return save_name_wo_extension


def enum_to_gtype(int_type):
    # gdal.GDT_
    gdal_types = ['Unknown', 'Byte', 'UInt16', 'Int16', 'UInt32', 'Int32', 'Float32', 'Float64', 'CInt16', 'CInt32',
                  'CFloat32', 'CFloat64']
    if 0 <= int_type < len(gdal_types):
        return gdal_types[int_type]
    else:
        raise ValueError('This integer does not correspond to any GDAL dat type')


def merge_geotiffs(fifty_overlap, recrop_height, recrop_width, r_start, c_start, master_image_fullpath, images_path,
                   save_full_path, ignore_prefix=None, insist_prefix=None, keep_border=False, compression="NONE"):
    valid_compressions = {'LZW', 'DEFLATE', 'ZSTD', 'NONE'}
    compression = compression.upper()
    if compression not in valid_compressions:
        print(f"Geotiff compression {compression} not found in: {', '.join(valid_compressions)}")
        compression = "NONE"
    if ignore_prefix is None:
        ignore_prefix = []
    if not isinstance(ignore_prefix, list):
        ignore_prefix = [ignore_prefix]
    if insist_prefix is None:
        insist_prefix = []
    if not isinstance(insist_prefix, list):
        insist_prefix = [insist_prefix]

    prj_im = None
    if images_path is None:
        raise ValueError("images_path should be provided")
    else:
        if master_image_fullpath is not None:
            ds_im = gdal.Open(master_image_fullpath)
            prj_im = ds_im.GetProjection()

    file_list = glob.glob(images_path + "/*.tif")
    if ignore_prefix:
        for prfx in ignore_prefix:
            file_list = [x for x in file_list if not os.path.basename(x).startswith(prfx)]
    if insist_prefix:
        file_list = [x for x in file_list if os.path.basename(x).startswith(tuple(insist_prefix))]

    new_file_list = []

    if fifty_overlap:
        for im_full_path in file_list:
            im_path = os.path.split(im_full_path)[0]
            im_name = os.path.split(im_full_path)[1]
            im_new_name = get_partial_tiff(im_path, im_name, r_start, c_start, recrop_height, recrop_width)
            new_file_list.append(os.path.join(im_path, im_new_name + '.tif'))
    else:
        new_file_list = file_list

    _, ds_sub, bands_sub, no_data_vals = read_geotiff(file_list[0])
    gdal_type = ds_sub.GetRasterBand(1).DataType
    gdal_type_string = enum_to_gtype(gdal_type)

    files_strings = " ".join(new_file_list)

    if no_data_vals[0] is not None:
        argv = "-ot " + gdal_type_string + " -n " + str(int(no_data_vals[0])) + " -a_nodata " + str(int(no_data_vals[0])) + " -of GTiff -co tfw=yes -co BIGTIFF=YES "
    else:
        argv = "-ot " + gdal_type_string + " -of GTiff -co tfw=yes -co BIGTIFF=YES "

    if fifty_overlap and keep_border:
        save_full_path_recovery = save_full_path
        save_path_temp = os.path.split(save_full_path)[0]
        save_name_temp = os.path.split(save_full_path)[1]
        save_full_path = os.path.join(save_path_temp, os.path.splitext(save_name_temp)[0] + '_midmergetemp.tif')

    argv += '-o {} '.format(save_full_path)
    argv += files_strings
    argv = argv.split(' ')

    if os.path.exists(save_full_path):
        os.remove(save_full_path)

    gdal_merge(argv, prj_im)

    if fifty_overlap:
        for im_full_path in new_file_list:
            try:
                im_path = os.path.split(im_full_path)[0]
                im_name = os.path.split(im_full_path)[1]
                os.remove(im_full_path)
                os.remove(os.path.join(im_path, os.path.splitext(im_name)[0] + '.tfw'))
            except OSError as e:
                print(e)

    if fifty_overlap and keep_border:
        files_strings = " ".join(file_list)
        if no_data_vals[0] is not None:
            argv = "-ot " + gdal_type_string + " -n " + str(int(no_data_vals[0])) + " -a_nodata " + str(
                int(no_data_vals[0])) + " -of GTiff -co tfw=yes -co BIGTIFF=YES "
        else:
            argv = "-ot " + gdal_type_string + " -of GTiff -co tfw=yes -co BIGTIFF=YES "
        if compression != 'NONE':  # 'NONE' is technically valid but no need to add it as an option
            argv = argv + f'-co COMPRESS={compression} '
        save_full_path_all = os.path.join(images_path, '_allmergedtemp.tif')
        argv += '-o {} '.format(save_full_path_all)
        argv += files_strings
        argv = argv.split(' ')

        if os.path.exists(save_full_path_all):
            os.remove(save_full_path_all)

        gdal_merge(argv, prj_im)

        final_file_list = [save_full_path_all, save_full_path]  # to make sure the good stuff is on top!
        files_strings = " ".join(final_file_list)
        if no_data_vals[0] is not None:
            argv = "-ot " + gdal_type_string + " -n " + str(int(no_data_vals[0])) + " -a_nodata " + str(
                int(no_data_vals[0])) + " -of GTiff -co tfw=yes -co BIGTIFF=YES "
        else:
            argv = "-ot " + gdal_type_string + " -of GTiff -co tfw=yes -co BIGTIFF=YES "
        if compression != 'NONE':
            argv = argv + f'-co COMPRESS={compression} '
        argv += '-o {} '.format(save_full_path_recovery)
        argv += files_strings
        argv = argv.split(' ')

        if os.path.exists(save_full_path_recovery):
            os.remove(save_full_path_recovery)

        gdal_merge(argv, prj_im)
        for im_full_path in final_file_list:
            try:
                im_path = os.path.split(im_full_path)[0]
                im_name = os.path.split(im_full_path)[1]
                os.remove(im_full_path)
                os.remove(os.path.join(im_path, os.path.splitext(im_name)[0] + '.tfw'))
            except OSError as e:
                print(e)

