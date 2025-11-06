from osgeo import gdal, osr
import os
import string
import random


def id_generator(size=8, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))


# Transform gdal Error from sys.stdout to runtimeerror
gdal.UseExceptions()


def write_geotiff(bands, geoTransformation, prj_im, im, save_path, save_name_wo_extension, alpha_im=None,
                  data_type=gdal.GDT_Byte, save_tfw=True, no_data_values=None, compression="NONE",
                  band_descriptions=None):

    if os.path.exists(os.path.join(save_path, save_name_wo_extension + '.tif')):
        os.remove(os.path.join(save_path, save_name_wo_extension + '.tif'))

    if len(im.shape) == 2:
        n_bands = 1
    else:
        n_bands = im.shape[2]
        if n_bands == 1:
            im = im[:,:,0]
    if band_descriptions:
        if len(band_descriptions) != n_bands:
            band_descriptions = None

    if no_data_values is None:
        no_data_values = [None] * n_bands

    valid_compressions = {'LZW', 'DEFLATE', 'ZSTD', 'NONE'}
    options = ['BIGTIFF=YES']
    compression = compression.upper()
    if compression not in valid_compressions:
        raise ValueError(f"Geotiff compression {compression} not found in: {', '.join(valid_compressions)}")
    if compression != 'NONE':  # 'NONE' is technically valid but no need to add it as an option
        options.append(f'COMPRESS={compression}')

    driver = gdal.GetDriverByName('GTiff')

    grid_data_name = f'grid_data_{save_name_wo_extension}' + id_generator()

    if alpha_im is not None:
        grid_data = driver.Create(grid_data_name, im.shape[1], im.shape[0], n_bands + 1, data_type, options=options)
    else:
        grid_data = driver.Create(grid_data_name, im.shape[1], im.shape[0], n_bands, data_type, options=options)

    # safer to set nodata values before writing arrays
    for i in range(n_bands):
        if no_data_values[i] is not None:
            grid_data.GetRasterBand(i + 1).SetNoDataValue(no_data_values[i])
        # grid_data.GetRasterBand(i + 1).ComputeStatistics(True)

    if bands == 'gray' or n_bands == 1:
        grid_data.GetRasterBand(1).WriteArray(im)
        grid_data.GetRasterBand(1).SetColorInterpretation(gdal.GCI_GrayIndex)
        if alpha_im is not None:
            grid_data.GetRasterBand(2).WriteArray(alpha_im)
            grid_data.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)

    elif bands == 'rgb' and n_bands == 3:
        grid_data.GetRasterBand(1).WriteArray(im[:, :, 0])
        grid_data.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        grid_data.GetRasterBand(2).WriteArray(im[:, :, 1])
        grid_data.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        grid_data.GetRasterBand(3).WriteArray(im[:, :, 2])
        grid_data.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        if alpha_im is not None:
            grid_data.GetRasterBand(4).WriteArray(alpha_im)
            grid_data.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    else:
        for i in range(n_bands):
            grid_data.GetRasterBand(i + 1).WriteArray(im[:, :, i])
            grid_data.GetRasterBand(i + 1).SetColorInterpretation(gdal.GCI_GrayIndex)
        if alpha_im is not None:
            grid_data.GetRasterBand(n_bands + 1).WriteArray(alpha_im)
            grid_data.GetRasterBand(n_bands + 1).SetColorInterpretation(gdal.GCI_AlphaBand)

    if band_descriptions is not None:
        for i in range(n_bands):
            grid_data.GetRasterBand(i + 1).SetDescription(band_descriptions[i])

    if geoTransformation is not None:
        grid_data.SetGeoTransform(geoTransformation)
    grid_data.SetProjection(prj_im)  # pass '' for prj_im if you don't know it!

    if save_tfw and geoTransformation is not None:
        with open(os.path.join(save_path, save_name_wo_extension + '.tfw'), 'w') as the_file:
            the_file.write('{}\n'.format(geoTransformation[1]))
            the_file.write('{}\n'.format(geoTransformation[2]))
            the_file.write('{}\n'.format(geoTransformation[4]))
            the_file.write('{}\n'.format(geoTransformation[5]))
            the_file.write('{}\n'.format(geoTransformation[0] + geoTransformation[1]/2.0))
            the_file.write('{}\n'.format(geoTransformation[3] + geoTransformation[5]/2.0))

    grid_data.BuildOverviews('average', [2, 4, 8, 16])
    driver.CreateCopy(os.path.join(save_path, save_name_wo_extension + '.tif'), grid_data, 0)

    # Close the file
    grid_data.FlushCache()
    grid_data = None
    driver = None
    try:
        os.remove(grid_data_name)
    except OSError:
        'no temporary file to be removed'
