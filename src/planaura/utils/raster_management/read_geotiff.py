import numpy as np
from osgeo import gdal

def read_geotiff(filename, strow=0, stcol=0, rowlen=None, collen=None, mask_no_data=False):
    """" Reads geotiff raster from filename into a numpy array by
    cropping it to upper left corner at (strow and stcol) with width=collen and height=rowlen """
    ds = gdal.Open(filename)
    if ds is None:
        print('Could not open raster file')
        return None, None
    arr = []
    no_datas = []
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i + 1)
        rows = ds.RasterYSize
        cols = ds.RasterXSize
        # print('rows = {}, cols= {}'.format(rows, cols))
        if rowlen is None:
            rowlen = rows - strow
        if collen is None:
            collen = cols - stcol
        if band.GetColorInterpretation() != gdal.GCI_AlphaBand:
            arr.append(band.ReadAsArray(stcol, strow, collen, rowlen))
            no_datas.append(band.GetNoDataValue())
        band = None

    if len(arr) == 1:  # gray or gray-alpha
        if mask_no_data:
            arr[0] = np.ma.masked_equal(arr[0], no_datas[0])
        return arr[0], ds, 'gray', no_datas
    elif len(arr) == 3:  # rgb or rgb-alpha
        if mask_no_data:
            for i in range(3):
                arr[i] = np.ma.masked_equal(arr[i], no_datas[i])
        return np.stack(arr[0:3], axis=2), ds, 'rgb', no_datas
    else:
        n = len(arr)
        return np.stack(arr[0:n], axis=2), ds, 'unknown', no_datas
