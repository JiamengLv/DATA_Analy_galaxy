import numpy as np
import cv2 as cv

from glob import glob
import json
"""
Transformation from raw image data (nanomaggies) to the rgb values displayed
at the legacy viewer https://www.legacysurvey.org/viewer

Code copied from
https://github.com/legacysurvey/imagine/blob/master/map/views.py
"""

def sdss_rgb(imgs, bands, scales=None,
             m = 0.02):
    import numpy as np
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    rgb = np.clip(rgb, 0, 1)
    return rgb

def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

from astropy.io import fits

def DESI_find_Contour(fits_file):

    # labelme 的 标签修正

    # labelme_dataset_jsons = glob("/home/lvjm/Work_dir/Vision_downtasks/code/Single_objection/code/Label_me_json/"+fits_file.split("/")[-1].replace(".fits",".json"))

    # print(labelme_dataset_jsons)

    # for labelme_dataset_json in labelme_dataset_jsons:

    #     datainfo = json.load(open(labelme_dataset_json))["shapes"]
    #     for ii in range(len(datainfo)):

    #         x1 = datainfo[ii]["points"][0][0]
    #         y1 = datainfo[ii]["points"][0][1]
    #         x2 = datainfo[ii]["points"][1][0]
    #         y2 = datainfo[ii]["points"][1][1]

    #         if x2<100 and y2<100  and x1<100 and y1<100:

    #             print(datainfo[ii]["points"])

    #         else:
    #             print(max(x2-x1,y2-y1))
    #             return max(x2-x1,y2-y1)


    with fits.open(fits_file) as hdul:
        image = hdul[0].data

    print(image.shape)

    data = dr2_rgb(image, ['g', 'r', 'z'])
    gray = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    s = gray.shape[1]

    gray = np.uint8((gray - np.min(gray)) * 100 / (np.max(gray) - np.min(gray)))

    _, thresh = cv.threshold(gray, 20, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if x <= 0 or y <= 0:
            pass
        elif x + w > s or y + h > s:
            pass
        else:
            if x < s / 2 and x + w > s / 2 and y < s / 2 and y + h > s / 2:
                crop_size = max(int(x + w - s / 2), int(y + h - s / 2), int(s / 2 - x), int(s / 2 - y)) + 10
                if crop_size > int(s / 2):
                    crop_size = int(s / 2)
                return crop_size

    return 40