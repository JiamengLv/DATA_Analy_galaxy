import os
import tarfile
from urllib.request import urlretrieve

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import requests
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS


# Legacy surveys 包含 DECaLS、BASS、DESI和MzLS等多个巡天项目，涵盖了北天和南天的广泛天空区域。
#          数据集：Legacy Surveys DR10 Cutouts at RA,Dec 是一组在指定坐标处提取的图像切割（cutouts），每个切割图像的文件名包含有对应的波段信息。
#          包含fits，以及jpg数据的下载


def save_fit(img, path):
    if os.path.exists(path):
        os.remove(path)
    grey = fits.PrimaryHDU(img)
    greyHDU = fits.HDUList([grey])
    greyHDU.writeto(path)


def read_fits(path):
    hdu = fits.open(path, ignore_missing_simple=True)
    img = hdu[0].data
    # img = np.array(img, dtype=np.float32)
    hdu.close()
    return img


def pltimage(data):
    plt.imshow(data)
    plt.show()


def get_jpg(download_jpg_url, file_name):
    r = requests.get(download_jpg_url)
    with open(file_name, "wb") as code:
        code.write(r.content)


class DataRadecLegacy():

    def __init__(self, ra, dec, savepath, band="grz"):

        """
        reference: https://www.legacysurvey.org/dr10/description/
        :param ra: RA of the target in degrees
        :param dec: Dec of the target in degrees
        :param band: desired filter (g, ,i, r, u, or z)
        :return:
        """
        self.ra, self.dec, self.band, self.savepath = ra, dec, band, savepath 

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.fitsfilename = self.savepath + f'cutout_{self.ra}_{self.dec}.fits'
        self.fitsurl = f'https://legacysurvey.org/viewer/fits-cutout?ra={self.ra}&dec={self.dec}&layer=ls-dr9&pixscale=0.262&bands={self.band}&size=2048'

        self.jpgfilename = self.savepath + f'cutout_{self.ra}_{self.dec}.jpg'
        self.jpgurl = f'https://legacysurvey.org/viewer/jpeg-cutout?ra={self.ra}&dec={self.dec}&ls-dr9&pixscale=0.262&size=2048'

    def downloader(self):

        if not os.path.exists(self.fitsfilename):

            print(self.fitsurl)

            urlretrieve(self.fitsurl, self.fitsfilename)
            get_jpg(self.jpgurl, self.jpgfilename)

    def imshow(self, plt=False):

        fitsdata = read_fits(self.fitsfilename)

        jpgdata = cv.imread(self.jpgfilename)
        jpgdata = cv.cvtColor(jpgdata, cv.COLOR_BGR2RGB)

        if plt:
            pltimage(jpgdata)
            pltimage(fitsdata)

        return fitsdata, jpgdata


if __name__ == "__main__":

    import pandas as pd
    import tqdm

    class_data = "/home/dell461/lys/Make_dection_json/csv/Du 2015 Low Surface Brightness Galaxies.csv"


    dfinfo =  pd.read_csv(class_data)

    # for i in tqdm.tqdm(range(len(dfinfo))):
    for i in tqdm.tqdm(range(10)): 
        savepath = "./LSBG-galaxy/"
        ra = dfinfo.ra[i]
        dec = dfinfo.dec[i] 
        try:

            DESI_object = DataRadecLegacy(ra, dec,savepath)
            DESI_object.downloader()
            DESI_fitsdata, DESI_jpgdata = DESI_object.imshow()

        except:
            pass

    





