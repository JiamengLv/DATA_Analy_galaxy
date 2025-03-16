'''
单波段.
'''

import glob
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import astrophot as ap
# import sys
# print(sys.path)
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import label, find_objects

# from lib.residual import residuals_fits,model_fits
from astrophot.patch.residual import model_fits,residuals_fits
import multiprocessing
from astrophot import AP_config

def window_list(input_filename, data_shape, width=15, height=10):
    '''
    :param input_filename: 真值list路径，*************** 内容是：种类，x,y ***************
    :param data_shape:
    :param distance: 待拟合源的window，半长半宽，default:8
    :return: 左下右上点的坐标（x1，x2）（y1，y2）
    '''
    windows = []
    with open(input_filename, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split()[0:2])  # 读取每行的x和y坐标。 .strip() 删除行首和行尾的空白字符（空格或者换行符）  .split() 空格分隔符将文本行分割为多个部分   map（），映射成什么类型的数据
            x, y = int(x), int(y)  # 整数
            left_bottom = [max(0, x - width), max(0, y - height)]  # 计算左下点坐标
            if data_shape is not None:  # 计算右上点坐标
                right_top = [min(data_shape[1], x + width), min(data_shape[0], y + height)]
            else:
                right_top = [x + width, y + height]
            entry = {
                "r": [[left_bottom[0], right_top[0]], [left_bottom[1], right_top[1]]],# 左下点（x1，y1）和右上点（x2，y2），变为左下点（x1，x2），右上点变为（y1，y2）
            }
            windows.append(entry)
    # pprint(windows)
    return windows  # 不能超过边框

def center_window(img,data_shape,distance=8):
    '''
    :param img:ndarry
    :param  data_shape:
    :return:三个相同的左下右上点的坐标（x1，x2）（y1，y2）
    '''

    local_img = img
    # Set pixel values less than one sigma to 0, and retain values greater than
    yi, xi = np.indices(local_img.shape)
    sigma = np.std(local_img)
    gt_img_sigma = np.where(local_img > sigma, local_img, 0)

    labels, num_regions = label(gt_img_sigma != 0)  # Connect non zero elements into regions
    regions = find_objects(labels)  # Finding Objects in a Tagged Array

    # Search for local peaks_center and windowset
    peaks_center = np.array([])
    for i, region in enumerate(regions):
        ymin, ymax, xmin, xmax = region[0].start, region[0].stop, region[1].start, region[1].stop   # Obtain the coordinate range of the region in the x and y directions
        sub_image = gt_img_sigma[ymin:ymax, xmin:xmax]      # Slice and extract the region
        coordinates = peak_local_max(sub_image, min_distance=3, exclude_border=False)   # 在子图像中查找峰值坐标。min_distance:峰值之间的最小距离，exclude_border:是否排除边界上的峰值。不排除
        peaks_center = np.append(peaks_center, np.array(np.hstack([xmin + coordinates[:, 1].reshape(-1, 1), ymin + coordinates[:, 0].reshape(-1, 1)]))).reshape(-1,2)   #峰值的x/y从子图转到原始图像
        peaks_center = np.unique(peaks_center, axis=0)  # 以防重复

    windows = []
    for line in peaks_center:
        x, y = map(float, line.strip().split()[1:3])  # 读取每行的x和y坐标。 .strip() 删除行首和行尾的空白字符（空格或者换行符）  .split() 空格分隔符将文本行分割为多个部分   map（），映射成什么类型的数据
        x, y = int(x), int(y)  # 整数
        left_bottom = [max(0, x - distance), max(0, y - distance)]  # 计算左下点坐标
        if data_shape is not None:  # 计算右上点坐标
            right_top = [min(data_shape[1], x + distance), min(data_shape[0], y + distance)]
        else:
            right_top = [x + distance, y + distance]
        entry = {
            "r": [[left_bottom[0], right_top[0]], [left_bottom[1], right_top[1]]],  # 左下点（x1，y1）和右上点（x2，y2），变为左下点（x1，x2），右上点变为（y1，y2）
        }
        windows.append(entry)

    return windows  # 不能超过边框

def Astrofitting(Input_Fitspath,gt_path,fitsname):

    # Joint models with multiple models
    # If you want to analyze more than a single astronomical object, you will need to combine many models for each image in a reasonable structure.
    # There are a number of ways to do this that will work, though may not be as scalable. For small images, just about any arrangement is fine when using the LM optimizer.
    # But as images and number of models scales very large, it may be neccessary to sub divide the problem to save memory.
    # To do this you should arrange your models in a hierarchy so that AstroPhot has some information about the structure of your problem.
    # There are two ways to do this. First, you can create a group of models where each sub-model is a group which holds all the objects for one image.
    # Second, you can create a group of models where each sub-model is a group which holds all the representations of a single astronomical object across each image.
    # The second method is preferred. See the diagram below to help clarify what this means.
    # Here we will see an example of a multiband fit of an image which has multiple astronomical objects.

    rimg = fits.open(Input_Fitspath)
    target_r = ap.image.Target_Image(
        data = np.array(rimg[0].data, dtype = np.float64),
        zeropoint = 22.5,
        variance = np.ones(rimg[0].data.shape)*0.008**2, # note that the variance is important to ensure all images are compared with proper statistical weight. Here we just use the IQR^2 of the pixel values as the variance, for science data one would use a more accurate variance value
        # psf = ap.utils.initialize.gaussian_psf(1.12/2.355, 51, 0.262), # we construct a basic gaussian psf for each image by giving the simga (arcsec), image width (pixels), and pixelscale (arcsec/pixel)
        psf = ap.utils.initialize.gaussian_psf(1.12 / 2.355, 31, 1.1),
        wcs = WCS(rimg[0].header).sub([1,2]),
    )

    target_full = ap.image.Target_Image_List((target_r,))

    fig1, ax1 = plt.subplots(1, 2, figsize = (12,6))   # 单个画布不可遍历，库里面得修改
    ap.plots.target_image(fig1, ax1, target_full, flipx=True)
    ax1[0].set_title("ori image")
    plt.savefig('../Astro_tri_imgs/' + fitsname + '_ori.jpg')
    plt.show()
    plt.close()

    # There is barely any signal in the GALEX data and it would be entirely impossible to analyze on its own. With simultaneous multiband fitting it is a breeze to get relatively robust results!
    # Next we need to construct models for each galaxy. This is understandably more complex than in the single band case, since now we have three times the amout of data to keep track of.
    # Recall that we will create a number of joint models to represent each astronomical object, then put them all together in a larger group model.

    windows = window_list(gt_path,rimg[0].data.shape,width=15,height=10)
    # windows = center_window(rimg[0].data,rimg[0].data.shape )

    model_list = []
    for i, window in enumerate(windows):
        # create the submodels for this object
        sub_list = []
        sub_list.append(
            ap.models.AstroPhot_Model(
                name = f"same_band_1 model {i}",
                model_type = "sersic galaxy model",
                target = target_r,
                window = window["r"],
                psf_mode = "full",
                parameters = {"q": 0.3},
            )
        )

        # Make the multiband model for this object
        model_list.append(
            ap.models.AstroPhot_Model(
                name = f"model {i}",
                model_type = "group model",
                target = target_full,
                models = sub_list,
            )
        )
    # Make the full model for this system of objects
    MODEL = ap.models.AstroPhot_Model(
        name = f"full model",
        model_type = "group model",
        target = target_full,
        models = model_list,
    )

    # fig, ax = plt.subplots(1, 2, figsize = (12,5))
    # ap.plots.target_image(fig, ax, MODEL.target, flipx=True)
    # ap.plots.model_window(fig, ax, MODEL)     #有冲突
    # ax[0].set_title("window image")
    # plt.savefig('../Astro_tri_imgs/' + fitsname + '_window.jpg')
    # plt.show()
    # plt.close()

    # MODEL.initialize()    #bug修复拟合里面也初始化了,2s

    # result = ap.fit.LM(MODEL, verbose = 1, ).fit()

    result = ap.fit.LM(MODEL, verbose=1, max_iter=50).fit()

    fig1, ax1 = plt.subplots(1, 2, figsize = (12,4))
    ap.plots.model_image(fig1, ax1, MODEL, flipx=True)
    ax1[0].set_title(" model image")
    plt.savefig('../Astro_tri_imgs/' + fitsname + '_fitted.jpg')
    plt.show()
    plt.close()

    model_fits(model_path, fitsname, MODEL)  # Save the fitted fits

    fig, ax = plt.subplots(1, 2, figsize = (12,6))
    ap.plots.residual_image(fig, ax, MODEL, flipx=True)
    ax[0].set_title("residual image")
    plt.savefig('../Astro_tri_imgs/' + fitsname + '_residual.jpg')
    plt.show()
    plt.close()

    residuals_fits(MODEL,fitsname,residuals_path)      # Save the residual fits

    MODEL.save('../Astro_output_yaml/' + fitsname + '.yaml')    # 19s

def process_single_scene(fits_path):
    print(f'index: {fits_path}')
    path = r'/home/amax/zhangh/zh/S.Africa_data/MIGHTEE_COSMOS2021/small_40_wcs/'  # glob,最后要有/
    with open('AstroPhot.log', 'a') as file:
        file.write(f'index: {fits_path}\n')
    Input_Fitspath = fits_path
    filename = os.path.basename(Input_Fitspath).split('.fits')[0]  # 取文件名去掉扩展名
    gt_path = path + filename + '.list'
    Astrofitting(Input_Fitspath, gt_path, filename)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', True) #解决CUDA初始化问题。确保每个进程都在一个新的进程中启动，避免 CUDA 冲突。
    start_time = time.time()
    path = r'/home/amax/zhangh/zh/S.Africa_data/MIGHTEE_COSMOS2021/small_6050_wcs/'  # glob,最后要有/
    # path = r'/home/amax/zhangh/zh/S.Africa_data/MIGHTEE_COSMOS2021/small_40_wcs/'  # glob,最后要有/
    # fits_list = sorted(glob.glob(path + '*.fits'))
    fits_list = glob.glob(path + '*_4_0.fits')
    # fits_list = glob.glob(path + '*_38.fits')
    AP_config.ap_logger.info('\n')
    '''***************************** 
    :parameter
    原图（60，50）,window（30，20）
    1.路径
    2.window的width，height，(x,y)的读取 
    3.max_iter base：5，default：100 
    *****************************'''

    # fits_list = fits_list[:100]
    # fits_list = fits_list if fits_list is not None else []
    # with Pool(processes=5) as pool:     # 设置进程池，可以根据需要调整进程数
    #     pool.map(process_single_scene, fits_list)   #使用map（）映射。即process_single_scene（）将以并行方式在多个线程上运行，每个线程处理一个 fits_list 中的元素。

    for idx,fits_path in enumerate(fits_list[:100]):
        AP_config.ap_logger.info(f'index {idx}: {fits_path}')
        Input_Fitspath = fits_path
        filename = os.path.basename(Input_Fitspath).split('.fits')[0]  # 取文件名去掉扩展名
        gt_path = path + filename + '.list'

        model_path = os.path.join(os.path.dirname(gt_path), 'model_fits_save/')  # 构建 fits_model_path
        residuals_path = os.path.join(os.path.dirname(gt_path), 'residual_fits_save/') # 构建 fits_residual_path
        Astrofitting(Input_Fitspath,gt_path,filename)

    end_time = time.time()
    total_time = end_time - start_time      # unit:second
    minutes, seconds = divmod(total_time, 60)       #内置除法模块，（被除数，除数）返回商和余数
    AP_config.ap_logger.info(f"Program completed in {minutes:.0f} minutes and {seconds:.4f} seconds.")
    print('Program completed, you are awesome!')