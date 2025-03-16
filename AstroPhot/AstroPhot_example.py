import glob
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import astrophot as ap
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy.stats import iqr
import os




# First we need some data to work with, let's use LEDA 41136 as our example galaxy

# The images must be aligned to ======a common coordinate system=====. From the DESI Legacy survey we are extracting
# each image from a common center coordinate, so we set the center as (0,0) for all the images and they
# should be aligned.

# It is also important to have a good estimate of the =====variance and the PSF===== for each image since these
# affect the relative weight of each image. For the tutorial we use simple approximations, but in
# science level analysis one should endeavor to get the best measure available for these.

# Our first image is from the DESI Legacy-Survey r-band. This image has a pixelscale of 0.262 arcsec/pixel and is 500 pixels across
# lrimg = fits.open("https://www.legacysurvey.org/viewer/fits-cutout?ra=187.3119&dec=12.9783&size=500&layer=ls-dr9&pixscale=0.262&bands=r")
x = 0
if x:
    lrimg = fits.open("/home/dell460/zhangh/zh/S.Africa_data/MIGHTEE_COSMOS2021/astro_img/cutout_187.3119_12.9783_1.fits")
    target_r = ap.image.Target_Image(       # 看一下各参数
        data = np.array(lrimg[0].data, dtype = np.float64),
        zeropoint = 22.5,
        variance = np.ones((500,500))*0.008**2, # Here we just use the IQR^2 of the pixel values as the variance, for science data one would use a more accurate variance value
        psf = ap.utils.initialize.gaussian_psf(1.12/2.355, 51, 0.262), # we construct a basic gaussian psf for each image by giving the simga (arcsec), image width (pixels), and pixelscale (arcsec/pixel)
        wcs = WCS(lrimg[0].header), # note pixelscale and origin not needed when we have a WCS object!
    )
    # The second image is a unWISE W1 band image. This image has a pixelscale of 2.75 arcsec/pixel and is 52 pixels across
    # lw1img = fits.open("https://www.legacysurvey.org/viewer/fits-cutout?ra=187.3119&dec=12.9783&size=52&layer=unwise-neo7&pixscale=2.75&bands=1")
    lw1img = fits.open("/home/dell460/zhangh/zh/S.Africa_data/MIGHTEE_COSMOS2021/astro_img/cutout_187.3119_12.9783_2.fits")
    target_W1 = ap.image.Target_Image(
        data = np.array(lw1img[0].data, dtype = np.float64),
        zeropoint = 25.199,
        variance = np.ones((52,52))*4.9**2,
        psf = ap.utils.initialize.gaussian_psf(6.1/2.355, 21, 2.75),
        wcs = WCS(lw1img[0].header),
    )
    # The third image is a GALEX NUV band image. This image has a pixelscale of 1.5 arcsec/pixel and is 90 pixels across
    # lnuvimg = fits.open("https://www.legacysurvey.org/viewer/fits-cutout?ra=187.3119&dec=12.9783&size=90&layer=galex&pixscale=1.5&bands=n")
    lnuvimg = fits.open('/home/dell460/zhangh/zh/S.Africa_data/MIGHTEE_COSMOS2021/astro_img/cutout_187.3119_12.9783_3.fits')
    target_NUV = ap.image.Target_Image(
        data = np.array(lnuvimg[0].data, dtype = np.float64),
        zeropoint = 20.08,
        variance = np.ones((90,90))*0.0007**2,
        psf = ap.utils.initialize.gaussian_psf(5.4/2.355, 21, 1.5),
        wcs = WCS(lnuvimg[0].header),
    )

    fig1, ax1 = plt.subplots(1, 3, figsize = (18,6))
    ap.plots.target_image(fig1, ax1[0], target_r, flipx=True)
    ax1[0].set_title("r-band image")
    ap.plots.target_image(fig1, ax1[1], target_W1, flipx=True)
    ax1[1].set_title("W1-band image")
    ap.plots.target_image(fig1, ax1[2], target_NUV, flipx=True)
    ax1[2].set_title("NUV-band image")
    plt.savefig('../Astro_tri_imgs/example_single_ori.jpg')
    plt.show()



    # The joint model will need a target to try and fit, but now that we have multiple images the "target" is
    # a Target_Image_List object which points to all three.
    target_full = ap.image.Target_Image_List((target_r, target_W1, target_NUV))
    # It doesn't really need any other information since everything is already available in the individual targets


    # To make things easy to start, lets just fit a sersic model to all three.
    # In principle one can use arbitrary group models designed for each band individually, but that would be unecessarily complex for a tutorial
    model_r = ap.models.AstroPhot_Model(
        name = "rband model",
        model_type = "sersic galaxy model",     # sersic模型去拟合
        target = target_r,
        psf_mode = "full",
    )
    model_W1 = ap.models.AstroPhot_Model(
        name = "W1band model",
        model_type = "sersic galaxy model",
        target = target_W1,
        psf_mode = "full",
    )
    model_NUV = ap.models.AstroPhot_Model(
        name = "NUVband model",
        model_type = "sersic galaxy model",
        target = target_NUV,
        psf_mode = "full",
    )


    # At this point we would just be fitting three separate models at the same time, not very interesting. Next
    # we add constraints so that some parameters are shared between all the models. It makes sense to fix
    # structure parameters while letting brightness parameters vary between bands so that's what we do here.
    model_W1.add_equality_constraint(model_r, ["center", "q", "PA", "n", "Re"])     # 限制，让参数共享
    model_NUV.add_equality_constraint(model_r, ["center", "q", "PA", "n", "Re"])
    # Now every model will have a unique Ie, but every other parameter is shared for all three


    # We can now make the joint model object
    model_full = ap.models.AstroPhot_Model(
        name = "LEDA 41136",
        model_type = "group model",
        models = [model_r, model_W1, model_NUV],
        target = target_full,
    )
    model_full.initialize()

    result = ap.fit.LM(model_full, verbose = 1).fit()
    print(result.message)


    # here we plot the results of the fitting,
    # notice that each band has a ========different PSF and pixelscale========(每个波段有不同的PSF和pixelscale).
    # Also, notice that the colour bars represent significantly different ranges since each model was allowed to fit its own Ie.
    # meanwhile the center, PA, q, and Re is the same for every model.
    fig1, ax1 = plt.subplots(1, 3, figsize = (18,6))
    ap.plots.model_image(fig1, ax1, model_full, flipx=True)
    ax1[0].set_title("r-band model image")
    ax1[1].set_title("W1-band model image")
    ax1[2].set_title("NUV-band model image")
    plt.savefig('../Astro_tri_imgs/example_single_fitted.jpg')
    plt.show()



    # We can also plot the residual images.
    # As can be seen, the galaxy is fit in all three bands simultaneously with the majority of the light removed in all bands.
    # A residual can be seen in the r band. This is likely due to there being more structure in the r-band than just a sersic. The W1 and NUV bands look excellent though
    fig1, ax1 = plt.subplots(1, 3, figsize = (18,6))
    ap.plots.residual_image(fig1, ax1, model_full, flipx=True)      # 画残差图
    ax1[0].set_title("r-band residual image")
    ax1[1].set_title("W1-band residual image")
    ax1[2].set_title("NUV-band residual image")
    plt.savefig('../Astro_tri_imgs/example_single_residual.jpg')
    plt.show()


    # Save a joint model just like any other model
    model_full.save("./AstroPhot_output/example_single_jointsave.yaml")

    # This code cant run because it would create a bunch of models with names that already exist.
    # You can test it in a new file if you like
    ############################################
    # Load the joint model just like any other
    model_reload = ap.models.AstroPhot_Model(
        name = "reload LEDA 41136",
        filename = "jointsave.yaml",
    )

    # However, targets are not saved when saving a model, so those must be re-assigned manually
    # Assign the group target
    model_reload.target = target_full
    # Assign the sub-model targets
    model_reload.models["rband model"].target = target_r
    model_reload.models["W1band model"].target = target_W1
    model_reload.models["NUVband model"].target = target_NUV

    # You must also update the full model window before proceeding
    model_reload.update_window()

    # Plot everything again to check its working
    fig1, ax1 = plt.subplots(2, 3, figsize = (18,12))
    ap.plots.model_image(fig1, ax1[0], model_reload, flipx=True)
    ax1[0][0].set_title("r-band model image")
    ax1[0][1].set_title("W1-band model image")
    ax1[0][2].set_title("NUV-band model image")
    ap.plots.residual_image(fig1, ax1[1], model_reload, flipx=True)
    ax1[1][0].set_title("r-band residual image")
    ax1[1][1].set_title("W1-band residual image")
    ax1[1][2].set_title("NUV-band residual image")
    plt.show()



######################################################

# Joint models with multiple models
# If you want to analyze more than a single astronomical object, you will need to combine many models for each image in a reasonable structure.
# There are a number of ways to do this that will work, though may not be as scalable. For small images, just about any arrangement is fine when using the LM optimizer.
# But as images and number of models scales very large, it may be neccessary to sub divide the problem to save memory.
# To do this you should arrange your models in a hierarchy so that AstroPhot has some information about the structure of your problem.
# There are two ways to do this. First, you can create a group of models where each sub-model is a group which holds all the objects for one image.
# Second, you can create a group of models where each sub-model is a group which holds all the representations of a single astronomical object across each image.
# The second method is preferred. See the diagram below to help clarify what this means.

# Here we will see an example of a multiband fit of an image which has multiple astronomical objects.

# First we need some data to work with, let's use another LEDA object, this time a group of galaxies: LEDA 389779, 389797, 389681
# RA = 320.5003
# DEC = -57.4585

# Our first image is from the DESI Legacy-Survey r-band. This image has a pixelscale of 0.262 arcsec/pixel
x = 3
if x:
    rsize = 250
    rimg = fits.open(r"/home/dell460/zhangh/zh/S.Africa_data/MIGHTEE_COSMOS2021/astro_img/cutout_320.5003_-57.4585_4.fits")
    target_r = ap.image.Target_Image(
        data = np.array(rimg[0].data, dtype = np.float64),
        zeropoint = 22.5,
        variance = np.ones((rsize,rsize))*0.008**2, # note that the variance is important to ensure all images are compared with proper statistical weight. Here we just use the IQR^2 of the pixel values as the variance, for science data one would use a more accurate variance value
        psf = ap.utils.initialize.gaussian_psf(1.12/2.355, 51, 0.262), # we construct a basic gaussian psf for each image by giving the simga (arcsec), image width (pixels), and pixelscale (arcsec/pixel)
        wcs = WCS(rimg[0].header),
    )

    # The second image is a unWISE W1 band image. This image has a pixelscale of 2.75 arcsec/pixel
    wsize = 25
    w1img = fits.open(r"/home/dell460/zhangh/zh/S.Africa_data/MIGHTEE_COSMOS2021/astro_img/cutout_320.5003_-57.4585_5.fits")
    target_W1 = ap.image.Target_Image(
        data = np.array(w1img[0].data, dtype = np.float64),
        zeropoint = 25.199,
        variance = np.ones((wsize,wsize))*4.9**2,
        psf = ap.utils.initialize.gaussian_psf(6.1/2.355, 21, 2.75),
        wcs = WCS(w1img[0].header),
    )

    # The third image is a GALEX NUV band image. This image has a pixelscale of 1.5 arcsec/pixel
    gsize = 40
    nuvimg = fits.open(r"/home/dell460/zhangh/zh/S.Africa_data/MIGHTEE_COSMOS2021/astro_img/cutout_320.5003_-57.4585_6.fits")
    target_NUV = ap.image.Target_Image(
        data = np.array(nuvimg[0].data, dtype = np.float64),
        zeropoint = 20.08,
        variance = np.ones((gsize,gsize))*0.0007**2,
        psf = ap.utils.initialize.gaussian_psf(5.4/2.355, 21, 1.5),
        wcs = WCS(nuvimg[0].header)
    )
    target_full = ap.image.Target_Image_List((target_r, target_W1, target_NUV))

    fig1, ax1 = plt.subplots(1, 3, figsize = (18,6))
    ap.plots.target_image(fig1, ax1, target_full, flipx=True)
    ax1[0].set_title("r-band image")
    ax1[1].set_title("W1-band image")
    ax1[2].set_title("NUV-band image")
    plt.savefig('../Astro_tri_imgs/example_mutiple_ori.jpg')
    plt.show()



    # There is barely any signal in the GALEX data and it would be entirely impossible to analyze on its own. With simultaneous multiband fitting it is a breeze to get relatively robust results!
    # Next we need to construct models for each galaxy. This is understandably more complex than in the single band case, since now we have three times the amout of data to keep track of.
    # Recall that we will create a number of joint models to represent each astronomical object, then put them all together in a larger group model.

    # Here we enter the window parameters by hand, in general one would use a segmentation map or
    # some other automated proceedure to pick out the area for many objects
    windows = [
        {"r":[[72,152],[140,234]], "W1": [[3,18],[12,25]], "NUV": [[8,27],[20,39]]},
        {"r":[[43,155],[138,237]], "W1": [[1,17],[12,25]], "NUV": [[4,22],[19,39]]},
        {"r":[[115,210],[100,228]], "W1": [[8,23],[8,25]], "NUV": [[17,35],[13,38]]},
        {"r":[[69,170],[10,115]], "W1": [[5,19],[0,14]], "NUV": [[8,30],[1,18]]},
    ]

    model_list = []

    for i, window in enumerate(windows):
        # create the submodels for this object
        sub_list = []
        sub_list.append(
            ap.models.AstroPhot_Model(
                name = f"rband model {i}",
                model_type = "spline galaxy model", # we use spline models for the r-band since it is well resolved
                target = target_r,
                window = window["r"],
                psf_mode = "full",
                parameters = {"q": 0.3},
            )
        )
        sub_list.append(
            ap.models.AstroPhot_Model(
                name = f"W1band model {i}",
                model_type = "sersic galaxy model", # we use sersic models for W1 and NUV since there isn't much visible detail, a simple model is sufficient
                target = target_W1,
                window = window["W1"],
                psf_mode = "full",
                parameters = {"q": 0.3},
            )
        )
        sub_list.append(
            ap.models.AstroPhot_Model(
                name = f"NUVband model {i}",
                model_type = "sersic galaxy model",
                target = target_NUV,
                window = window["NUV"],
                psf_mode = "full",
                parameters = {"q": 0.3},
            )
        )
        # ensure equality constraints
        # across all bands, same center, q, PA
        sub_list[1].add_equality_constraint(sub_list[0], ["center", "q", "PA"])
        sub_list[2].add_equality_constraint(sub_list[0], ["center", "q", "PA"])
        # across W1 and NUV, same n, Re
        sub_list[2].add_equality_constraint(sub_list[1], ["n", "Re"])

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
    fig, ax = plt.subplots(1,3, figsize = (16,5))
    ap.plots.target_image(fig, ax, MODEL.target, flipx=True)
    ap.plots.model_window(fig, ax, MODEL)
    ax1[0].set_title("r-band image")
    ax1[1].set_title("W1-band image")
    ax1[2].set_title("NUV-band image")
    plt.savefig('../Astro_tri_imgs/example_mutiple_window.jpg')
    plt.show()
    MODEL.initialize()
    result = ap.fit.LM(MODEL, verbose = 1).fit()

    fig1, ax1 = plt.subplots(1, 3, figsize = (18,4))
    ap.plots.model_image(fig1, ax1, MODEL, flipx=True)
    ax1[0].set_title("r-band model image")
    ax1[1].set_title("W1-band model image")
    ax1[2].set_title("NUV-band model image")
    plt.savefig('../Astro_tri_imgs/example_mutiple_fitted.jpg')
    plt.show()


    # The models look excellent! The power of multiband fitting lets us know that we have extracted all the available information here, no forced photometry required!

    fig, ax = plt.subplots(1, 3, figsize = (18,6))
    ap.plots.residual_image(fig, ax, MODEL, flipx=True)
    ax[0].set_title("r-band residual image")
    ax[1].set_title("W1-band residual image")
    ax[2].set_title("NUV-band residual image")
    plt.savefig('../Astro_tri_imgs/example_mutiple_residual.jpg')
    plt.show()

    MODEL.save("./AstroPhot_output/example_mutiple_jointsave.yaml")

    print('completed')


    # Unfortunately the residuals do not look very good, in this case it is because of poor alignment in the three images.
    # Ensuring they are all on the same coordinate system is very important.
    # We will update this section in the future with a better example that actually gets a good result.
    # For now, this tutorial still shows how to set up a multi-band fit and serves as a warning to make sure you have good coordinates!











