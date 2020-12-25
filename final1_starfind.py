##########################################
'''
author: @mauritzwicker
date: 03.12.2020

Purpose: METHOD 1: using starfinder

'''
##########################################

########
#IMPORTS
import time
import psutil
import os

import pickle
import cv2
import numpy as np
from xml.etree import ElementTree
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.ttk import Progressbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sklearn
from sklearn import model_selection
import random
import seaborn as sns
import pandas as pd

from datetime import datetime
import math
import ephem
import warnings

warnings.filterwarnings("ignore")

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import DAOStarFinder
from astropy.stats import mad_std
from photutils import aperture_photometry, CircularAperture
from astropy.stats import biweight_location
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
from photutils import centroid_com, centroid_1dg, centroid_2dg, centroid_sources
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from astropy.visualization import simple_norm


#from other files
from parameters import *
# from other_functions import *

########
def background_noise_est(img_data, param_sigma = 3):
    ''' to remove the background using photoutils '''
    #we use sigma_clippins (read documentation) but first want to mask our sources, for an accurate value
    #we set nsigma, npixels, and dilate_size:
    #mask source docu: https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.make_source_mask.html#photutils.segmentation.make_source_mask
    # mask = make_source_mask(img_data, nsigma=param_nsgiam, npixels=param_npixl)
    #so npixels is how many pixels for something to be considered a source
    #sigma is amount of sigma a pixel needs to be above background to be considered for source
    # mean, median, std = sigma_clipped_stats(img_data, sigma=3.0, mask=mask)

    #to get the actual background:
    sigma_clip = SigmaClip(sigma=param_sigma)
    bkg_estimator = MedianBackground()
    bkg = Background2D(img_data, (5, 5), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    return(img_data - bkg.background)

def source_detection(img_data, param_fwhm = 3, param_thresh = 5):
    ''' to detect sources using photoutils '''
    #We will estimate the background and background noise using sigma-clipped statistics:
    mean, median, std = sigma_clipped_stats(img_data, sigma=3.0)

    #Now we will subtract the background and use an instance of DAOStarFinder to find the
    #stars in the image that have FWHMs of around 3 pixels and have peaks approximately
    #5-sigma above the background. Running this class on the data yields an astropy Table containing
    daofind = DAOStarFinder(fwhm=param_fwhm, threshold=param_thresh*std)
    sources = daofind(img_data)
    if sources == None:
        return(0)
    else:
        for col in sources.colnames:
            sources[col].info.format = '%.8g'  # for consistent table output
        return(len(sources))

########


def starcounter_m1(datavals, params_selected):
    '''
        input: the datavals array, and the parameters (as lists)
        Task: to run through the star finder algorith with the data and these params and determine starcount
        outpu: Dataframe of how params affected the number of stars detected (and other output depending on documentation)
    '''




    pass

def cut_data(data):
    '''
        input: full np.array
        Task: to cut the np.array
        output: the cut data as a np.array
    '''
    #cut the data
    a = 700
    b = 1400
    c = 1000
    d = 1700

    cut_data = data[a:b, c:d]
    return(cut_data)

def read_data(path):
    '''
        input: path to the data
        Task: to read the data
        output: the data as a np.array
    '''
    with open(path,'rb') as file:
        object_file = pickle.load(file)
    return(object_file['xml_counts'])


########





if __name__ == '__main__':
    t_start = time.perf_counter()
    pid = os.getpid()
    process = psutil.Process(pid)


    ########
    '''
    What we need to do:
    - load the data we want to count the number of stars for
    - read the documentation to know what parameters do what and what the outputs are
    - run the data through our star counter and get the number of counted stars usw
    - here we can do more eval because i think it gives us more data about the stars (so look)
    '''

    path_img = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/data_out/20201118230658_eval.pickle'

    data_img = read_data(path_img)
    data_img = cut_data(data_img)

    #background_noise_est to remove background with parameters [param_sigma]
    #source_detection to determine starcount with parameters [param_fwhm, param_thresh]

    '''
    # run a: determine different number of background reduced for different parameters
    #parameters: param_nsgiam, param_npixl
    #so run param_sigma from 1 - 10 and see how affects background
    sigmas_params = np.linspace(2, 10, 50)
    bkg_median = []
    for set_sigma in sigmas_params:
        print(set_sigma)
        img_new, bkg_sigma = background_noise_est(data_img, set_sigma)
        bkg_median.append(bkg_sigma)

    plt.scatter(sigmas_params, bkg_median)
    plt.show()
    '''

    min_val, max_val = 0, 15
    intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))


    # run b: determine the number of stars in the image
    thresh_params = np.linspace(1, 4, 7)
    fwhm_params = np.linspace(1, 8, 15)
    sources_found = []

    for fwhm_param in fwhm_params:
        sources_thres = []
        for thresh_param in thresh_params:
            print(fwhm_param, thresh_param)
            num_sources = source_detection(data_img, fwhm_param, thresh_param)
            # thisrun = [fwhm_param, thresh_param, num_sources]
            sources_thres.append(num_sources)
        sources_found.append(sources_thres)

    #so sourced found i list of lists, where first list is for one fwhm, second for another
    df = pd.DataFrame(sources_found, columns = thresh_params, index = fwhm_params)
    print()
    print()
    print(df)
    print()
    print()
    sns.heatmap(df, xticklabels=df.columns.values.round(2), yticklabels=df.index.values.round(2))
    plt.xlabel('Threshold')
    plt.yticks(rotation = 0)
    plt.ylabel('FWHM')
    plt.show()




    ########


    #Done
    t_fin = time.perf_counter()
    print('\n \n \n********* Process-{0} over ************'.format(pid))
    print('Runtime: {0} seconds'.format(t_fin - t_start))
    p_mem = process.memory_info().rss
    print('Memory used: {0} bytes ({1} GB) '.format(p_mem, p_mem/(1e9)))
