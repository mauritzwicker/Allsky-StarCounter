##########################################
'''
author: @mauritzwicker
date: 03.12.2020

Purpose: METHOD 2: using threshold

'''
##########################################

########
#IMPORTS
import time
import psutil
import os

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
from PIL import Image
import pickle

#from other files

########




########
def delete_multiples(arr_input):
    '''
        go through true/false data and if star connected -> set False
    '''
    star_arr_corr = []
    for i in range(0, len(arr_input)):
        #loop through all the rows
        # print('in row, ', i)
        row_arr_corr = []
        for j in range(0, len(arr_input[i])):
            #loop through the values of the row
            pix_val = arr_input[i][j]
            if pix_val == False:
                row_arr_corr.append(False)
                continue
            else:
                #now go around the ring and if other True -> false
                # the vals we want:
                # arr_input[i-1][j-1], arr_input[i-1][j], arr_input[i-1][j+1]
                # arr_input[i][j-1],                    , arr_input[i][j+1]
                # arr_input[i+1][j-1], arr_input[i+1][j], arr_input[i+1][j+1]
                if i-1 >=0:
                    if arr_input[i-1][j] == True:
                        arr_input[i][j] = False
                        row_arr_corr.append(False)
                        continue
                    if j-1>=0:
                        if arr_input[i-1][j-1] == True:
                            arr_input[i][j] = False
                            row_arr_corr.append(False)
                            continue
                    if j+1<len(arr_input[i]):
                        if arr_input[i-1][j+1] == True:
                            arr_input[i][j] = False
                            row_arr_corr.append(False)
                            continue
                if j-1>=0:
                    if arr_input[i][j-1] == True:
                        arr_input[i][j] = False
                        row_arr_corr.append(False)
                        continue
                if j+1<len(arr_input[i]):
                    if arr_input[i][j+1] == True:
                        arr_input[i][j] = False
                        row_arr_corr.append(False)
                        continue
                if i+1<len(arr_input):
                    if arr_input[i+1][j] == True:
                        arr_input[i][j] = False
                        row_arr_corr.append(False)
                        continue
                    if j-1>=0:
                        if arr_input[i+1][j-1] == True:
                            arr_input[i][j] = False
                            row_arr_corr.append(False)
                            continue
                    if j+1<len(arr_input[i]):
                        if arr_input[i+1][j+1] == True:
                            arr_input[i][j] = False
                            row_arr_corr.append(False)
                            continue
                if arr_input[i][j] == True:
                    row_arr_corr.append(True)
                    continue

        star_arr_corr.append(row_arr_corr)
    return(star_arr_corr)



def run_through_sky(datavals, thresh = 5):
    '''
        to run through the datavals and find if pixelval above threshold
    '''
    star_arr = []
    for i in range(0, len(datavals)):
        #loop through all the rows
        row_arr = []
        for j in range(0, len(datavals[i])):
            #loop through the values of the row
            pix_val = datavals[i][j]
            if pix_val > thresh:
                row_arr.append(True)
            else:
                row_arr.append(False)
        star_arr.append(row_arr)

    # print('filtering out multiple stars')
    #now delete where more than one connection (doing all 8 around it)
    corr_star_arr = delete_multiples(star_arr)
    return(corr_star_arr)



def starcounter_m2(datavals):
    '''
        input: data array of the area for which we want to find the number of stars
        Task: to find the stars in the data, and also the size + position of stars
        output: the number of stars, size+position of stars (using different threshold values we can see what is good/bad)
    '''
    #for the array we need to cut the edge so that we can hypothetically find stars that go over the edge
    # so cut 5 pixels of edges
    #just find for each pixel that is above threshold, then later run through, back to front, if touching other pix set to false

    #run the image through the data eval and determine how many stars and where/what shape
    # pd_starcounts_m2 = starcounter_m2(data_img)

    #scrap this

    pass


def read_data(path):
    '''
        input: path to the data
        Task: to read the data
        output: the data as a np.array
    '''
    img = Image.open(path, 'r')
    return(list(img.getdata()))


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

def read_pickle_data(path):
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
    - go through the pixels in this array and decide on a way to classify a threshold above which something is considered a star
    - find a way to find if pixels around this pixel are also bright -> if yes then counts as only one star
    - determine the number of stars in this array and return it
    - since we are looking for the pixels around it that are also bright we can get data on the shape/size of stars


    '''
    '''
    path_img = '/Users/mauritz/Desktop/image-20201126175626.png'
    #read the data
    data_img = read_data(path_img)
    data_img = np.array(data_img).reshape(2080, 3096)
    #cut the image
    '''

    path_img = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/data_out/20201118230658_eval.pickle'

    data_img = read_pickle_data(path_img)
    data_img_cut = cut_data(data_img)
    '''
    #set a threshold
    threshold = 40
    threshold = np.median(data_img_cut) + 5*np.std(data_img_cut)

    #run the analysis
    data_corr = run_through_sky(data_img_cut, threshold)    #returns the masked array of the locations of stars

    stars = 0
    for i in range(0, len(data_corr)):
        xx = [x for x in data_corr[i] if x == True]
        stars += len(xx)

    print('Image {0}'.format(path_img[-20:]))
    print('Image Diemsions {0} x {1}  <=> {2} area'.format(len(data_img_cut), len(data_img_cut[0]), len(data_img_cut[0]) * len(data_img_cut)))
    print('threshold set: ', threshold)
    print('Stars counted = ', stars)

    '''

    ################### SEE how different thresholds calculate a different number of stars ###################

    # Note, there is no point ind having non integer thresholds because pixel values are integers

    print('FOR THRESHOLDS BETWEEN 10 and 250')
    # data_img = data_img[800:1400, 1200:2000]
    # plt.imshow(data_img, cmap='gray')
    # plt.show()
    trs = np.linspace(10, 250, 25)

    trs = np.linspace(np.min(data_img_cut), np.max(data_img_cut), 200)
    # 10000 - 20000
    med_trs = np.median(data_img_cut)
    std_trs = np.std(data_img_cut)
    trs = [med_trs, med_trs + std_trs, med_trs + 2*std_trs, med_trs + 3*std_trs, med_trs + 4*std_trs, med_trs + 5*std_trs, med_trs + 6*std_trs]


    tots_thrs = []
    for threshold in trs:
        print(threshold)
        data_corr = run_through_sky(data_img_cut, threshold)
        # print(data_corr)
        tots = 0
        for i in range(0, len(data_corr)):
            xx = [x for x in data_corr[i] if x == True]
            tots += len(xx)
        tots_thrs.append(tots)

    lbls = ['Med', 'Med + std', 'Med + 2*std', 'Med + 3*std', 'Med + 4*std', 'Med + 5*std', 'Med + 6*std']
    plt.scatter(trs, tots_thrs, color = 'darkblue', label = lbls)
    plt.xlabel('Threshold')
    plt.legend()
    plt.show()

    # plt.bar(trs, tots_thrs, color = 'darkblue')
    # plt.xlabel('Threshold')
    # plt.show()

    # print('FOR THRESHOLDS BETWEEN 40 and 50')
    # data_img = data_img[800:1400, 1200:2000]
    # # plt.imshow(data_img, cmap='gray')
    # # plt.show()
    # trs = np.linspace(40, 50, 20)
    # tots_thrs = []
    # for threshold in trs:
    #     print(threshold)
    #     data_corr = run_through_sky(data_img, threshold)
    #     # print(data_corr)
    #     tots = 0
    #     for i in range(0, len(data_corr)):
    #         xx = [x for x in data_corr[i] if x == True]
    #         tots += len(xx)
    #     tots_thrs.append(tots)

    # plt.scatter(trs, tots_thrs)
    # plt.show()
    # quit()

    ##################################################################################################################


    ##################### See where the stars are observerd (dots on imshow image) ##############################
    '''
    # data_img = data_img[800:1400, 1200:2000]

    # threshold = 70
    # data_corr = run_through_sky(data_img_cut, threshold)
    # # print(data_corr)
    # tots = 0
    # for i in range(0, len(data_corr)):
    #     xx = [x for x in data_corr[i] if x == True]
    #     tots += len(xx)
    # print(tots)
    plt.imshow(data_img_cut,  cmap = 'gray')
    for i in range(0, len(data_corr)):
        for j in range(0, len(data_corr)):
            if data_corr[i][j] == True:
                data_corr[i][j] = 250
            else:
                data_corr[i][j] = 0
    plt.imshow(data_corr, alpha = 0.3)
    plt.show()
    quit()
    '''
    ##################################################################################################################

    ########


    #Done
    t_fin = time.perf_counter()
    print('\n \n \n********* Process-{0} over ************'.format(pid))
    print('Runtime: {0} seconds'.format(t_fin - t_start))
    p_mem = process.memory_info().rss
    print('Memory used: {0} bytes ({1} GB) '.format(p_mem, p_mem/(1e9)))
