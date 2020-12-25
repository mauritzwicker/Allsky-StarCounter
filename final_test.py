####### Final test for three methods:

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

import parameters

# for starfinder
import final1_starfind
#for threshold
import final2_thresh
# for nn
import final3_nn



class Image:
    def __init__(self, file_name, dir_path):
        self.file_name_xml = file_name
        self.directory_xml = dir_path

        #get the date and time of the exposure
        self.get_datetime()

        #get the data from the xml file
        self.get_xml_dats()

    def get_datetime(self):
        ''' To get the information about date and time of Image'''
        # date_time := the YMDHMS in the file name
        date_time = self.file_name_xml[9:-4]
        # these are the YMDHMS extracted from the filename
        self.year = date_time[0:4]
        self.month = date_time[4:6]
        self.day = date_time[6:8]
        self.hour = date_time[8:10]
        self.minute = date_time[10:12]
        self.second = date_time[12:14]
        self.date_time = date_time

    def get_xml_dats(self):
        '''To get the xml data and format it'''
        # tree_xmlimage := is the xml_parsed_obxect for our xml data
        try:
            self.tree_xmlimage = ElementTree.parse(self.directory_xml + self.file_name_xml)
            # self.root_xmlimage := is the root of the xml parsed data
            self.root_xmlimage = self.tree_xmlimage.getroot()
            # sind we only have one root element we need to select this one
            #self.att_xmlimage := is the attribute we are looking at (only available one)
            self.att_xmlimage = self.root_xmlimage[0]
        except:
            print('error parsing')
            print(self.directory_xml)
            print(self.file_name_xml)
            quit()
            # #move the files so away in bad files folder
            # dict_remove = {'file_name_png': self.file_name_png, 'file_name_xml': self.file_name_xml, 'file_name_txt': self.file_name_txt}
            # move_bad = saver.move_paramBAD_img(dict_remove, self.params)
            # if move_bad:
            #     print('moved bad image')
            #     print('quitting')
            #     quit()
            # else:
            #     print('unable to move bad image')
            #     print('quitting')
            #     quit()

        #now we get the data from our attribute
        # **** THIS DEPENDS ON HOW THE XML FILES ARE SAVED / FORMATTED ****
        # xml_rows := the number of rows (value is saved at top of xml)
        self.xml_rows = int(self.att_xmlimage.find('rows').text)
        # xml_cols := the number of columns (value is saved at top of xml)
        self.xml_cols = int(self.att_xmlimage.find('cols').text)
        # xml_counts := the data, counts for each pixel
        self.xml_counts = self.att_xmlimage.find('data').text
        # we want to reformat xml_counts so that it is a 2d array
        # so that each data val = 1 pixels' photon count value
        self.xml_counts = np.fromstring(self.xml_counts, dtype=int, sep=' ')
        self.xml_counts = np.reshape(self.xml_counts,(-1, self.xml_cols))

        return



def cut_and_crop(obj):
    # now we need to cut it up into smaller sections
    # cut up into 20x20 pixels
    size = 20
    xml_cut_counts = []
    for i in range(0, int(len(obj)/size)):
        for j in range(0, int(len(obj[0])/size)):
            cut_arr = obj[i*20 : i*20 + 20, j*20 : j*20 + 20]
            xml_cut_counts.append(cut_arr)
    return(xml_cut_counts)

#for starfinder:
def starfinder(data_img, meth1):
    thresh = 3
    fwhm = 5
    data_img = final1_starfind.background_noise_est(data_img)
    m1_sources = final1_starfind.source_detection(data_img, fwhm, thresh)
    if m1_sources > 700:
        m1_cloudy = False
    else:
        m1_cloudy = True

    img_res1 = [m1_sources, m1_cloudy]
    meth1.append(img_res1)
    return(meth1)


#for classifier:
def classifier(data_img, meth2):
    trs = np.median(data_img) + 3 * np.std(data_img)

    data_corr = final2_thresh.run_through_sky(data_img, trs)

    m2_sources = 0
    for i in range(0, len(data_corr)):
        xx = [x for x in data_corr[i] if x == True]
        m2_sources += len(xx)

    if m2_sources > 1000:
        m2_cloudy = False
    else:
        m2_cloudy = True

    img_res2 = [m2_sources, m2_cloudy]
    meth2.append(img_res2)
    return(meth2)



#for nn:
def nn(data_img_cutt, m3_sources):
    m3_sources = 0
    weights = [-1.21591061e+00, -3.98304460e+00, 2.58400083e+00, -2.18382695e+00, -2.98174183e-03]
    bias = 0.6367690212833454
    #loop through all the cut sections
    for arr in data_img_cutt:
        arr = arr / np.max(data_img_cutt)
        inputs = [np.average(arr), np.median(arr), np.max(arr), np.min(arr),  np.std(arr)]
        z = weights[0] * inputs[0] + weights[1] * inputs[1] + weights[2] * inputs[2] + weights[3] * inputs[3] + weights[4] * inputs[4] + bias
        y_prediction = final3_nn.sigmoid(z)
        if y_prediction >=0.5:
            m3_sources +=1
        else:
            continue

    if m3_sources > 300:
        m3_cloudy = False
    else:
        m3_cloudy = True

    img_res3 = [m3_sources, m3_cloudy]
    meth3.append(img_res3)
    return(meth3)



#to hold the final data (# of star, and True/False for True if clear, false if cloudy)
meth1 = []
meth2 = []
meth3 = []



#LOAD THE DATA
#for starfinder we need it in the cut array (700x700)
#for threshold we need array (700x700)
#for nn we need array (700x700) cut into shapes of like 20x19 or so

# Image 1: Somewhat cloudy
dir_path1 = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/Raw_Daten/rawww20201118/'
file_name1 = 'xml_data-20201118215711.xml'
img1 = Image(file_name1, dir_path1)
data_img1 = img1.xml_counts
data_img1 = data_img1[780:1480, 950:1650]
data_img_cutt1 = cut_and_crop(data_img1)

# Image 2: strong moon
dir_path2 = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/Raw_Daten/20201126/'
file_name2 = 'xml_data-20201126190702.xml'
img2 = Image(file_name2, dir_path2)
data_img2 = img2.xml_counts
data_img2 = data_img2[800:1500, 1200:1900]
data_img_cutt2 = cut_and_crop(data_img2)

# Image 3: many stars
dir_path3 = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/Raw_Daten/rawww20201118/'
file_name3 = 'xml_data-20201118211011.xml'
img3 = Image(file_name3, dir_path3)
data_img3 = img3.xml_counts
data_img3 = data_img3[800:1500, 900:1600]
data_img_cutt3 = cut_and_crop(data_img3)

# Image 4: few stars
dir_path4 = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/Raw_Daten/rawww20201118/'
file_name4 = 'xml_data-20201118204907.xml'
img4 = Image(file_name4, dir_path4)
data_img4 = img4.xml_counts
data_img4 = data_img4[800:1500, 1000:1700]
data_img_cutt4 = cut_and_crop(data_img4)



#call the functions
'''
starfinder(filename1, meth1)
classifier(filename1, meth2)
nn(filename1, meth3)
'''

# Image 1: Somewhat cloudy
print('STARTING WITH IMAGE 1')
meth1 = starfinder(data_img1, meth1)
meth2 = classifier(data_img1, meth2)
meth3 = nn(data_img_cutt1, meth3)
print('DONE WITH IMAGE 1')

# Image 2: strong moon
print('STARTING WITH IMAGE 2')
meth1 = starfinder(data_img2, meth1)
meth2 = classifier(data_img2, meth2)
meth3 = nn(data_img_cutt2, meth3)
print('DONE WITH IMAGE 2')

# Image 3: many stars
print('STARTING WITH IMAGE 3')
meth1 = starfinder(data_img3, meth1)
meth2 = classifier(data_img3, meth2)
meth3 = nn(data_img_cutt3, meth3)
print('DONE WITH IMAGE 3')

# Image 4: few stars
print('STARTING WITH IMAGE 4')
meth1 = starfinder(data_img4, meth1)
meth2 = classifier(data_img4, meth2)
meth3 = nn(data_img_cutt4, meth3)
print('DONE WITH IMAGE 4')

print()
print('------------------------------------------------')
print('TEST RESULTS')
print('------------------------------------------------')
print('FOR SOMEWHAT CLOUDY')
print('#.           Stars?          Cloudy?')
print('1. SF ---    {0}.           {1}'.format(meth1[0][0], meth1[0][1]))
print('2. CL ---    {0}.           {1}'.format(meth2[0][0], meth2[0][1]))
print('3. NN ---    {0}.           {1}'.format(meth3[0][0], meth3[0][1]))
print()
print('FOR STRONG MOON')
print('#.           Stars?          Cloudy?')
print('1. SF ---    {0}.           {1}'.format(meth1[1][0], meth1[1][1]))
print('2. CL ---    {0}.           {1}'.format(meth2[1][0], meth2[1][1]))
print('3. NN ---    {0}.           {1}'.format(meth3[1][0], meth3[1][1]))
print()
print('FOR MANY STARS')
print('#.           Stars?          Cloudy?')
print('1. SF ---    {0}.           {1}'.format(meth1[2][0], meth1[2][1]))
print('2. CL ---    {0}.           {1}'.format(meth2[2][0], meth2[2][1]))
print('3. NN ---    {0}.           {1}'.format(meth3[2][0], meth3[2][1]))
print()
print('FOR FEW STARS')
print('#.           Stars?          Cloudy?')
print('1. SF ---    {0}.           {1}'.format(meth1[3][0], meth1[3][1]))
print('2. CL ---    {0}.           {1}'.format(meth2[3][0], meth2[3][1]))
print('3. NN ---    {0}.           {1}'.format(meth3[3][0], meth3[3][1]))
print()


    ########
