##########################################
'''
author: @mauritzwicker
date: 03.12.2020

Purpose: METHOD 4: using single neuron

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
import pandas as pd
#from other files
from parameters import *
import pickle


########




########


# new:


def sigmoid(x):
    '''  the sigmoid function '''
    return(1/(1+np.exp(-x)))

def d_sigmoid(x):
    ''' the derivative of the sigmoid function '''
    return(sigmoid(x) * (1-sigmoid(x)))

def cost(val, trueval):
    ''' determine the cost '''
    return(np.square(float(val)-float(trueval)))

def d_cost(val, trueval):
    ''' derivative of the cost '''
    return(2*(float(val)-float(trueval)))

def dz_dw(img_obj, ind):
    ''' the derivative of z by a weight '''
    return(img_obj[ind])

def dz_db():
    ''' the derivative of z by the bias '''
    return(1)

def dcost_dw(d_cost_val, d_pred_z, img_obj, ind_w):
    ''' the derivative of the cost function by the weight '''
    return(d_cost_val * d_pred_z * dz_dw(img_obj,ind_w))

def dcost_db(d_cost_val, d_pred_z):
    ''' the derivative of the cost function by the bias '''
    return(d_cost_val * d_pred_z * dz_db())



class NN_1layer:
    def __init__(self, data_object):
        self.xy_data = np.array(data_object.all_data_xy)

        self.resultssave = []


        self.X = np.array([[0,0,0,0], [0,1,0,0]])    # the pixel counts data (size is 1000x(20x19)) -> flatten so 20x19 becomes 380
        self.y = np.array([0,1])    # the classifications (size is 1000)
        self.split_train_test()


        print(len(self.X))
        print(len(self.X_train))
        print(len(self.X_test))

        # self.epochs = int(i)
        # self.learning_rate = j

        self.ratio_testrain = 0.2
        self.epochs = 15
        self.learning_rate = 0.1
        self.costs = np.array([])
        self.predictions = np.array([])
        self.count_wrong = 0    #to count how many predictions were wrong -> error

        self.weights = np.array([])
        self.bias = np.random.randn()
        self.init_weights()

        #now we run through all the trianing data
        self.train_data()

        print('DONE WITH TRAINING')

        print(self.weights)
        print(self.bias)
        quit()
        #now we run through and test the data
        # self.test_data()

        #save the data
        data_transposed = self.resultssave
        df = pd.DataFrame(data_transposed, columns=["lr", "ep", "tot", "wrong"])
        print(df.head())
        # quit()
        df.to_pickle('/Users/mauritz/Documents/Uni/Bachelorarbeit/Git_Repo_Allsky/test_results_stats_3.pkl')

    def split_train_test(self):
        ''' to split the data into test and train sections '''
        # self.X_train = np.array([])
        # self.y_train = np.array([])
        # self.X_test = np.array([])
        # self.y_test = np.array([])

        # trainingdata_x = [[0,0,0,0],[0,0,0,1],[0,0,0,0],[1,0,0,1],[1,0,0,0],[1,1,0,1],[1,1,0,0],[0,0,1,1],[0,0,0,0],[0,1,0,1],[1,1,0,1],[0,0,0,1],[0,0,1,0]]
        # self.X_train = np.array(trainingdata_x)
        # trainigdata_y = [0,1,0,1,1,1,1,1,0,1,1,1,1]
        # self.y_train = np.array(trainigdata_y)
        # test_x = [[0,0,1,0], [0,0,0,0], [0,0,0,1]]
        # self.X_test = np.array(test_x)
        # test_y = [1,0,1]
        # self.y_test = np.array(test_y)



        # self.xy_data = self.xy_data.reshape(1049, 2)
        X_all = self.xy_data[1:, 0]
        X_all_reshaped = []
        for i in range(0, len(X_all)):
            X_all_reshaped.append(X_all[i].flatten().reshape(380)/np.max(X_all[i]))
        X_all_reshaped = np.array(X_all_reshaped)

        self.X = X_all_reshaped

        y_all = self.xy_data[1:, 1]
        y_all_reshaped = []
        for i in range(0, len(y_all)):
            y_all_reshaped.append(y_all[i][0])
        y_all_reshaped = np.array(y_all_reshaped)

        #need to redefine y_all_reshaped so that it is 1 or 0
        y_all_reshaped[y_all_reshaped == 'l'] = 1
        y_all_reshaped[y_all_reshaped == 'm'] = 1
        y_all_reshaped[y_all_reshaped == 'o'] = 1
        y_all_reshaped[y_all_reshaped == 's'] = 1
        y_all_reshaped[y_all_reshaped == 'u'] = 1
        y_all_reshaped[y_all_reshaped == 'n'] = 0

        self.y = y_all_reshaped


        self.X_train , self.X_test , self.y_train , self.y_test = model_selection.train_test_split(X_all_reshaped, y_all_reshaped,
                                                                    test_size=0.4, random_state=0)

        ### correct the data to hold information: mean, median, max, min, std
        arr_corr_X_train = []
        for arr in self.X_train:
            new_train = [np.average(arr), np.median(arr), np.max(arr), np.min(arr), np.std(arr)]
            arr_corr_X_train.append(new_train)
        self.corr_X_train = np.array(arr_corr_X_train)

        self.corr_y_train = self.y_train

        arr_corr_X_test = []
        for arr in self.X_test:
            new_train = [np.average(arr), np.median(arr), np.max(arr), np.min(arr), np.std(arr)]
            arr_corr_X_test.append(new_train)
        self.corr_X_test = np.array(arr_corr_X_test)

        self.corr_y_test = self.y_test


        self.X = np.append(self.corr_X_train, self.corr_X_test).reshape(-1, 5)
        self.y = np.append(self.corr_y_train, self.corr_y_test).reshape(-1, 1)

        self.X_train = self.corr_X_train
        self.y_train = self.corr_y_train
        self.X_test = self.corr_X_test
        self.y_test = self.corr_y_test


        # print(self.corr_X_train.shape)
        # print(self.corr_y_train.shape)
        # print(self.corr_X_test.shape)
        # print(self.corr_y_test.shape)



    def init_weights(self):
        ''' to initiate the weights and bias of the function '''
        for i in range(0, self.corr_X_train.shape[1]):
            self.weights = np.append(self.weights, np.random.randn())

    def z(self, X_img):
        ''' input one array of size 20x19 and run it through the z function '''
        z = 0
        for i in range(0, self.X.shape[1]):
            z += X_img[i] * self.weights[i]
        z += self.bias
        return(z)

    # def sigmoid(x):
    #     '''  the sigmoid function '''
    #     return(1/(1+np.exp(-x)))

    # def d_sigmoid(x):
    #     ''' the derivative of the sigmoid function '''
    #     return(sigmoid(x) * (1-sigmoid(x)))

    # def cost(val, trueval):
    #     ''' determine the cost '''
    #     return(np.squared(val-trueval))

    # def d_cost(val, trueval):
    #     ''' derivative of the cost '''
    #     return(2*(val-trueval))

    # def dz_dw(img_obj, ind):
    #     ''' the derivative of z by a weight '''
    #     return(img_obj[ind])

    # def dz_db():
    #     ''' the derivative of z by the bias '''
    #     return(1)

    # def dcost_dw(d_cost_val, d_pred_z, img_obj, ind_w):
    #     ''' the derivative of the cost function by the weight '''
    #     return(d_cost_val * d_pred_z * dz_dw(img_obj*ind_w))

    # def dcost_db(d_cost_val, d_pred_z):
    #     ''' the derivative of the cost function by the bias '''
    #     return(d_cost_val * d_pred_z * dz_db)


    def weights_modification(self, d_cost_val, d_pred_z_val, img_obj, ind_w):
        ''' to modify the weights based on the cost determined '''
        for i in range(0, len(self.weights)):
            weight = self.weights[i]
            ind_w = i
            val_dcost_dw = dcost_dw(d_cost_val, d_pred_z_val, img_obj, ind_w)
            weight = weight - self.learning_rate * val_dcost_dw
            self.weights[i] = weight

    def bias_modification(self, d_cost_val, d_pred_z_val):
        ''' to modify the bias based on cost '''
        val_dcost_db = dcost_db(d_cost_val, d_pred_z_val)
        self.bias = self.bias - self.learning_rate * val_dcost_db

    def train_data(self):
        ''' this is to train the data '''
        cost_val = 0
        for j in range(self.epochs):
            print('epoch {0}, last costs {1}'.format(j, cost_val))
            for i in range(0, len(self.X_train)):
                img_obj = self.X_train[i]
                # img_obj = img_obj/np.average(img_obj)
                target_z_val = self.y_train[i]
                #we run it through the function
                z_val = self.z(img_obj)
                #we get the prediction from this
                pred_z_val = sigmoid(z_val)
                d_pred_z_val = d_sigmoid(z_val)
                #now we need to determine the cost
                cost_val = cost(pred_z_val, target_z_val)
                d_cost_val = d_cost(pred_z_val, target_z_val)

                #append the costs and predictions (for following along)
                self.costs = np.append(self.costs, cost_val)
                self.predictions = np.append(self.predictions, pred_z_val)

                self.weights_modification(d_cost_val, d_pred_z_val, img_obj, i)
                self.bias_modification(d_cost_val, d_pred_z_val)


    def test_data(self):
        ''' this is to test our model '''
        for i in range(0, len(self.X_test)):
            img_obj_test = self.X_test[i]
            target_z_val = int(self.y_test[i])

            z_val = self.z(img_obj_test)
            pred_z_val = sigmoid(z_val)
            if pred_z_val >=0.5:
                pred_val_rel = 1
            else:
                pred_val_rel = 0

            # print('Model predicted {0} -- True value {1}'.format(pred_val_rel, target_z_val))


            if pred_val_rel != target_z_val:
                self.count_wrong +=1
            else:
                continue

        print()
        print()
        print('lr;', self.learning_rate)
        print('ep;', self.epochs)
        print('tot;', len(self.X_test))
        print('wrong;', self.count_wrong)
        # print('Out of {0} predictions, predicted {1} wrong'.format(len(self.X_test), self.count_wrong))
        # print('Test Result: {0} % Wrong // {1} % Correct'.format(round(100*self.count_wrong/len(self.X_test), 4), round(100 - 100*self.count_wrong/len(self.X_test), 4)))
        # print('learning_rate = ', self.learning_rate)
        # print('Epochs = ', self.epochs)
        # print('ratio_testrain = ', self.ratio_testrain)
        # print('Out of {0} predictions, predicted {1} wrong'.format(len(self.X_test), self.count_wrong))
        # print('Test Result: {0} % Wrong // {1} % Correct'.format(round(100*self.count_wrong/len(self.X_test), 4), round(100 - 100*self.count_wrong/len(self.X_test), 4)))
        results_to_save = [self.learning_rate, self.epochs, len(self.X_test), self.count_wrong]
        self.resultssave.append(results_to_save)

class DataClassified:
    def __init__(self, path):
        self.path_data_before = path
        #get the data and according classification
        self.get_data_classified()
        #so now our object self.all_data_xy is all our data

    def get_data_classified(self):
        self.txt_file_names_classified = []     #list to hold all the classified text files (sort it)
        self.pixcounts_file_names_classified = []     #list to hold all the classified pixel counts files (sort it)

        classified_files = os.listdir(self.path_data_before)
        for file in classified_files:
            if file.endswith('classification.txt'):
                self.txt_file_names_classified.append(file)
            elif file.endswith('pixeldata.txt'):
                self.pixcounts_file_names_classified.append(file)
            else:
                print('ERROR with {0}'.format(file))

        sorted_txt = np.array([])
        sorted_pixcount = np.array([])
        count_1 = 0
        count_2 = 0
        count_3 = 0
        count_4 = 0

        for i in range(0, len(self.txt_file_names_classified)):
            fil_look_txt = self.txt_file_names_classified[i]
            fil_look_pixcount = self.pixcounts_file_names_classified[i]
            if fil_look_txt[15] == '_':
                print('Error')
            elif fil_look_txt[16] == '_':
                sorted_txt = np.append(sorted_txt, fil_look_txt[15])
            elif fil_look_txt[17] == '_':
                sorted_txt = np.append(sorted_txt, fil_look_txt[15:17])
            elif fil_look_txt[18] == '_':
                sorted_txt = np.append(sorted_txt, fil_look_txt[15:18])
            elif fil_look_txt[19] == '_':
                sorted_txt = np.append(sorted_txt, fil_look_txt[15:19])
            else:
                print('holy fuck how many values do you want to pass???')

            if fil_look_pixcount[15] == '_':
                print('Error')
            elif fil_look_pixcount[16] == '_':
                sorted_pixcount = np.append(sorted_pixcount, fil_look_pixcount[15])
            elif fil_look_pixcount[17] == '_':
                sorted_pixcount = np.append(sorted_pixcount, fil_look_pixcount[15:17])
            elif fil_look_pixcount[18] == '_':
                sorted_pixcount = np.append(sorted_pixcount, fil_look_pixcount[15:18])
            elif fil_look_pixcount[19] == '_':
                sorted_pixcount = np.append(sorted_pixcount, fil_look_pixcount[15:19])
            else:
                print('holy fuck how many values do you want to pass???')

        sorted_txt = sorted_txt.astype(int)
        sorted_txt = np.sort(sorted_txt)
        sorted_txt = sorted_txt.astype(str)
        sorted_pixcount = sorted_pixcount.astype(int)
        sorted_pixcount = np.sort(sorted_pixcount)
        sorted_pixcount = sorted_pixcount.astype(str)

        sorted_txt_filenames = []
        sorted_pixcount_filenames = []

        for i in range(0, len(sorted_txt)):
            sorted_txt_filenames.append('20201118222751_' + str(sorted_txt[i]) + '_classification.txt')
            sorted_pixcount_filenames.append('20201118222751_' + str(sorted_pixcount[i]) + '_pixeldata.txt')


        #so now sorted_txt_filenames and sorted_pixcount_filenames are the lists of the data + classification
        #we combine them into pairs so it is [raw_data, classification]
        all_data_xy_addr = []
        for i in range(0, len(sorted_txt_filenames)):
            new_xy_data_addr = []
            new_xy_data_addr.append(sorted_pixcount_filenames[i])
            new_xy_data_addr.append(sorted_txt_filenames[i])
            all_data_xy_addr.append(new_xy_data_addr)


        self.all_data_xy = []
        for pair in all_data_xy_addr:
            pair_xy = []
            with open(self.path_data_before + '/' + pair[0], "r") as pixels_file:
                dats_pix = np.array([])
                for row in pixels_file:
                    dats_pix = np.append(dats_pix, float(row))
                pair_xy.append(dats_pix.reshape(20, 19))
            with open(self.path_data_before + '/' + pair[1], "r") as txt_file:
                dats_txt = np.array([])
                for row in txt_file:
                    dats_txt = np.append(dats_txt, row)
                pair_xy.append(dats_txt)
            self.all_data_xy.append(pair_xy)

        #all_data_xy is now our complete data, where each element is a list of [pixel data, classification]
        #now we can use this to create X = all the pixel data, Y = classification
        #we can create X_train and y_train and X_test and Y_test





if __name__ == '__main__':
    t_start = time.perf_counter()
    pid = os.getpid()
    process = psutil.Process(pid)


    ########
    data_path = '/Users/mauritz/Documents/git_mauritzwicker/star_detection_Allsky/data_test_20112020/classified'

    #create the object that is the data and path and stufff
    obj1 = DataClassified(data_path)

    # run_nn()
    NN_1layer(obj1)

    ########


    #Done
    t_fin = time.perf_counter()
    print('\n \n \n********* Process-{0} over ************'.format(pid))
    print('Runtime: {0} seconds'.format(t_fin - t_start))
    p_mem = process.memory_info().rss
    print('Memory used: {0} bytes ({1} GB) '.format(p_mem, p_mem/(1e9)))
