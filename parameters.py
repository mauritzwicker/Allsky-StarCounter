##########################################
'''

author: @mauritzwicker
date: 28.11.2020
repo: star_detection_Allsky

Purpose: to define what data is available/ what we want/ what to do

'''
##########################################

########
#Imports
import time
import psutil
import os

########


########


class Parameters_allsky:

    def __init__(self):

        #User defined parameters

        # Locations
        self.input_dir_xml = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/Raw_Daten/rawww20201118/'         #directory where xml data is for input
        self.input_dir_txt = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/Raw_Daten/rawww20201118/'         #directory where txt data is for input
        self.input_dir_png = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/Bilder/20201118/'                 #directory where png data is
        self.output_dir_dats = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/data_out/'            #directory where outputs should be saved data
        self.move_dir_xml = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/xml_move/'            #directory where input moved be saved xml
        self.move_dir_txt = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/txt_move/'            #directory where input moved be saved txt
        self.move_dir_png = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/png_move/'            #directory where input moved be saved png

        if not os.path.exists(self.output_dir_dats):
            os.makedirs(self.output_dir_dats)
        if not os.path.exists(self.move_dir_xml):
            os.makedirs(self.move_dir_xml)
        if not os.path.exists(self.move_dir_txt):
            os.makedirs(self.move_dir_txt)
        if not os.path.exists(self.move_dir_png):
            os.makedirs(self.move_dir_png)

        self.move_dir_xml_bad = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/xml_move/bad_files/'            #directory where input should be moved xml when bad
        self.move_dir_txt_bad = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/txt_move/bad_files/'            #directory where input should be moved txt when bad
        self.move_dir_png_bad = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/png_move/bad_files/'            #directory where input should be moved png when bad

        if not os.path.exists(self.move_dir_xml_bad):
            os.makedirs(self.move_dir_xml_bad)
        if not os.path.exists(self.move_dir_txt_bad):
            os.makedirs(self.move_dir_txt_bad)
        if not os.path.exists(self.move_dir_png_bad):
            os.makedirs(self.move_dir_png_bad)

        # What is available (for now we need all three but for usablitiy include check at somepoint)
        self.data_exists_xml = True     #where the xml data is available (True=Yes // False=No)
        self.data_exists_txt = True     #where the txt data is available (True=Yes // False=No)
        self.data_exists_png = True     #where the png data is available (True=Yes // False=No)

        # What to show?
        pass



########






if __name__ == '__main__':
    t_start = time.perf_counter()
    pid = os.getpid()
    process = psutil.Process(pid)


    ########
    # CODE
    ########


    #Done
    t_fin = time.perf_counter()
    print('\n \n \n********* Process-{0} over ************'.format(pid))
    print('Runtime: {0} seconds'.format(t_fin - t_start))
    p_mem = process.memory_info().rss
    print('Memory used: {0} bytes ({1} GB) '.format(p_mem, p_mem/(1e9)))
