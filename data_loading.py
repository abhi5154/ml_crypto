# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:34:51 2021

@author: Abhishek
"""

import os
import pandas as pd
import numpy as np

base_folder = "C://Users//Abhishek//Desktop//WQU//CAPSTONE//PROJECT"
data_folder = "C://Users//Abhishek//Desktop//WQU//CAPSTONE//PROJECT//DATA"


def base_data_loader(asset_name,asset_type,time_frame):
    data_folder2 = data_folder + "//" + asset_type + "//" + time_frame 
    
    os.chdir(data_folder2)
    total_data               = pd.read_csv(asset_name+ ".csv")
    total_data['close_time'] = pd.to_datetime(total_data['close_time'])
    total_data['open_time']  = pd.to_datetime(total_data['open_time'])
    
    total_data['close_time'] = total_data['close_time'].dt.round('30min')
    total_data['open_time']  = total_data['open_time'].dt.round('30min')
    total_data               = total_data.set_index('close_time' )
    os.chdir(base_folder)
    
    total_data = total_data[['open','high','low','close','volume','trades']]
    
    return (total_data)


