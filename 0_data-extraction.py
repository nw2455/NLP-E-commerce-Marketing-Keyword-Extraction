# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:40:57 2021

@author: nicol
"""
# 1. Set up
import pandas as pd
import json
import pickle

path = "C:\\Users\\nicol\\Documents\\2021-2022\\QMSS\\GR5067 NLP\\Group Project\\data\\"


# 2. Extract original data from json
with open(path + 'data_orig.json', encoding='utf-8', errors='ignore') as json_data:
    data = json.load(json_data, strict=False)
    
from pandas.io.json import json_normalize
df1 = pd.json_normalize(data, max_level=0)
pickle.dump(df1, open(path + 'data_orig.pkl', 'wb'))


# 3. Obtain downsized df for analysis
data_orig = pickle.load(open('.\\data\\data_orig.pkl', 'rb'))
data = data_orig[['id', 'task_group', 'site_name', 'shop_name',
           'pro_code', 'pro_name', 'pro_brand',
           'pro_type', 'pro_class', 'class_strong', 'class_small',
           'sale_price', 'pro_sales_num']]
pickle.dump(data, open(path + 'data.pkl', 'wb'))


# 4. Exploring columns to see what column to filter on to obtain df of pet-related products
class_strong_counts = data.class_strong.value_counts()
class_small_counts = data.class_small.value_counts()
pro_type_counts = data.pro_type.value_counts()
pro_class_counts = data.pro_class.value_counts()

data_pet = data[data['pro_class'] == '宠物生活']
data_pet = data_pet.drop(['pro_type'], axis=1)
pet_class_strong_counts = data_pet.class_strong.value_counts()
data_pet_filtered = data_pet[data_pet['class_strong'] != '宠物活体'] # removing live animal products which we're not interested in
data_pet_filtered = data_pet_filtered.reset_index(drop=True)
pickle.dump(data_pet_filtered, open(path + 'data_pet.pkl', 'wb'))
