# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:35:44 2021

@author: nicole

Description: Code to clean and tokenize pet product data
"""
# 1. Set up
import pickle

path = "C:\\Users\\nicol\\Documents\\2021-2022\\QMSS\\GR5067 NLP\\Group Project\\data\\"
data_pet = pickle.load(open(path + 'data_pet.pkl', 'rb'))


# 2. Clean text
def clean_text(text):
    import re, string, zhon.hanzi
    text1 = re.sub(r'[{}]+'.format(string.punctuation), '', text).strip()
    text2 = re.sub(r'[一二两三四五六七八九十零]+', '', text1).strip()
    text3 = re.sub(r'[{}]+'.format(zhon.hanzi.punctuation), '', text2).strip()
    text4 = re.sub(r'[0-9]+', '', text3).strip().lower()
    text5 = text4.replace(' ', '')
    return(text5)

data_pet['pro_name_clean'] = data_pet.pro_name.apply(clean_text)


# 3. Tokenising
def tokenize(text):
    import jieba
    tokens = [tok[0] for tok in jieba.tokenize(text)]
    text_tokenized = " ".join(tokens)
    return text_tokenized

data_pet['pro_name_tok'] = data_pet.pro_name_clean.apply(tokenize)
pro_names = data_pet[['pro_name', 'pro_name_clean', 'pro_name_tok']] 
pro_names.head() # see differences before and after cleaning/tokenizing

pickle.dump(data_pet, open(path + 'data_pet_tok.pkl', 'wb')) # save as pkl object
