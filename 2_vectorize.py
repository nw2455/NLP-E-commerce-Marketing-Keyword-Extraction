# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:35:44 2021

@author: naijia, jackie

Description: Code to vectorize tokenized pet data
"""
# 1. Set up
import pandas as pd
import numpy as np
import pickle

path = "C:\\Users\\nicol\\Documents\\2021-2022\\QMSS\\GR5067 NLP\\Group Project\\data\\"
data_pet = pickle.load(open(path + 'data_pet_tok.pkl', 'rb'))
pro_names = data_pet[['pro_name', 'pro_name_clean', 'pro_name_tok']] 


# 2. Vectorize with CountVectoriser
def my_vec_fun(df_in, m, n, path_o):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    import pickle
    vectorizer = CountVectorizer(ngram_range=(m, n))
    my_vec_t = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_vec_t.columns = vectorizer.vocabulary_
    pickle.dump(vectorizer, open(path_o + "countvec.pkl", "wb" ))
    return my_vec_t

my_vec_data_1_1 = my_vec_fun(pro_names.pro_name_tok, 1, 1, path) # start with unigram to prevent overfitting
# my_vec_data_1_2 = my_vec_fun(pro_names.pro_name_tok, 1, 2, path)
# my_vec_data_2_2 = my_vec_fun(pro_names.pro_name_tok, 2, 2, path)

# 2.1 Output count vectorized data as pickle object
pickle.dump(my_vec_data_1_1, open(path + 'my_vec_data_1_1.pkl', 'wb')) 
# pickle.dump(my_vec_data_1_2, open(path + 'my_vec_data_1_2.pkl', 'wb')) 
# pickle.dump(my_vec_data_2_2, open(path + 'my_vec_data_2_2.pkl', 'wb'))


# 3. Vectorize with TF-IDF
def my_tfidf_fun(df_in, m, n, path_o):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import pickle
    vectorizer = TfidfVectorizer(ngram_range=(m, n))
    my_tfidf_t = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_tfidf_t.columns = vectorizer.vocabulary_
    pickle.dump(vectorizer, open(path_o + "tfidf.pkl", "wb" ))
    return my_tfidf_t

my_tfidf_data_1_1 = my_tfidf_fun(pro_names.pro_name_tok, 1, 1, path) # start with unigram to prevent overfitting
# my_tfidf_data_1_2 = my_tfidf_fun(pro_names.pro_name_tok, 1, 2, path)
# my_tfidf_data_2_2 = my_tfidf_fun(pro_names.pro_name_tok, 2, 2, path)

# 2.1 Output tfidf vectorized data as pickle object
pickle.dump(my_tfidf_data_1_1, open(path + 'my_tfidf_data_1_1.pkl', 'wb')) 
# pickle.dump(my_tfidf_data_1_2, open(path + 'my_tfidf_data_1_2.pkl', 'wb')) 
# pickle.dump(my_tfidf_data_2_2, open(path + 'my_tfidf_data_2_2.pkl', 'wb')) 

## 3.1 Code to obtain top 20 terms (based on TF-IDF) - optional code
import jieba.analyse
pro_names_all = " ".join(pro_names.pro_name_tok)
pro_names_top20 = jieba.analyse.extract_tags(pro_names_all, 
                                            topK=20, withWeight=False, allowPOS=())
print(pro_names_top20)




