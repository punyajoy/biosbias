# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:52:20 2020

@author: midas
"""

import os
import glob
import pandas as pd
import numpy as np


all_filenames=['Data/Train.csv', 'Data/Test.csv']

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])


combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

from tqdm import tqdm_notebook,tqdm
from sklearn import preprocessing



train_data=pd.read_csv("Data/Train.csv")

test_data=pd.read_csv("Data/Test.csv")


train_wo_g=[]
train_w_g=[]

test_wo_g=[]
test_w_g=[]

combined_csv
for index,row in tqdm(combined_csv.iterrows()):
    try:
        index_to_start=int(row['start_pos'])
    except:
        continue
    tuple1= [row['raw'][index_to_start:],row['title'],row['gender']]
    tuple2= [row['bio'][index_to_start:],row['title'],row['gender']]
    train_w_g.append(tuple1)
    train_wo_g.append(tuple2)


TrainTestWithGen = pd.DataFrame(train_w_g, columns =['Text', 'title', 'gender']) 

TrainTestWithoutGen= pd.DataFrame(train_wo_g, columns =['Text', 'title', 'gender'])


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 74595):
    review = re.sub('[^a-zA-Z]', ' ', TrainTestWithGen['Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 30000)
X = cv.fit_transform(corpus).toarray()

X_all=pd.DataFrame(X)
X_all['title']=TrainTestWithGen['title']
X_all['gender']=TrainTestWithGen['gender']


X_Train=X_all[:53754]
X_Test=X_all[53754:]


X_Train.to_csv('Train_With_Gen.csv')
X_Test.to_csv('Test_With_Gen.csv')


#Without Gender

corpus2 = []
for i in range(0, len(TrainTestWithGen)):
    review = re.sub('[^a-zA-Z]', ' ', TrainTestWithGen['Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus2.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv2 = CountVectorizer(max_features = 30000)
X2 = cv2.fit_transform(corpus2).toarray()

X_all2=pd.DataFrame(X2)
X_all2['title']=TrainTestWithoutGen['title']
X_all2['gender']=TrainTestWithoutGen['gender']


X_Train2=X_all2[:53754]
X_Test2=X_all2[53754:]


X_Train2.to_csv('Train_WithOut_Gen.csv')
X_Test2.to_csv('Test_WithOut_Gen.csv')

