# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:34:52 2021

@author: hasan
"""
# 1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib as plt

# 2.veri onisleme
# 2.1.veri yukleme
veriler=pd.read_csv('eksikveriler.csv')

#test
#print(veriler)

#veri on isleme
boy=veriler [["boy"]]
#print(boy)

boyKilo=veriler[["boy","kilo"]]
#print(boyKilo)


# 3.eksikVeriler
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
Yas=veriler.iloc[:,1:4].values

#print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
#print(Yas)

# 4.encoder: Kategorik->Numeric

ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

# 5.1.numpy dizileri dataframe donusumu

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

# 5.2. dataframe birlestirme
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#veri kumesinin egitim ve test olarak bolunmesi

from sklearn.model_selection import train_test_split

x_train,x_test=train_test_split(s,test_size=0.33,random_state=0)

#oznitelik olcekleme

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)














