# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:34:52 2021

@author: hasan
"""
# 1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2.veri onisleme
# 2.1.veri yukleme
veriler=pd.read_csv('satislar.csv')

#test
print(veriler)

aylar=veriler[['Aylar']]
print(aylar)

satislar=veriler[['Satislar']]
print(satislar)



#veri kumesinin egitim ve test olarak bolunmesi

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)
'''
#oznitelik olcekleme
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
'''
#model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin=lr.predict(x_test)

#model gorsellestirmesi

x_train=x_train.sort_index()    #x_train deki veriler sıralanıyor
y_train=y_train.sort_index()    #y_train deki veriler sıralanıyor

plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")









