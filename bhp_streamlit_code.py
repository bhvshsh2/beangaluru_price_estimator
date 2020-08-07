import pandas as pd
import numpy as np
import streamlit as st 
import json


st.title('''
    Bangaluru Home Price Estimator 
      
      This WebApp is powered by Streamlit & Machine Learning Concepts''')
bhk=st.slider('No of bedrooms',1,5,2)
st.write('No of bedrooms are',bhk)
bath=st.slider('No of bathrooms',1,5,2)
sqft=st.slider('No of squarefeet',0,25000,1000)
file1=open('columns.json')
js_data=json.load(file1)
#type(js_data.items())
loc1=[]
for i in js_data:
    loc1.append(js_data[i])
for x in loc1:
    print(x)
x=tuple(x)
x=x[3:]

location=st.selectbox('choose the location',x)
file1.close()
#js_data
#model training
df=pd.read_csv('cleaned_bhp.csv')
X = df.drop(['price'],axis='columns')
y = df.price
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]
    #predict_price('Rajaji Nagar',1010,2,2)
    
#print(result)
result=predict_price(location,sqft,bath,bhk)
if result < 0:
    st.write('price of property is: ',-result*100000)
else:
    st.write('price of property is: ',result*100000)

#print(result)
