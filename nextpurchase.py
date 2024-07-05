from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st


import warnings
warnings.filterwarnings("ignore")

import chart_studio as py
import plotly.offline as pyoff
import plotly.graph_objs as go

from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

st.title('Next Purchase Day Prediction')
#IMPORTS

uploaded_file = st.file_uploader("Upload Data File",type=["xlsx","csv"])

if st.button('Predict Next Purchase Day'):
    #load our data from CSV to tx_data
    tx_data = pd.read_csv(uploaded_file,encoding="latin-1")

    #tx_data = pd.read_csv("C:/Users/PhilBiju(G10XIND)/Downloads/customer_analytics/datasets/online_retail.csv",encoding = 'latin1')


    tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])


    tx_6m = tx_data[(tx_data.InvoiceDate < datetime(2011,9,1)) & (tx_data.InvoiceDate >= datetime(2011,3,1))].reset_index(drop=True)
    tx_next = tx_data[(tx_data.InvoiceDate >= datetime(2011,9,1)) & (tx_data.InvoiceDate < datetime(2011,12,1))].reset_index(drop=True)


    tx_user = pd.DataFrame(tx_6m['CustomerID'].unique())
    tx_user.columns = ['CustomerID']

    #Adding Label
    tx_next_first_purchase = tx_next.groupby('CustomerID').InvoiceDate.min().reset_index()
    tx_next_first_purchase.columns = ['CustomerID','MinPurchaseDate']


    tx_last_purchase = tx_6m.groupby('CustomerID').InvoiceDate.max().reset_index()

    tx_last_purchase.columns = ['CustomerID','MaxPurchaseDate']

    tx_purchase_dates = pd.merge(tx_last_purchase,tx_next_first_purchase,on='CustomerID',how='left')

    tx_purchase_dates['NextPurchaseDay'] = (tx_purchase_dates['MinPurchaseDate'] - tx_purchase_dates['MaxPurchaseDate']).dt.days


    tx_user = pd.merge(tx_user, tx_purchase_dates[['CustomerID','NextPurchaseDay']],on='CustomerID',how='left')


    tx_user = tx_user.fillna(999)

    #recency

    tx_max_purchase = tx_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
    tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
    tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
    tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

    #plotting not done


    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Recency']])
    tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

    def order_cluster(cluster_field_name, target_field_name,df,ascending):
        new_cluster_field_name = 'new_' + cluster_field_name
        df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
        df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
        df_new['index'] = df_new.index
        df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
        df_final = df_final.drop([cluster_field_name],axis=1)
        df_final = df_final.rename(columns={"index":cluster_field_name})
        return df_final

    tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)


    #Frequency 

    tx_frequency = tx_6m.groupby('CustomerID').InvoiceDate.count().reset_index()
    tx_frequency.columns = ['CustomerID','Frequency']
    tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Frequency']])
    tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])
    tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)

    #Monetary Value

    tx_6m['Revenue'] = tx_6m['UnitPrice'] * tx_6m['Quantity']
    tx_revenue = tx_6m.groupby('CustomerID').Revenue.sum().reset_index()
    tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')


    #Plotting not done

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Revenue']])
    tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
    tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)


    #Overall Segmentation

    tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
    tx_user.groupby('OverallScore')[['Recency','Frequency','Revenue']].mean()
    tx_user['Segment'] = 'Low-Value'
    tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
    tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value'

    #create a dataframe with CustomerID and Invoice Date
    tx_day_order = tx_6m[['CustomerID','InvoiceDate']]

    #Convert Invoice Datetime to day
    tx_day_order['InvoiceDay'] = tx_6m['InvoiceDate'].dt.date
    tx_day_order = tx_day_order.sort_values(['CustomerID','InvoiceDate'])

    #Drop duplicates
    tx_day_order = tx_day_order.drop_duplicates(subset=['CustomerID','InvoiceDay'],keep='first')

    #shifting last 3 purchase dates
    tx_day_order['PrevInvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(1)
    tx_day_order['T2InvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(2)
    tx_day_order['T3InvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(3)

    tx_day_order['InvoiceDay'] = pd.to_datetime(tx_day_order['InvoiceDay'])

    # Checking the data type of the columns

    # If they are not datetime64[ns], convert them
    date_columns = ['T2InvoiceDate', 'T3InvoiceDate', 'PrevInvoiceDate']
    for col in date_columns:
        if tx_day_order[col].dtype != 'datetime64[ns]':
            tx_day_order[col] = pd.to_datetime(tx_day_order[col])

    # Drop rows with missing dates
    tx_day_order.dropna(subset=date_columns, inplace=True)

    # Recalculate the differences in days after ensuring all are datetime type
    if 'InvoiceDay' in tx_day_order.columns and 'PrevInvoiceDate' in tx_day_order.columns:
        tx_day_order['DayDiff'] = (tx_day_order['InvoiceDay'] - tx_day_order['PrevInvoiceDate']).dt.days

    if 'InvoiceDay' in tx_day_order.columns and 'T2InvoiceDate' in tx_day_order.columns:
        tx_day_order['DayDiff2'] = (tx_day_order['InvoiceDay'] - tx_day_order['T2InvoiceDate']).dt.days

    if 'InvoiceDay' in tx_day_order.columns and 'T3InvoiceDate' in tx_day_order.columns:
        tx_day_order['DayDiff3'] = (tx_day_order['InvoiceDay'] - tx_day_order['T3InvoiceDate']).dt.days

    # Output to verify the changes


    tx_day_diff = tx_day_order.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()

    tx_day_diff.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']


    tx_day_order_last = tx_day_order.drop_duplicates(subset=['CustomerID'],keep='last')


    tx_day_order_last = tx_day_order_last.dropna()

    tx_day_order_last = pd.merge(tx_day_order_last, tx_day_diff, on='CustomerID')

    tx_user = pd.merge(tx_user, tx_day_order_last[['CustomerID','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='CustomerID')
    

    #Grouping the Label
    tx_class = tx_user.copy()
    tx_class = pd.get_dummies(tx_class)
    


    tx_class['NextPurchaseDayRange'] = 2
    tx_class.loc[tx_class.NextPurchaseDay>20,'NextPurchaseDayRange'] = 1
    tx_class.loc[tx_class.NextPurchaseDay>50,'NextPurchaseDayRange'] = 0

    corr = tx_class[tx_class.columns].corr()
    fig, ax = plt.subplots(figsize = (30,20))
    sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f", ax=ax)

    st.header('Correlation Matrix of Features', divider='orange')
    st.pyplot(fig)
    st.header('Customer Clusters: Next Purchase Window', divider='orange')
    st.write(tx_class)
    st.write("Customers that will purchase in 0–20 days — Class name: 2")
    st.write("Customers that will purchase in 21–49 days — Class name: 1")
    st.write("Customers that will purchase in more than 50 days — Class name: 0")
    tx_class = tx_class.drop('NextPurchaseDay',axis=1)

    X, y = tx_class.drop('NextPurchaseDayRange',axis=1), tx_class.NextPurchaseDayRange
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
    models = []
    models.append(("LR",LogisticRegression()))
    models.append(("NB",GaussianNB()))
    models.append(("RF",RandomForestClassifier()))
    models.append(("SVC",SVC()))
    models.append(("Dtree",DecisionTreeClassifier()))
    models.append(("XGB",xgb.XGBClassifier()))
    models.append(("KNN",KNeighborsClassifier()))

    from sklearn.impute import SimpleImputer


    for name,model in models:
        # Impute missing values in X_train
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        kfold = KFold(n_splits=2, shuffle=True, random_state=22)
        cv_result = cross_val_score(model,X_train_imputed,y_train, cv = kfold,scoring = "accuracy")
     
    xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    from sklearn.model_selection import GridSearchCV

    param_test1 = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
    }
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(), 
    param_grid = param_test1, scoring='accuracy',n_jobs=-1,cv=2)
    gsearch1.fit(X_train,y_train)
    #gsearch1.best_params_, gsearch1.best_score_

    xgb_model = xgb.XGBClassifier(max_depth=3, min_child_weight=5).fit(X_train, y_train)
    st.header('Model Training Results', divider='orange')
    st.write('Accuracy of XGB classifier on training set: {:.2f}'
        .format(xgb_model.score(X_train, y_train)))
    st.write('Accuracy of XGB classifier on test set: {:.2f}'
        .format(xgb_model.score(X_test[X_train.columns], y_test)))







