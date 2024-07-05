# import libraries
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

#IMPORTS
st.title("Cusotmer Segmentation")
uploaded_file = st.file_uploader("Upload Data File",type=["xlsx","csv"])
if st.button('Segment Customers'):
    #load our data from CSV to tx_data
    tx_data = pd.read_csv(uploaded_file,encoding="latin-1")

    #convert the string date field to datetime
    tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])

    #__________________________________________________________________________________________________________________

    #RECENCY:

    #create a tx_user to keep CustomerID and new segmentation scores
    tx_user = pd.DataFrame(tx_data['CustomerID'].unique())
    tx_user.columns = ['CustomerID']

    #get the max purchase date for each customer and create tx_max_purchse with it
    tx_max_purchase = tx_data.groupby('CustomerID').InvoiceDate.max().reset_index()
    tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']

    #we take our observation point as the max invoice date in our dataset
    tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

    #merge this dataframe to our new user dataframe
    tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')



    #Clustering recency with 4 clusters
    from sklearn.cluster import KMeans


    tx_recency = tx_user[['Recency']]

    #build 4 clusters for recency and add it to dataframe
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Recency']])
    tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

    #function for ordering cluster numbers
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

    #__________________________________________________________________________________________________________________

    #FREQUENCY:

    #get order counts for each user and create a dataframe with it
    tx_frequency = tx_data.groupby('CustomerID').InvoiceDate.count().reset_index()
    tx_frequency.columns = ['CustomerID','Frequency']

    #add this data to our main dataframe
    tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

    #k-means
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Frequency']])
    tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

    #order the frequency cluster
    tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)

    #see details of each cluster


    #__________________________________________________________________________________________________________________

    #REVENUE:

    #calculate revenue for each customer
    tx_data['Revenue'] = tx_data['UnitPrice'] * tx_data['Quantity']
    tx_revenue = tx_data.groupby('CustomerID').Revenue.sum().reset_index()

    #merge it with our main dataframe
    tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

    #apply clustering
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Revenue']])
    tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])

    #order the cluster numbers
    tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)

    #show details of the dataframe

    #__________________________________________________________________________________________________________________

    #OVERALL SCORE & GROUPING:

    #calculate overall score and use mean() to see details
    tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']

    tx_user['Segment'] = 'Low-Value'
    tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
    tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value'
    
    # bar chart
    fig, ax = plt.subplots()
    sns.set_theme(style='darkgrid', palette='deep', font='Verdana')
    ax.set_title('Customer Clusters Based on RFM', fontsize=12)
    tx_user['Segment'].value_counts().plot(kind='bar', ax=ax)
    st.header('Exploratory Data Analysis', divider='orange')
    # Display the plot
    st.write(fig)


    #Revenue vs Frequency
    

    plot_data = [
        go.Scatter(
            x=tx_user.query("Segment == 'Low-Value'")['Frequency'],
            y=tx_user.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=tx_user.query("Segment == 'Mid-Value'")['Frequency'],
            y=tx_user.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=tx_user.query("Segment == 'High-Value'")['Frequency'],
            y=tx_user.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Revenue"},
            xaxis= {'title': "Frequency"},
            title='Segments'
        )
    fig2 = go.Figure(data=plot_data, layout=plot_layout)
    st.write(fig2)

    #Revenue Recency

    

    plot_data = [
        go.Scatter(
            x=tx_user.query("Segment == 'Low-Value'")['Recency'],
            y=tx_user.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=tx_user.query("Segment == 'Mid-Value'")['Recency'],
            y=tx_user.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=tx_user.query("Segment == 'High-Value'")['Recency'],
            y=tx_user.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Revenue"},
            xaxis= {'title': "Recency"},
            title='Segments'
        )
    fig2 = go.Figure(data=plot_data, layout=plot_layout)
    st.write(fig2)

    # Revenue vs Frequency
    

    plot_data = [
        go.Scatter(
            x=tx_user.query("Segment == 'Low-Value'")['Recency'],
            y=tx_user.query("Segment == 'Low-Value'")['Frequency'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=tx_user.query("Segment == 'Mid-Value'")['Recency'],
            y=tx_user.query("Segment == 'Mid-Value'")['Frequency'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=tx_user.query("Segment == 'High-Value'")['Recency'],
            y=tx_user.query("Segment == 'High-Value'")['Frequency'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Frequency"},
            xaxis= {'title': "Recency"},
            title='Segments'
        )
    fig2 = go.Figure(data=plot_data, layout=plot_layout)
    st.write(fig2)

    st.header('Customer Segments Based On RFM Model', divider='orange')

    st.write(tx_user)
    
    

    
  