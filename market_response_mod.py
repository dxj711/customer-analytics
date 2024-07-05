from datetime import datetime, timedelta,date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import sklearn
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import streamlit as st


st.title("Market Response Modelling")
uploaded_file = st.file_uploader("Upload Data File",type=["xlsx","csv"])
if uploaded_file:
    df_data = pd.read_excel(uploaded_file)
    def order_cluster(cluster_field_name, target_field_name,df,ascending):
        new_cluster_field_name = 'new_' + cluster_field_name
        df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
        df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
        df_new['index'] = df_new.index
        df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
        df_final = df_final.drop([cluster_field_name],axis=1)
        df_final = df_final.rename(columns={"index":cluster_field_name})
        return df_final

    #df_data = pd.read_excel("C:/Users/PhilBiju(G10XIND)/Downloads/customer_analytics/datasets/offers.xlsx")

    def calc_uplift(df):
        #assigning 25$ to the average order value
        avg_order_value = 25
        
        #calculate conversions for each offer type
        base_conv = df[df.offer == 'No Offer']['conversion'].mean()
        disc_conv = df[df.offer == 'Discount']['conversion'].mean()
        bogo_conv = df[df.offer == 'Buy One Get One']['conversion'].mean()
        
        #calculate conversion uplift for discount and bogo
        disc_conv_uplift = disc_conv - base_conv
        bogo_conv_uplift = bogo_conv - base_conv
        
        #calculate order uplift
        disc_order_uplift = disc_conv_uplift * len(df[df.offer == 'Discount']['conversion'])
        bogo_order_uplift = bogo_conv_uplift * len(df[df.offer == 'Buy One Get One']['conversion'])
        
        #calculate revenue uplift
        disc_rev_uplift = disc_order_uplift * avg_order_value
        bogo_rev_uplift = bogo_order_uplift * avg_order_value
        
        
        st.write('Discount Conversion Uplift: {0}%'.format(np.round(disc_conv_uplift*100,2)))
        st.write('Discount Order Uplift: {0}'.format(np.round(disc_order_uplift,2)))
        st.write('Discount Revenue Uplift: ${0}\n'.format(np.round(disc_rev_uplift,2)))
            
        st.write('-------------- \n')

        st.write('BOGO Conversion Uplift: {0}%'.format(np.round(bogo_conv_uplift*100,2)))
        st.write('BOGO Order Uplift: {0}'.format(np.round(bogo_order_uplift,2)))
        st.write('BOGO Revenue Uplift: ${0}'.format(np.round(bogo_rev_uplift,2)))     

    st.header('Exploratory Data Analysis', divider='orange')

    df_plot = df_data.groupby('recency').conversion.mean().reset_index()
    plot_data = [
        go.Bar(
            x=df_plot['recency'],
            y=df_plot['conversion'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={'title': "Recency"},
            yaxis={'title': "Conversion"},
            title='Recency vs Conversion'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.write(fig)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df_data[['history']])
    df_data['history_cluster'] = kmeans.predict(df_data[['history']])
    df_data = order_cluster('history_cluster', 'history',df_data,True)

    df_plot = df_data.groupby('history_cluster').conversion.mean().reset_index()
    plot_data = [
        go.Bar(
            x=df_plot['history_cluster'],
            y=df_plot['conversion'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={'title': "History Cluster"},
            yaxis={'title': "Conversion"},
            title='History vs Conversion'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.write(fig)

    df_plot = df_data.groupby('zip_code').conversion.mean().reset_index()
    plot_data = [
        go.Bar(
            x=df_plot['zip_code'],
            y=df_plot['conversion'],
            marker=dict(
            color=['green', 'blue', 'orange'])
        )
    ]

    plot_layout = go.Layout(
            xaxis={'title': "Zip-Code"},
            yaxis={'title': "Conversion"},
            title='Zip Code vs Conversion'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.write(fig)

    df_plot = df_data.groupby('is_referral').conversion.mean().reset_index()
    plot_data = [
        go.Bar(
            x=df_plot['is_referral'],
            y=df_plot['conversion'],
            marker=dict(
            color=['green', 'blue', 'orange'])
        )
    ]

    plot_layout = go.Layout(
            xaxis={'title': "Is Referral"},
            yaxis={'title': "Conversion"},
            title='Referrals Conversion'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.write(fig)

    df_plot = df_data.groupby('channel').conversion.mean().reset_index()
    plot_data = [
        go.Bar(
            x=df_plot['channel'],
            y=df_plot['conversion'],
            marker=dict(
            color=['green', 'blue', 'orange'])
        )
    ]

    plot_layout = go.Layout(
            xaxis={'title': "Channel"},
            yaxis={'title': "Conversion"},
            title='Channel vs Conversion'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.write(fig)

    df_plot = df_data.groupby('offer').conversion.mean().reset_index()
    plot_data = [
        go.Bar(
            x=df_plot['offer'],
            y=df_plot['conversion'],
            marker=dict(
            color=['green', 'blue', 'orange'])
        )
    ]

    plot_layout = go.Layout(
            xaxis={'title': "Offer"},
            yaxis={'title': "Conversion"},
            title='Offer vs Conversion'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.write(fig)

    st.header('Uplift For Uploaded Data', divider='orange')
    calc_uplift(df_data)




    df_model = df_data.copy()
    df_model = pd.get_dummies(df_model)


    #create feature set and labels
    X = df_model.drop(['conversion'],axis=1)
    y = df_model.conversion

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
    xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
    X_test['proba'] = xgb_model.predict_proba(X_test)[:,1]
    st.header('Conversion Probability', divider='orange')
    st.write(X_test)
    X_test['conversion'] = y_test


    real_disc_uptick = int(len(X_test)*(X_test[X_test['offer_Discount'] == 1].conversion.mean() - X_test[X_test['offer_No Offer'] == 1].conversion.mean()))
    pred_disc_uptick = int(len(X_test)*(X_test[X_test['offer_Discount'] == 1].proba.mean() - X_test[X_test['offer_No Offer'] == 1].proba.mean()))

    st.header('Real vs. Predicted: Evaluation', divider='orange')
    st.write('Real Discount Uptick - Order: {}, Revenue: {}'.format(real_disc_uptick, real_disc_uptick*25))
    st.write('Predicted Discount Uptick - Order: {}, Revenue: {}'.format(pred_disc_uptick, pred_disc_uptick*25))

    real_bogo_uptick = int(len(X_test)*(X_test[X_test['offer_Buy One Get One'] == 1].conversion.mean() - X_test[X_test['offer_No Offer'] == 1].conversion.mean()))
    pred_bogo_uptick = int(len(X_test)*(X_test[X_test['offer_Buy One Get One'] == 1].proba.mean() - X_test[X_test['offer_No Offer'] == 1].proba.mean()))


    st.write('Real Discount Uptick - Order: {}, Revenue: {}'.format(real_bogo_uptick, real_bogo_uptick*25))
    st.write('Predicted Discount Uptick - Order: {}, Revenue: {}'.format(pred_bogo_uptick, pred_bogo_uptick*25))