from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import streamlit as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report,confusion_matrix

uploaded_file = st.file_uploader("Upload Data File",type=["xlsx","csv"])


#select target variable
if uploaded_file:
    tx_data = pd.read_csv(uploaded_file,encoding="latin-1")
    target_var = st.sidebar.selectbox("Select Target Variable", tx_data.columns)

    if st.button("Exlporatory Data Analysis"):
        
        # identifying categorical and numerical columns
        num_cols = []
        alph_cols = []

        for col in tx_data.columns:
            if col != target_var:
                if tx_data[col].dtype == float:
                    num_cols.append(col)
            
        for col in tx_data.columns:
            if col != target_var:      
                try:
                    pd.to_numeric(tx_data[col])
                    num_cols.append(col)
                except:
                    alph_cols.append(col)

        cat_vars =[]
        num_vars =[]

        for col in tx_data.columns:
            if col in num_cols:
                if tx_data[col].nunique() > 5:
                    num_vars.append(col)
                else:
                    cat_vars.append(col)
        
        for vars in alph_cols:
            cat_vars.append(vars)

        #target variable conversion:
        tx_data.loc[tx_data.Churn=='No','Churn'] = 0 
        tx_data.loc[tx_data.Churn=='Yes','Churn'] = 1

        print("Categoical variables = ",cat_vars)
        print("Numerical variables = ",num_vars)

        
    # EDA graphs
        for col in tx_data:
            if col in cat_vars:
                if tx_data[col].nunique() < 6:
                    churn_rate = tx_data.groupby(col)['Churn'].mean()
                    plot_data = [
                        go.Bar(
                            x=churn_rate.index,
                            y=churn_rate.values,
                            width = [0.5, 0.5],
                            marker=dict(
                            color=['orange','grey','purple','blue'])
                        )
                    ]
                    plot_layout = go.Layout(
                            xaxis={"type": "category"},
                            yaxis={"title": "Churn Rate"},
                            title= col
                        )
                    fig = go.Figure(data=plot_data, layout=plot_layout)
                    st.write(fig)
            elif col in num_vars:
                churn_rate = tx_data.groupby(col)['Churn'].mean()
                plot_data = [
                    go.Scatter(
                        x=churn_rate.index,
                        y=churn_rate.values,
                        mode='markers',
                        name='Low',
                        marker= dict(size= 7,
                            line= dict(width=1),
                            color= 'orange',
                            opacity= 0.8
                        ),
                    )
                ]
                plot_layout = go.Layout(
                        yaxis= {'title': "Churn Rate"},
                        xaxis= {'title': col},
                        title=f'{col} based Churn rate'
                    )
                fig = go.Figure(data=plot_data, layout=plot_layout)
                st.write(fig)

    if st.button("Calculate Churn Probability For Existing Customers"):   
        
        def order_cluster(cluster_field_name, target_field_name,df,ascending):
            new_cluster_field_name = 'new_' + cluster_field_name
            df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
            df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
            df_new['index'] = df_new.index
            df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
            df_final = df_final.drop([cluster_field_name],axis=1)
            df_final = df_final.rename(columns={"index":cluster_field_name})
            return df_final
        
        df_data = tx_data
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(df_data[['tenure']])
        df_data['TenureCluster'] = kmeans.predict(df_data[['tenure']])
        df_data = order_cluster('TenureCluster', 'tenure',df_data,True)
        df_data['TenureCluster'] = df_data["TenureCluster"].replace({0:'Low',1:'Mid',2:'High'})
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(df_data[['MonthlyCharges']])
        df_data['MonthlyChargeCluster'] = kmeans.predict(df_data[['MonthlyCharges']])
        df_data['MonthlyChargeCluster'] = df_data["MonthlyChargeCluster"].replace({0:'Low',1:'Mid',2:'High'})
        df_data.loc[pd.to_numeric(df_data['TotalCharges'], errors='coerce').isnull(),'TotalCharges'] = np.nan
        df_data = df_data.dropna()
        df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce')
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(df_data[['TotalCharges']])
        df_data['TotalChargeCluster'] = kmeans.predict(df_data[['TotalCharges']])
        df_data = order_cluster('TotalChargeCluster', 'TotalCharges',df_data,True)
        df_data['TotalChargeCluster'] = df_data["TotalChargeCluster"].replace({0:'Low',1:'Mid',2:'High'})
        
        #import Label Encoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        dummy_columns = [] #array for multiple value columns

        for column in df_data.columns:
            if df_data[column].dtype == object and column != 'customerID':
                if df_data[column].nunique() == 2:
                    #apply Label Encoder for binary ones
                    df_data[column] = le.fit_transform(df_data[column]) 
                else:
                    dummy_columns.append(column)

        #apply get dummies for selected columns
        df_data = pd.get_dummies(data = df_data,columns = dummy_columns)
        all_columns = []
        for column in df_data.columns:
            column = column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")
            all_columns.append(column)

        df_data.columns = all_columns
        glm_columns = 'gender'

        for column in df_data.columns:
            if column not in ['Churn','customerID','gender']:
                glm_columns = glm_columns + ' + ' + column
                glm_model = smf.glm(formula='Churn ~ {}'.format(glm_columns), data=df_data, family=sm.families.Binomial())
                res = glm_model.fit()
                
                
        #create feature set and labels
        X = df_data.drop(['Churn','customerID'],axis=1)
        y = df_data.Churn
        #train and test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)
        #building the model
        xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)

        st.write('Accuracy of XGB classifier on training set: {:.2f}'
        .format(xgb_model.score(X_train, y_train)))
        st.write('Accuracy of XGB classifier on test set: {:.2f}'
        .format(xgb_model.score(X_test[X_train.columns], y_test)))
        y_pred = xgb_model.predict(X_test)
        
        from xgboost import plot_tree
        ##set up the parameters
        fig, ax = plt.subplots(figsize=(100,100))
        plot_tree(xgb_model, ax=ax)
        from xgboost import plot_importance
        fig, ax = plt.subplots(figsize=(10,8))
        plot_importance(xgb_model, ax=ax)
        st.write("Feature Importance",fig)
        df_data['Churn Probability'] = xgb_model.predict_proba(df_data[X_train.columns])[:,1]
        st.write("Customer Churn Probability", df_data[['customerID', 'Churn Probability']])


    if st.button("Predict Churn For New Customer"):   
        
        def order_cluster(cluster_field_name, target_field_name,df,ascending):
            new_cluster_field_name = 'new_' + cluster_field_name
            df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
            df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
            df_new['index'] = df_new.index
            df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
            df_final = df_final.drop([cluster_field_name],axis=1)
            df_final = df_final.rename(columns={"index":cluster_field_name})
            return df_final
        
        df_data = tx_data
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(df_data[['tenure']])
        df_data['TenureCluster'] = kmeans.predict(df_data[['tenure']])
        df_data = order_cluster('TenureCluster', 'tenure',df_data,True)
        df_data['TenureCluster'] = df_data["TenureCluster"].replace({0:'Low',1:'Mid',2:'High'})
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(df_data[['MonthlyCharges']])
        df_data['MonthlyChargeCluster'] = kmeans.predict(df_data[['MonthlyCharges']])
        df_data['MonthlyChargeCluster'] = df_data["MonthlyChargeCluster"].replace({0:'Low',1:'Mid',2:'High'})
        df_data.loc[pd.to_numeric(df_data['TotalCharges'], errors='coerce').isnull(),'TotalCharges'] = np.nan
        df_data = df_data.dropna()
        df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce')
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(df_data[['TotalCharges']])
        df_data['TotalChargeCluster'] = kmeans.predict(df_data[['TotalCharges']])
        df_data = order_cluster('TotalChargeCluster', 'TotalCharges',df_data,True)
        df_data['TotalChargeCluster'] = df_data["TotalChargeCluster"].replace({0:'Low',1:'Mid',2:'High'})
        
        #import Label Encoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        dummy_columns = [] #array for multiple value columns

        for column in df_data.columns:
            if df_data[column].dtype == object and column != 'customerID':
                if df_data[column].nunique() == 2:
                    #apply Label Encoder for binary ones
                    df_data[column] = le.fit_transform(df_data[column]) 
                else:
                    dummy_columns.append(column)

        #apply get dummies for selected columns
        df_data = pd.get_dummies(data = df_data,columns = dummy_columns)
        all_columns = []
        for column in df_data.columns:
            column = column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")
            all_columns.append(column)

        df_data.columns = all_columns
        glm_columns = 'gender'

        for column in df_data.columns:
            if column not in ['Churn','customerID','gender']:
                glm_columns = glm_columns + ' + ' + column
                glm_model = smf.glm(formula='Churn ~ {}'.format(glm_columns), data=df_data, family=sm.families.Binomial())
                res = glm_model.fit()
                
            
        #create feature set and labels
        X = df_data.drop(['Churn','customerID'],axis=1)
        y = df_data.Churn
        #train and test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)
        #building the model
        xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)

        st.write('Accuracy of XGB classifier on training set: {:.2f}'
        .format(xgb_model.score(X_train, y_train)))
        st.write('Accuracy of XGB classifier on test set: {:.2f}'
        .format(xgb_model.score(X_test[X_train.columns], y_test)))
        y_pred = xgb_model.predict(X_test)
        
        #test_output = pd.read_csv("C:/Users/PhilBiju(G10XIND)/Downloads/customer_analytics/test_output.csv",encoding="latin-1")
        #predictions = xgb_model.predict(X_test)
        st.write("Predicted class label: ", y_pred)
            
    

  
    
    
    


