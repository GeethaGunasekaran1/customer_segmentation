import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
from sklearn.preprocessing import StandardScaler
import warnings 
import altair as alt
  
# Settings the warnings to be ignored 
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings('ignore') 

#read the data from the local drive
retail_df=pd.read_csv('retail_df.csv')
retail_df_df_scaled=pd.DataFrame()

# create the title for the Application
st.title("Online Retail store Clustering")

#create button to show database and descriptive statistics
df_button=st.sidebar.button("Show Dataframe")
des_button=st.sidebar.button("Descriptive statistics")

#operation should be performed when dataframe and descriptive statistics button is pressed
if df_button:    
    st.write("Retail Dataframe")
    st.dataframe(retail_df)
if des_button:    
    st.write("Retail Dataframe- Descriptive statistics")
    st.table(pd.DataFrame(retail_df.describe()))

# Feature column selection for deploying k-means algorithm
options = st.sidebar.multiselect("Select the feature columns",['Amount', 'Frequency', 'Recency'])


st.sidebar.write("Choose the method to find K-value ")
ch1=st.sidebar.checkbox("Elbow Method")
ch2=st.sidebar.checkbox("Silhoutte Method")
num=int(st.sidebar.selectbox("select the K-value",range(2,8)))


deploy_button=st.sidebar.button("Deploy k-Means algorithm")
y_axis=st.sidebar.selectbox("select the column to plot",['Amount', 'Frequency', 'Recency'])
deployplot_button=st.sidebar.button("plot the clusters")

if deployplot_button:
    st.title("Cluster ID Vs " + y_axis)
    scatter_plot_variation = alt.Chart(retail_df).mark_boxplot().encode(
    x=alt.X('Cluster_Id:O', title='Cluster ID'),
    y=alt.Y(y_axis, title=y_axis),
    color=alt.Color('Cluster_Id:N', legend=None)).properties(
        width=600,
        height=400
        )
    st.altair_chart(scatter_plot_variation)


try:
    #y_axis=st.sidebar.selectbox("select the column to check the cluster",['Amount', 'Frequency', 'Recency'])
    retail_df = retail_df[options]
    # scale the value
    scaler = StandardScaler()
        # fit_transform
    retail_df_df_scaled = scaler.fit_transform(retail_df)
    retail_df_df_scaled = pd.DataFrame(retail_df_df_scaled)
    retail_df_df_scaled.columns = list(options)

    #create widget to select Elbow or silhoutte method to find k-value
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    elbow_ssd = []
    sil_scores = []
    #deploying the kmeans model with different k value 
    for num_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
            kmeans.fit(retail_df_df_scaled)
            elbow_ssd.append(kmeans.inertia_)
            cluster_labels = kmeans.labels_
            # silhouette score
            sil_scores.append(silhouette_score(retail_df_df_scaled, cluster_labels))

        # Create a DataFrame with two columns.
        # First column must contain K values from 2 to 10 and second column must contain Silhouette values obtained after the for loop.
    sil_data = pd.DataFrame({'Clusters': range_n_clusters, 'Silhouette Scores': sil_scores, "elbow_ssd value":elbow_ssd})
    
   
    # elbow method
    if ch1:
        
        st.dataframe(sil_data)
        # plot the SSDs for each n_clusters
        plt.figure(figsize = (17, 8))
        plt.title("Elbow Method")
        st.line_chart(elbow_ssd)
        plt.show()
       
    
    if ch2:
        
        st.dataframe(sil_data)
        plt.figure(figsize = (17, 8))
        plt.title("Silhouette Scores")
        st.line_chart(sil_scores)
        plt.xlabel("K")
        plt.ylabel("Silhouette Score")
        plt.grid()
        plt.xticks(range(2,11))
        plt.show()
        
    
    if deploy_button:        
        
        kmeans = KMeans(n_clusters=num, max_iter=50)
        kmeans.fit(retail_df_df_scaled)
        kmeans.labels_
        # assign the label
        retail_df['Cluster_Id'] = kmeans.labels_



    
    
except:
    if retail_df_df_scaled.columns.empty:
        st.write("select the column to deploy the model")
        
    else:
        st.write("")
