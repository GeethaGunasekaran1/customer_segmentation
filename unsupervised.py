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

# Ignore warnings
warnings.filterwarnings('ignore') 

# Read the data from the local drive
retail_df = pd.read_csv('retail_df.csv')

# Create the title for the application
st.title("Online Retail Store Clustering")

# Create buttons to show database and descriptive statistics
df_button = st.sidebar.button("Show Dataframe")
des_button = st.sidebar.button("Descriptive Statistics")

# Display dataframe and descriptive statistics when buttons are pressed
if df_button:    
    st.write("Retail Dataframe")
    st.dataframe(retail_df)
if des_button:    
    st.write("Retail Dataframe - Descriptive Statistics")
    st.table(pd.DataFrame(retail_df.describe()))

# Feature column selection for deploying K-means algorithm
options = st.sidebar.multiselect("Select the feature columns", ['Amount', 'Frequency', 'Recency'])

# Choose the method to find K-value
st.sidebar.write("Choose the method to find K-value")
ch1 = st.sidebar.checkbox("Elbow Method")
ch2 = st.sidebar.checkbox("Silhouette Method")
num = int(st.sidebar.selectbox("Select the K-value", range(2, 8)))

deploy_button = st.sidebar.button("Deploy K-Means Algorithm")
y_axis = st.sidebar.selectbox("Select the column to plot", ['Amount', 'Frequency', 'Recency'])
deployplot_button = st.sidebar.button("Plot the Clusters")

if deployplot_button:
    st.title("Cluster ID vs " + y_axis)
    scatter_plot_variation = alt.Chart(retail_df).mark_boxplot().encode(
        x=alt.X('Cluster_Id:O', title='Cluster ID'),
        y=alt.Y(y_axis, title=y_axis),
        color=alt.Color('Cluster_Id:N', legend=None)
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(scatter_plot_variation)

try:
    if options:
        # Select the columns and scale the values
        retail_df_selected = retail_df[options]
        scaler = StandardScaler()
        retail_df_scaled = scaler.fit_transform(retail_df_selected)
        retail_df_scaled = pd.DataFrame(retail_df_scaled, columns=options)

        # Create widget to select Elbow or Silhouette method to find k-value
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
        elbow_ssd = []
        sil_scores = []

        # Deploying the KMeans model with different k values
        for num_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
            kmeans.fit(retail_df_scaled)
            elbow_ssd.append(kmeans.inertia_)
            cluster_labels = kmeans.labels_
            sil_scores.append(silhouette_score(retail_df_scaled, cluster_labels))

        # Create a DataFrame with results
        sil_data = pd.DataFrame({'Clusters': range_n_clusters, 'Silhouette Scores': sil_scores, "Elbow SSD Value": elbow_ssd})

        # Elbow method
        if ch1:
            st.dataframe(sil_data)
            plt.figure(figsize=(17, 8))
            plt.title("Elbow Method")
            st.line_chart(elbow_ssd)
            plt.show()

        # Silhouette method
        if ch2:
            st.dataframe(sil_data)
            plt.figure(figsize=(17, 8))
            plt.title("Silhouette Scores")
            st.line_chart(sil_scores)
            plt.xlabel("K")
            plt.ylabel("Silhouette Score")
            plt.grid()
            plt.xticks(range(2, 11))
            plt.show()

        # Deploy K-Means algorithm
        if deploy_button:
            kmeans = KMeans(n_clusters=num, max_iter=50)
            kmeans.fit(retail_df_scaled)
            retail_df['Cluster_Id'] = kmeans.labels_

    else:
        st.write("Select the columns to deploy the model")

except Exception as e:
    st.write("An error occurred: ", e)
