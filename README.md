# Online Retail Store Clustering

This project implements a web application using Streamlit for clustering analysis of an online retail dataset. The application allows users to explore the dataset, perform clustering analysis using the k-Means algorithm, and visualize the clusters.

## Introduction

This project aims to provide an interactive tool for clustering analysis of an online retail dataset. It leverages Streamlit, a Python library for building web applications, to create an intuitive user interface for data exploration, clustering, and visualization.

## Features

- Data loading: Load the online retail dataset from a CSV file.
- Descriptive statistics: Display descriptive statistics of the dataset.
- Feature selection: Select feature columns for clustering analysis.
- K-value selection: Choose the number of clusters (K) using the Elbow Method or Silhouette Method.
- Clustering analysis: Perform clustering using the k-Means algorithm.
- Cluster visualization: Visualize the clusters using scatter plots with Altair.

## Installation

To run the application locally, follow these steps:

1. Clone this repository:

   
   git clone https://github.com/GeethaGunasekaran1/customer_segmentation.git
   
3. Navigate to the project directory:

   cd online-retail-store-clustering
   

4. Install the required Python dependencies:

  
   pip install -r requirements.txt
   

5. Run the Streamlit application:

   streamlit run app.py
   

## Usage

- Open the application in your web browser.
- Use the sidebar widgets to explore the dataset, select feature columns, choose the method for finding the K-value, and deploy the k-Means algorithm.
- Click on the "Show Dataframe" button to display the dataset.
- Click on the "Descriptive statistics" button to view descriptive statistics of the dataset.
- Select feature columns and choose the method for finding the K-value.
- Click on the "Deploy k-Means algorithm" button to perform clustering analysis.
- Select a column to plot against the cluster IDs and click on the "Plot the clusters" button to visualize the clusters.



