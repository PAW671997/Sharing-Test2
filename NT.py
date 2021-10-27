# Imports
# -----------------------------------------------------------
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
# -----------------------------------------------------------

# Helper functions
# -----------------------------------------------------------
# Load data from external source
df = pd.read_csv(
    "https://raw.githubusercontent.com/PAW671997/Sharing-Test/main/Agustus%20Repo.csv?token=APDGMSGHFVM6Z2S6WKCMNSDBPEGMK"
)


def run_kmeans(df, n_clusters=2):
    kmeans = KMeans(n_clusters, random_state=0).fit(df[["NT", "OT"]])

    fig, ax = plt.subplots(figsize=(16, 9))

    #Create scatterplot
    ax = sns.scatterplot(
        ax=ax,
        x=df.NT,
        y=df.OT,
        hue=kmeans.labels_,
        palette=sns.color_palette("colorblind", n_colors=n_clusters),
        legend=None,
    )

    return fig
# -----------------------------------------------------------


# Sidebar
# -----------------------------------------------------------
sidebar = st.sidebar
df_display = sidebar.checkbox("Display Raw Data", value=True)

n_clusters = sidebar.slider(
    "Select number of Clusters",
    min_value=2,
    max_value=10
)
# -----------------------------------------------------------


# Main
# -----------------------------------------------------------
# Create a title for your app
st.title("Interactive K-Means Clustering - Timesheet NT OT")

# A description
st.write("Here is the dataset used in this analysis:")

# Display the dataframe


if df_display:
    st.write(df)
# -----------------------------------------------------------


# Show cluster scatter plot
st.write(run_kmeans(df, n_clusters=n_clusters))
# -----------------------------------------------------------
