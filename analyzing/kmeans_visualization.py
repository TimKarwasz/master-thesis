# general imports
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Dataframes
df = pd.read_csv('df_test.csv') # df_train contains german and hinter_den_dingen, nachrichten;
df_all = pd.read_csv('df.csv', low_memory=False)

df_clean = df_all[
    [
        "episode_duration",
        "podcast_episodeCount",
        "podcast_popularityScore",
        "episode_datePublished",
        "podcast_lastUpdate",
        "Loudness_sma3_middle",
        "mfcc1_sma3_middle",
        "mfcc2_sma3_middle",
        "mfcc3_sma3_middle",
        "mfcc4_sma3_middle",
        "F0semitoneFrom27.5Hz_sma3nz_middle",
        "syllables_audio_count",
        "jitterLocal_sma3nz_middle",
        "shimmerLocaldB_sma3nz_middle",
        "HNRdBACF_sma3nz_middle",
        "spectralFlux_sma3_middle",
        "slope0-500_sma3_middle",
        "slope500-1500_sma3_middle",
        "F1frequency_sma3nz_middle",
        "F2frequency_sma3nz_middle",
        "F3frequency_sma3nz_middle",
        "F1bandwidth_sma3nz_middle",
        "F2bandwidth_sma3nz_middle",
        "F3bandwidth_sma3nz_middle",
        "number_of_pauses",
        "total_pause_time",
        "standardized_pause_rate",
        "mean_pause_duration",
        "n_stop_words",
        "n_tokens",
        "n_unique_tokens",
        "n_sentences",
        "n_characters",
        "token_length_mean",
        "sentence_length_mean",
        "flesch_reading_ease",
        "syllables_text_count",
        "frazier_mean",
        "yngve_mean",
        "grm_checker_count",
        "prob_is_german",
        "smog",
        "gunning_fog",
        "lix",
        "rix",
        "wiener_sachtext_formel",
        "duplicate_ngram_chr_fraction_5",
        "duplicate_ngram_chr_fraction_6",
        "duplicate_ngram_chr_fraction_7",
        "duplicate_ngram_chr_fraction_8",
        "duplicate_ngram_chr_fraction_9",
        "duplicate_ngram_chr_fraction_10",
        "symbol_to_word_ratio_#",
        "pos_prop_ADJ",
        "pos_prop_ADV", 
        "pos_prop_NOUN", 
        "pos_prop_VERB",
        "phrase_amount",
        "uid"              
        
    ]
]




df2 = df_clean.dropna()

## Elbow Method to find the best amount of clusters
# https://www.researchgate.net/profile/Hestry-Humaira/publication/339670247_Determining_The_Appropiate_Cluster_Number_Using_Elbow_Method_for_K-Means_Algorithm/links/6142ce3a7d081355ccef105b/Determining-The-Appropiate-Cluster-Number-Using-Elbow-Method-for-K-Means-Algorithm.pdf
inertias = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df2.drop(["uid"], axis=1))
    inertias.append(kmeans.inertia_)

# Finding the right amount of clusters with elbow method
fig = px.scatter(x = range(1,10), y = inertias, title="Elbow method").update_traces(mode="lines+markers")
fig.update_layout(xaxis={"title": "Number of clusters",},yaxis={"title": "Inertia"},)
fig.show()
#  inertia is the sum of squared distances from each data point to the centroid (the center) of its assigned cluster.
# high inertia > If the inertia value is high, it means that the average distance between the data points 
# and their assigned cluster centroids is large. This suggests that the clusters are not well-defined, 
# and the data points are spread out and far from the centroids, which indicates poor clustering.

## K-means with pc
# drop nan rows
df2 = df_clean.dropna()

to_drop = ["n_tokens", "uid"]

scaler = StandardScaler()
df_std = scaler.fit_transform(df2.drop(to_drop, axis=1))
pca = PCA()
pca.fit(df_std)

#do pca
pca = PCA(n_components=7)
pca.fit(df_std)
scores_pca = pca.transform(df_std)

# do kmeans
kmeans_pca = KMeans(n_clusters=2, random_state=42)
kmeans_pca.fit(scores_pca)


df_pca_kmeans = pd.concat([df2.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_pca_kmeans.columns.values[-7:] = ["Component 1", "Component 2", "Component 3", "Component 4", "Component 5", "Component 6",
                                     "Component 7"]
df_pca_kmeans["K-means PCA Label"] = kmeans_pca.labels_
df_pca_kmeans["Cluster"] = df_pca_kmeans["K-means PCA Label"].map({0:"Erstes", 1:"Zweites", 2:"Drittes", 3:"fourth"})

x_axis = df_pca_kmeans["Component 1"]
y_axis = df_pca_kmeans["Component 2"]

fig = px.scatter(df_pca_kmeans, x= "n_tokens", y= "episode_duration", color="Cluster",
                 color_continuous_scale="turbo", title="Mit PCA")
fig.show()


fig_PCA = px.scatter(df_pca_kmeans, x = "Component 1", y = "Component 2", 
                     title="", color="episode_duration", labels={'n_tokens': 'Wortanzahl'}, hover_name="uid")
fig_PCA.update_layout(xaxis={"title": "Erste Hauptkomponente",},yaxis={"title": "Zweite Hauptkomponente"})
fig_PCA.show()
#fig_PCA.write_image("plots/" + "kmeans_pca" + ".png")

# Result : alles wird in zwei Cluster aufgeteilt, wobei das zweite cluster aus den Datenpunkten besteht die eine sehr 
# lange Dauer haben 


## Just k-means without pca
df = df_clean.dropna()

kmeans = KMeans(n_clusters=2,  random_state=42)
kmeans.fit(df)

# add to df
df_new = pd.concat([df.reset_index(drop=True)], axis=1)
df_new["K-means Label"] = kmeans.labels_
df_new["K-means Labelname"] = df_new["K-means Label"].map({0:"first", 1:"second", 2:"third"})

fig2 = px.scatter(df_new, x= "n_tokens", y= "episode_duration", color="K-means Labelname",
                 color_continuous_scale="turbo", title="Ohne PCA")
fig2.show()
 

# Result : Aufteilung in zwei cluster 