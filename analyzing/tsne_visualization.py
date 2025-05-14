# general imports
import time
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
        "podcast_category1"
    ]
]

## t-SNE 2d
df = df_clean.dropna()
to_drop = ["podcast_category1", "n_tokens"]

tsne = TSNE(n_components=2, random_state=42)
start_time = time.time()
X_tsne = tsne.fit_transform(df.drop(to_drop, axis=1))
print(f"Finished t-SNE after {time.time() - start_time} sec ")

df_tsne = pd.concat([df.reset_index(drop=True)], axis=1)
df_tsne["x_tsne"] = X_tsne[:, 0]
df_tsne["y_tsne"] = X_tsne[:, 1]
fig = px.scatter(df_tsne,x="x_tsne", y="y_tsne", color=to_drop[1], labels={'n_tokens': 'Wortanzahl',
                                                                           'duration': 'Podcastlänge'})
title = "t-SNE visualization colored by " + str(to_drop[1])
fig.update_layout(
    #title=title,
    xaxis_title="",
    yaxis_title="",
)
fig.update_layout(xaxis={'showticklabels': False})
fig.update_layout(yaxis={'showticklabels': False})
#fig.write_image("plots/" + "tsne_tokens" + ".png")
fig.show()

