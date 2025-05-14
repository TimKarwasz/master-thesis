# This file is used to create visualizations and analyze data used in the master thesis

# general imports
import numpy as np
import plotly.express as px
import pandas as pd
from langcodes import *
from datetime import datetime
import pytz

# Dataframes
df = pd.read_csv('df_test.csv') # df_train contains german and hinter_den_dingen, nachrichten;
df_all = pd.read_csv('df.csv', low_memory=False)


# A Graph that shows grammatik errors in relation to podcast length
fig = px.scatter(df_all, x="grm_checker_count", y="n_tokens", title="", 
                 #color_discrete_map={"ENG": 'red', "DEU": 'blue'},
                 color_discrete_sequence=px.colors.qualitative.Plotly,
                 color="detected_language", labels={'detected_language': 'Sprache'})
fig.update_layout(xaxis={"title": "Grammatikfehler",},yaxis={"title": "Wortanzahl"},)
#fig.write_image("plots/" + "grm_checker_count_n_tokens" + ".png")
#fig.show()


# A Graph which shows mean_pauzse_duration in relation to podcast_length
fig = px.scatter(df_all, x="mean_pause_duration", y="n_tokens", title="", hover_name="uid", color="phrase_amount"
                 ,labels={'phrase_amount': '„Untertitelung des ZDF“'})
fig.update_layout(xaxis={"title": "Durchschnittliche Pausenzeit in Sekunden",},yaxis={"title": "Wortanzahl"})
#fig.write_image("plots/" + "mean_pause_time_vs_tokens_vs_unter" + ".png")
#fig.show()

# Graph showing syllables_text and syl audio
fig = px.scatter(df_all, x="syllables_audio_count", y="syllables_text_count", title="", 
                  hover_name="uid")
fig.update_layout(xaxis={"title": "Silben in der Audiospur",},yaxis={"title": "Silben im Transkript"},)
#fig.write_image("plots/" + "silben_audio_vs_text_all" + ".png")
#fig.show()

# for testdata
for index, row in df.iterrows():
    for col in df.columns:
        if col == "podcast_title":
            if df.at[index, col] == "Langsam Gesprochene Nachrichten | Audios | DW Deutsch lernen":
                df.at[index, col] = 'Langsam Gesprochene Nachrichten'
fig = px.scatter(df, x="syllables_audio_count", y="syllables_text_count", title="", 
                  hover_name="uid", color="episode_title",labels={'title': ''})
fig.update_layout(xaxis={"title": "Silben in der Audiospur",},yaxis={"title": "Silben im Transkript"},)
#fig.write_image("plots/" + "silben_audio_vs_text" + ".png")
#fig.show()


# A Graph showing the WER of the testdata set
# select the all "Hinter den Dingen" episodes
df_hinter = df.loc[df['podcast_title'] == 'Hinter den Dingen']
fig = px.bar(df_hinter, x = df_hinter["uid"].astype(str), y = "wer",)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_layout(xaxis={'showticklabels': False})
fig.update_layout(xaxis={"title": "",},yaxis={"title": "WER"},)
#fig.write_image("plots/" + "hinter_wer" + ".png")
#fig.show()

# select all "Slow german" episodes
df_slow = df.loc[df['podcast_title'] == 'Slow German']
fig = px.bar(df_slow, x = df_slow["uid"].astype(str), y = "wer",)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_layout(xaxis={'showticklabels': False})
fig.update_layout(xaxis={"title": "",},yaxis={"title": "WER"},)
#fig.write_image("plots/" + "slow_wer" + ".png")
#fig.show()

#  select all "Langsam Gesprochene Nachrichten" episodes
df_news = df.loc[df['podcast_title'] == 'Langsam Gesprochene Nachrichten']
fig = px.bar(df_news, x = df_news["uid"].astype(str), y = "wer",)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_layout(xaxis={'showticklabels': False})
fig.update_layout(xaxis={"title": "",},yaxis={"title": "WER"},)
#fig.write_image("plots/" + "news_wer" + ".png")
#fig.show()

# Graphs showing the wer sorted by release date of the episodes
# for the other graphs switch the name of the df in the next for loop to
# df_news or df_hinter or df_slow
date_and_wer = []
for elem in zip(df_slow["episode_datePublished"], df_slow["wer"], df_slow["uid"]):
    date_and_wer.append((elem[0],elem[1]))

sorted_list =  sorted(date_and_wer, key=lambda x: x[0])
date_and_wer_nicely = []
for elem in sorted_list:
    dt = datetime.fromtimestamp(elem[0], pytz.timezone("Europe/Berlin"))
    date_and_wer_nicely.append((str(dt.strftime("%d.%m.%Y")), elem[1]))
    
dates = list(map(lambda x: x[0], date_and_wer_nicely))
wers = list(map(lambda x: x[1], date_and_wer_nicely))
fig = px.bar(x=dates, y=wers, title="", barmode="group")
fig.update_layout(xaxis={'showticklabels': True})
fig.update_layout(xaxis={"title": "Datum",},yaxis={"title": "WER"})
#fig.write_image("plots/" + "slow_wer_sorted" + ".png")
#fig.show()

# A graph showing the number of deletions, insertions and substitutions errors
# to switch between the three posssible polts switch the dataframe in the next lines around (df_hinter, df_slow, df_news)
df_slow = df_slow.rename(columns={"substitutions": "Substitution-Fehler", "insertions": "Insertion-Fehler", "deletions" : "Deletion-Fehler" })
df_hinter = df_hinter.rename(columns={"substitutions": "Substitution-Fehler", "insertions": "Insertion-Fehler", "deletions" : "Deletion-Fehler" })
df_news = df_news.rename(columns={"substitutions": "Substitution-Fehler", "insertions": "Insertion-Fehler", "deletions" : "Deletion-Fehler" })
fig = px.bar(df_hinter, x=df_hinter["uid"].astype(str), y=["Substitution-Fehler", "Deletion-Fehler", "Insertion-Fehler"], title="", barmode="group")
fig.update_layout(xaxis={'showticklabels': False})
fig.update_layout(xaxis={"title": "",},yaxis={"title": "Anzahl"}, legend_title="Fehlertyp")
#fig.write_image("plots/" + "hinter_fehlerklassen" + ".png")
#fig.show()

# A Graph showing the missing syllables for the testdata in comparison to WER
missing_syl = []
for elem in zip(df["syllables_audio_count"], df["syllables_text_count"]):
    missing_syl.append( 1 - (elem[1]/elem[0]))

df_miss = pd.concat([df.reset_index(drop=True), pd.DataFrame(missing_syl)], axis=1)
df_miss.columns.values[-1:] = ["missing"]
fig = px.scatter(df_miss, x = "missing", y = "wer", color="podcast_title",labels={'podcast_title': ''})
fig.update_layout(xaxis={"title": "Anteil der fehlenden Silben",},yaxis={"title": "WER"})
#fig.write_image("plots/" + "anteil_fehlender_silben" + ".png")
#fig.show()


# count how often the 4grams occur
gram_dist = df_all["four_gram"].value_counts()
gram_tuples = []
for key in gram_dist.keys():
    gram_tuples.append((key, gram_dist[key]))

grams_sorted = sorted(gram_tuples, key=lambda x: x[1])[-6:]
grams = list(map(lambda x: x[0], grams_sorted))
counts = list(map(lambda x: x[1], grams_sorted))
fig = px.bar(x=grams, y=counts, title="", text_auto=True)
fig.update_layout(xaxis={'categoryorder': 'total descending', 'showticklabels': True})
fig.update_layout(xaxis={"title": "4-Gramme",},yaxis={"title": "Anzahl"})
#fig.write_image("plots/" + "4_gramme_anzahl" + ".png")
#fig.show()

# A Graph showing the categories
for index, row in df_all.iterrows():
    for col in df_all.columns:
        if col == "podcast_category1":
            # if the value is NaN
            if isinstance(row[col], float):
                #print(df_all.at[index, col])
                df_all.at[index, col] = 'Nicht gesetzt'

categories_dist = df_all["podcast_category1"].value_counts()
categories_tuples = []
for key in categories_dist.keys():
    categories_tuples.append((key, categories_dist[key]))

# to get the top 5 topics use [-5:] 
categories_sorted = sorted(categories_tuples, key=lambda x: x[1])
categories = list(map(lambda x: x[0], categories_sorted))
counts = list(map(lambda x: x[1], categories_sorted))
fig = px.bar(x=categories, y=counts, title="", text_auto=True)
fig.update_layout(xaxis={'categoryorder': 'total descending', 'showticklabels': True})
fig.update_layout(xaxis={"title": "Kategorie",},yaxis={"title": "Anzahl"})
#fig.write_image("plots/" + "categorie_all" + ".png")
#fig.show()


# A Graph showing the authors
for index, row in df_all.iterrows():
    for col in df_all.columns:
        if col == "podcast_itunesAuthor":
            # if the value is NaN
            if isinstance(row[col], float):
                #print(df_all.at[index, col])
                df_all.at[index, col] = 'nicht angegeben'

authors_dist = df_all["podcast_itunesAuthor"].value_counts()
author_tuples = []
for key in authors_dist.keys():
    author_tuples.append((key, authors_dist[key]))

# to get the top 5 topics use [-5:] 
authors_sorted = sorted(author_tuples, key=lambda x: x[1])[-10:] 
authors = list(map(lambda x: x[0], authors_sorted))
counts = list(map(lambda x: x[1], authors_sorted))
fig = px.bar(x=authors, y=counts, title="", text_auto=True)
fig.update_layout(xaxis={'categoryorder': 'total descending', 'showticklabels': True})
fig.update_layout(xaxis={"title": "Author",},yaxis={"title": "Anzahl"})
#fig.write_image("plots/" + "author_all_top10" + ".png")
#fig.show()


# Graph showing the languages in the dataset
language_dict = {}
for elem in df_all["detected_language"]:
    if elem in language_dict.keys():
        language_dict[elem] +=1
    else:
        language_dict[elem] = 1

languages = []
counts = []
for key in language_dict.keys():
    if isinstance(key,str):
        languages.append(Language.get(key).display_name('de'))
        counts.append(language_dict[key])

fig = px.bar(x=languages, y=counts, title="", text_auto=True)
fig.update_layout(xaxis={'categoryorder': 'total descending', 'showticklabels': True})
fig.update_layout(xaxis={"title": "Sprache",},yaxis={"title": "Anzahl"})
#fig.write_image("plots/" + "languages" + ".png")
#fig.show()

# A Graph visualizing the percentages that the podcast is german
fig = px.histogram(df_all, x = "prob_is_german",)
#fig.write_image("plots/" + "prob_is_german" + ".png")
#fig.show()


# A Graph showing the relation of Loudness to WER

# for the testdata
fig = px.scatter(df, x = "Loudness_sma3_middle", y = "wer")
fig.update_layout(xaxis={"title": "Loudness_sma3_middle",},yaxis={"title": "WER"})
#fig.write_image("plots/" + "loudness_wer" + ".png")
#fig.show()

# and for the normal data
fig = px.histogram(df_all, x="Loudness_sma3_middle", nbins=30,
                   opacity=0.8, text_auto=True)
fig.update_layout(xaxis={"title": "Loudness_sma3_middle",},yaxis={"title": "Anzahl"})
fig.update_layout(bargap=0.1) 
#fig.write_image("plots/" + "loudness_hist" + ".png")
#fig.show()


# A graph showing the f1 frequency 
fig = px.histogram(df_all, x="F1frequency_sma3nz_middle", nbins=30,
                   opacity=0.8, text_auto=True)
fig.update_layout(xaxis={"title": "F1frequency_sma3nz_middle",},yaxis={"title": "Anzahl"})
fig.update_layout(bargap=0.1) 
#fig.write_image("plots/" + "freq_hist" + ".png")
#fig.show()

fig = px.scatter(df_all, x = "pos_prop_VERB", y = "n_tokens", color="phrase_amount",labels={'phrase_amount': '„Untertitelung des ZDF“'})
fig.update_layout(xaxis={"title": "Anteil des Part-of-Speech Tags VERB",},yaxis={"title": "Wortanzahl"})
#fig.write_image("plots/" + "pos_verb" + ".png")
fig.show()