import pandas as pd
import numpy as np
import plotly.express as px
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X has feature names, but ExtraTreeRegressor was fitted without feature names")

# Modelling
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold

# This python script was used to create and train the extra trees models


# Import and prepare data
# df is used for training the model with the labeled data (transcripts with WER)
# df_test ist the larger dataset used for testing the model
df = pd.read_csv('df_test.csv', low_memory=False) # df_train contains german and hinter_den_dingen, nachrichten; df_testMITWDR contains WDR as well
df_test = pd.read_csv('df.csv', low_memory=False)

##########################################################################################################################################

# 1. The used features

metadata_features = [
    
        "episode_duration",
        "podcast_episodeCount",
        "podcast_popularityScore",
        "episode_datePublished",
        "podcast_lastUpdate"
    ]


audio_features = [
    
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
        "F3bandwidth_sma3nz_middle"
    ]


transcription_features = [
    
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
        "phrase_amount"
    ]

##########################################################################################################################################

# 2. Preprocessing

# do one-hot-encoding for category and detected_language
# note: we have to pass the one-hot-encoding column names from the test data first
# otherwise the model will throw an error when predicting, that it has not seen these rows

original_columns = df_test.columns.tolist()
df_test = pd.get_dummies(df_test, columns=["podcast_category1", "detected_language"])
df = pd.get_dummies(df, columns=["podcast_category1", "detected_language"])
new_columns = [col for col in df_test.columns if col not in original_columns]

columns_in_test = df_test.columns.tolist()
columns_in_train = df.columns.tolist()

for col in columns_in_test:
    if col not in columns_in_train:
        df[col] = 0  # oder np.nan

# then add the new columns to the metadata and the transcription features
for col in new_columns:
    if "podcast" in col:
        metadata_features.append(col)
    else:
        transcription_features.append(col)
    
complete_features = audio_features + metadata_features + transcription_features
used_feature_set = complete_features

colors = []
for elem in used_feature_set:
    if elem in audio_features:
        colors.append("Audio")
    elif elem in metadata_features:
        colors.append("Metadaten")
    elif elem in transcription_features:
        colors.append("Transkription")

Z = df_test[used_feature_set]
X = df[used_feature_set]

y = df["wer"]

def print_statistics(y_test,y_pred, model):
    
    print(model)
    results_df = pd.DataFrame()
    results_df['predicted'] = list(y_pred)
    results_df['actual'] = list(y_test)
    results_df['residual'] = results_df['predicted'] - results_df['actual']
    results_df = results_df.sort_values(by='residual').reset_index(drop=True)
    print(results_df.describe())
    
    #for elem in zip(y_test, y_pred):
        #print(f"Actual label: {elem[0]} ; Predicted label: {elem[1]}")
        
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print('Mean Absolute Error:', round(mae, 4))
    print('Mean Squared Error:', round(mse, 4))
    print('R-squared scores:', round(r2, 4))
    
    #fig = px.scatter(x = y_test, y =y_pred, title=model, trendline="ols")
    #fig.update_layout(xaxis_title="Labeled", yaxis_title="Predicted")
    
    
    #fig.show()
    #fig.write_image("plots/" + "Random_forest" + ".png")
    

    print("#" * 30)


# hyperparameter optimization
# uncomment the following code to see the best hyperparameters
""" 
model = ExtraTreesRegressor(random_state=42)
param_grid = {
    'n_estimators': [10, 30, 50, 100, 150],
    'max_depth': [None, 10, 20, 50, 100],
    'min_samples_split': [2, 5, 10, 30, 100],
    'min_samples_leaf': [2, 5, 10, 30, 100]
}
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
cv = KFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model,param_grid=param_grid, scoring=mae_scorer,cv=cv, n_jobs=-1, verbose=1)
grid_search.fit(X, y)
print("Best Kombination:", grid_search.best_params_)
print("Best MAE:", grid_search.best_score_)
 """
# Best Kombination: {'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 30}
    

##########################################################################################################################################

# 3. Training


# with optimized hyperparameters
model_without_pseudo = ExtraTreesRegressor(n_estimators=30, random_state=42, max_depth=10, min_samples_split=2, min_samples_leaf=5)

# without optimized hyperparameters
#model_without_pseudo = ExtraTreesRegressor(random_state=42)


kf = KFold(n_splits=10, shuffle=True, random_state=42)

n = len(X)
all_preds = np.empty(n)
all_true = np.empty(n)

y_true = []
y_predictions = []
feature_importance_list = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model_without_pseudo.fit(X_train, y_train)
    y_pred = model_without_pseudo.predict(X_test)
    feature_importance_list.append(model_without_pseudo.feature_importances_)

    y_true.extend(y_test)
    y_predictions.extend(y_pred)
    
    all_preds[test_index] = y_pred
    all_true[test_index] = y_test
    
# Analyzing the results
mean_importances = np.mean(feature_importance_list, axis=0)
std_importances = np.std(feature_importance_list, axis=0)
#print_statistics(y_true,y_predictions,model="Extra tree")

train_pred_without_pseudo = y_predictions


##########################################################################################################################################

# Do good Pseudo-Labeling (good meaning we take pseudo labels that get really low label)

# finding "zuverlässige" pseudo labels
tree_preds = np.array([tree.predict(Z) for tree in model_without_pseudo.estimators_])

# Mittelwert der 30 Baum Predctions = Pseudo-Label, Standardabweichung = Unsicherheit
pseudo_labels = tree_preds.mean(axis=0)#
std_abw = tree_preds.std(axis=0)

# Schwellenwert für "zuverlässig"
std_threshold = 0.02 
# jede row deren Mittelwert unter 0.02 liegt ist ein zuverlässiges pseudo label
# mask ist eine Liste aus True und False, mit deren Hilfe wir alle rows aus z extrahieren können auf die unser Kriterium
# zutrifft
mask = std_abw < std_threshold

# auflegen der Maske 
pseudo_labels = pseudo_labels[mask]
pseudo_labels_rows = Z[mask]
#print(f"Anzahl zuverlässiger Pseudo-Labels: {len(pseudo_labels)} von {len(Z)}")

# get the remaining unlabeled data with the opposite of the mask
remaining_z = Z[~mask]
#print(len(remaining_z))

# concat these pseudo labels with the 300 existing train data
X_with_pseudo = pd.concat([X, pseudo_labels_rows], axis=0)
y_with_pseudo = pd.concat([y, pd.Series(pseudo_labels)], axis=0)
X_with_pseudo = X_with_pseudo.reset_index(drop=True)
y_with_pseudo = y_with_pseudo.reset_index(drop=True)

# then train again as before 

model_with_pseudo = ExtraTreesRegressor(n_estimators=30, random_state=42, max_depth=10, min_samples_split=2, min_samples_leaf=5)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

y_true = []
y_predictions = []
feature_importance_list_pseudo = []

for train_index, test_index in kf.split(X_with_pseudo):
    X_train, X_test = X_with_pseudo.iloc[train_index], X_with_pseudo.iloc[test_index]
    y_train, y_test = y_with_pseudo[train_index], y_with_pseudo[test_index]
    
    model_with_pseudo.fit(X_train, y_train)
    y_pred = model_with_pseudo.predict(X_test)
    feature_importance_list_pseudo.append(model_with_pseudo.feature_importances_)

    y_true.extend(y_test)
    y_predictions.extend(y_pred)
    
mean_importances_pseudo = np.mean(feature_importance_list_pseudo, axis=0)
std_importances_pseudo = np.std(feature_importance_list_pseudo, axis=0)
#print_statistics(y_true,y_predictions,model="Extra tree")
train_pred_with_pseudo = y_predictions
##########################################################################################################################################

# Do bad Pseudo-Labeling (bad meaning we take datapoints are for example in english > bad quality or have high occurrence of Untertitelung des ZDF)

# the mask finds all row where the detected language is not german
# and the Phrase "Untertitelung des ZDF" has been found more than x times
#bad_mask = (Z["detected_language_DEU"]) == 0 | (Z["phrase_amount"] > 100) | (Z["mean_pause_duration"] > 3.74)
bad_mask = (Z["detected_language_DEU"] == 0) | (Z["phrase_amount"] > 80 )
#bad_mask =  (Z["detected_language_DEU"] == 0) | (Z["mean_pause_duration"] > 3.74)

bad_pseudo_labels = Z[bad_mask]
#print(len(bad_pseudo_labels))
remaining_labels = Z[~bad_mask]

# create a new train and testset
X_with_bad_pseudo = pd.concat([X, bad_pseudo_labels], axis=0)
# to combat overfitting dont assign each label a straight one
y_with_bad_pseudo = pd.concat([y, pd.Series([random.uniform(0.9, 1) for _ in range(len(bad_pseudo_labels))])], axis=0)
X_with_bad_pseudo = X_with_bad_pseudo.reset_index(drop=True)
y_with_bad_pseudo = y_with_bad_pseudo.reset_index(drop=True)

# train again
model_with_bad_pseudo = ExtraTreesRegressor(n_estimators=30, random_state=42, max_depth=10, min_samples_split=2, min_samples_leaf=5)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

y_true = []
y_predictions = []
feature_importance_list_bad_pseudo = []

for train_index, test_index in kf.split(X_with_bad_pseudo):
    X_train, X_test = X_with_bad_pseudo.iloc[train_index], X_with_bad_pseudo.iloc[test_index]
    y_train, y_test = y_with_bad_pseudo[train_index], y_with_bad_pseudo[test_index]
    
    model_with_bad_pseudo.fit(X_train, y_train)
    y_pred = model_with_bad_pseudo.predict(X_test)
    feature_importance_list_bad_pseudo.append(model_with_bad_pseudo.feature_importances_)

    y_true.extend(y_test)
    y_predictions.extend(y_pred)

mean_importances_bad_pseudo = np.mean(feature_importance_list_bad_pseudo, axis=0)
print_statistics(y_true,y_predictions,model="Extra tree")
train_pred_with_bad_pseudo = y_predictions
##########################################################################################################################################

# 4. Predicting unseen transcripts

test_pred_without_pseudo = model_without_pseudo.predict(Z)
test_pred_with_pseudo = model_with_pseudo.predict(remaining_z)
test_pred_with_bad_pseudo = model_with_bad_pseudo.predict(remaining_labels)


##########################################################################################################################################

# 5. Plot plots


# Analyze the feature importance 
df = pd.DataFrame({
    'Feature': used_feature_set,
    # change between mean_importances_pseudo, mean_importances and mean_importances_bad_pseudo  here, depoending if you are evaluating the pseudo labels
    'Importance': mean_importances_bad_pseudo,
    'Color': colors
})

df_top = df.sort_values(by='Importance', ascending=False).head(15)
fig = px.bar(
    x=df_top['Feature'],
    y=df_top['Importance'],
    color=df_top['Color'],
    labels={'color': 'Featuretyp'},
    #title="Top 10 Feature Importances im Modell mit Pseudo"
    color_discrete_map={"Audio": '#d62728', "Transkription": '#1f77b4', 'Metadaten': '#2ca02c'}
)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_layout(xaxis={"title": "",},yaxis={"title": ""},)
#fig.show()
#fig.write_image("plots/" + "bad_pseudo_feature_importance" + ".png")


# create dataframes again that contain the predictions for the unlabeled data so we can inspect the transcripts
# Note : to switch between the model with or without hyperparameter optimization it is still needed to uncomment the line in 2.
# as they both depend on the same model

df_without_pseudo = pd.concat([df_test.reset_index(drop=True), pd.DataFrame(test_pred_without_pseudo)], axis=1)
df_without_pseudo.columns.values[-1:] = ["pred"]

df_original = df_test[~mask]
df_good_pseudo = pd.concat([df_original.reset_index(drop=True), pd.DataFrame(test_pred_with_pseudo)], axis=1)
df_good_pseudo.columns.values[-1:] = ["pred"]

df_original = df_test[~bad_mask]
df_bad_pseudo = pd.concat([df_original.reset_index(drop=True), pd.DataFrame(test_pred_with_bad_pseudo)], axis=1)
df_bad_pseudo.columns.values[-1:] = ["pred"]


# without pseudo labeling, to switch between hyperparameter optimization uncomment line 231 or 234
fig = px.histogram(df_without_pseudo,x="pred", nbins=30,
                   text_auto=True,
                   opacity=0.7)
fig.update_layout(bargap=0.1)
fig.update_layout(xaxis={"title": "Vorhergesagte WER",},yaxis={"title": "Anzahl"},)
#fig.write_image("plots/" + "wer_verteilung_hyper" + ".png")
#fig.show()

# with good pseudo labeling
fig = px.histogram(df_good_pseudo,x="pred", nbins=30,
                   text_auto=True,
                   opacity=0.7)
fig.update_layout(bargap=0.1)
fig.update_layout(xaxis={"title": "Vorhergesagte WER",},yaxis={"title": "Anzahl"},)
#fig.write_image("plots/" + "wer_verteilung_pseudo" + ".png")
#fig.show()

# with bad pseudo labeling
fig = px.histogram(df_bad_pseudo,x="pred", nbins=30,
                   text_auto=True,
                   opacity=0.7)
fig.update_layout(bargap=0.1)
fig.update_layout(xaxis={"title": "Vorhergesagte WER",},yaxis={"title": "Anzahl"},)
fig.update_layout(xaxis=dict(tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
)
#fig.write_image("plots/" + "wer_verteilung_bad_pseudo" + ".png")
#fig.show()

## get transcripts with predicted wer values
print(df_bad_pseudo[df_bad_pseudo["pred"] >= 0.7])
#print(df_bad_pseudo[df_bad_pseudo["pred"] < 0.1])

# actually bad transcript
#print(df_without_pseudo[df_without_pseudo["uid"] == 31052023209])
#print(df_good_pseudo[df_good_pseudo["uid"] == 31052023209])

#actually good transcript
#print(df_without_pseudo[df_without_pseudo["uid"] == 33913361187])
#print(df_good_pseudo[df_good_pseudo["uid"] == 33913361187])