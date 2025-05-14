# This file holds the implementation for every feature for the transcribed texts. The idea is to save all features in a json file, so that there
# is a one to one correlation of podcasts to json files. After every calculation of a feature it is directly written to the the json file. This might
# be less efficient than one single big writing operation but acts as a failsafe in case something happens (power outage etc...)

# my own modules
from feature_helper import read_feature_json, read_transcription_json, write_features, read_reference_txt, find_phrase
from calc_yngve_frazier import calc_scores

# general imports
import shutil
import os
import csv
import time
import textstat
import textdescriptives as td
import spacy
import pickle
import numpy as np
import pandas as pd
from nltk.parse.corenlp import CoreNLPParser
import language_tool_python
import jiwer
from transformers import BertTokenizer, BertModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from nltk import FreqDist
from nltk.util import ngrams
from nltk.translate.meteor_score import meteor_score
from lingua import Language, LanguageDetectorBuilder


# setup paths
base_path = "path/to/base/dir/"

output_path = os.path.join(base_path, "created_features/")          
data_path = os.path.join(base_path, "transcribed_podcasts/")
metadata_path = os.path.join(base_path, "metadata/")
slow_features_path = os.path.join(base_path, "slow_features/")
ref_path = os.path.join(base_path, "ref/")
processed_podcasts_path = os.path.join(base_path, "processed_podcasts/")


# setup global stuff once
nlp = spacy.load("de_core_news_md")
textstat.set_lang("de")
nlp.add_pipe("textdescriptives/all")
parser = CoreNLPParser(url='http://localhost:9000')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
tool = language_tool_python.LanguageTool('de-DE', config={ 'cacheSize': 1000, 'pipelineCaching': True })
languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.INDONESIAN,
             Language.PORTUGUESE, Language.TURKISH, Language.HEBREW, Language.POLISH, Language.ITALIAN,
             Language.DUTCH, Language.UKRAINIAN, Language.KOREAN, Language.RUSSIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

# time stuff
start_time = time.time()
counter = 0
removed_counter = 0
removed_podcasts_ids = []
# analyze each transcript in the dir
for filename in os.listdir(data_path):
    
    # 0. Setup
    # filename.split(".")[0] returns the 123 from 123.json 
    podcast_id = str(filename.split(".")[0])
    print(f"Started processing {podcast_id}")
    transcribed_text = read_transcription_json(data_path, podcast_id)["text"]
    
    ##########################################################################################################################################

    # 1. Firstly check that a .pkl file with metadata for the transcription exists
    # if there is no metadata file skip the iteration
    
    if not os.path.isfile(metadata_path + podcast_id + ".pkl"):
        continue

    # 1.1 Check if if one 4-grams represents over 5% of the total 4-gram frequency
    # idea is from https://arxiv.org/pdf/2411.07892

    tokens = word_tokenize(transcribed_text)
    four_grams = list(ngrams(tokens, 4))
    
    freq = pd.Series(dict(FreqDist(four_grams)))
    df = pd.DataFrame(list(freq.items()), columns = ["ngram","freq"])
    five_percent = round((5 / 100) * df["freq"].sum() ,2)
    
    loop_breaker = False
    four_gram = ""
    four_gram_freq = 0
    for (elem,ngram) in zip(df["freq"], df["ngram"]):
        if elem > five_percent:
            four_gram = ngram
            four_gram_freq = elem
            loop_breaker = True
            break
    
    if loop_breaker:
        # this if clause triggers when the such a 4-grams exists
        four_grams_features = {}
        four_grams_features["four_gram"] = " ".join(four_gram)
        four_grams_features["four_gram_percent"] = 100 * float(four_gram_freq)/float(df["freq"].sum())
        four_grams_features["four_gram_triggered"] = "yes"
        write_features(output_path, podcast_id, four_grams_features)
        #continue
    else:
        four_grams_features = {}
        four_grams_features["four_gram_triggered"] = "no"
        write_features(output_path, podcast_id, four_grams_features)
    
    
    ##########################################################################################################################################
                                                          
    # 2. If there is metadata read it and write it to the json
    with open(metadata_path + podcast_id + ".pkl", 'rb') as f:
        metadata = pickle.load(f)
    

    write_features(output_path, podcast_id, metadata)
    print(f"Wrote metadata features after {time.time() -start_time} sec")

    ##########################################################################################################################################
       
    # 3. Get all features from textdescriptves and spacy
    doc = nlp(transcribed_text)
    features = td.extract_dict(doc)[0]

    # the readability metrics flesch kincaid grade, automated readability index and colemanliau index
    # were orignally made for english american texts
    to_be_removed = ["text", "flesch_reading_ease", "flesch_kincaid_grade", "automated_readability_index", "coleman_liau_index"]
    for feature in to_be_removed:
        features.pop(feature)
    
    # the textdescriptives module calculates the flesch reading ease with the english formula
    # hence we get that feature from the textstat module where the correct formula is used
    features["flesch_reading_ease"] = textstat.flesch_reading_ease(transcribed_text)
    features["wiener_sachtext_formel"] = textstat.wiener_sachtextformel(transcribed_text,1)
    features["syllables_text_count"] = sum(doc._._n_syllables)
        
    write_features(output_path, podcast_id, features)
    print(f"Wrote spacy features after {time.time() - start_time} sec")
    
    ##########################################################################################################################################

    # 4. Calculate frazier and yngve scores
    # this needs :
    # the doc on the CoreNLP server is here : https://stanfordnlp.github.io/CoreNLP/corenlp-server.html
    # to start the parsing server run this cmd : java -Xmx3G -cp C:\Users\Tim\stanza_corenlp\* edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 -threads 5 -maxCharLength 100000 -quiet False -serverProperties german -annotators tokenize,ssplit,pos,parse -preload -outputFormat serialized
    # Error 1: parsing takes to long and throws a Timeout Error
    try:
        frazier = []
        yngve = []
        
        for sent in doc.sents:
            sentence = str(sent)
            if len(sentence) > 2:
                tree = next(parser.raw_parse(sentence))
                frazier_tot,yngve_tot = calc_scores(tree)
                frazier.append(frazier_tot)
                yngve.append(yngve_tot)
        
        yngve_arr = np.array(yngve)
        frazier_arr = np.array(frazier)
        
        syntactic_complexity = {}
        syntactic_complexity["frazier_mean"] = np.mean(frazier_arr)
        syntactic_complexity["yngve_mean"] = np.mean(yngve_arr)
        
        write_features(output_path, podcast_id, syntactic_complexity)
        print(f"Wrote text complexity features after {time.time() - start_time} sec")
    except Exception as e:
        print(f"Skipped text complexity features after {time.time() - start_time} sec due to timeout")
        print(f"Error : {e}")

    ##########################################################################################################################################
    
    # 5. Run the grammar checker
        
    matches = tool.check(transcribed_text)
        
    rule_id_dict = {}
    category_dict = {}
    
    # Add every ruleId and every category of the matches to a dict
    for match in matches:
        
        if match.ruleId in rule_id_dict.keys():
            rule_id_dict[match.ruleId] += 1
        else:
            rule_id_dict[match.ruleId] = 1

        if match.category in category_dict.keys():
            category_dict[match.category] += 1
        else:
            category_dict[match.category] = 1
    
    grammar_checker_features = {}
    grammar_checker_features["grm_checker_ruleIds"] = rule_id_dict
    grammar_checker_features["grm_checker_categories"] = category_dict
    grammar_checker_features["grm_checker_count"] = len(matches)
    
    write_features(output_path, podcast_id, grammar_checker_features)
    print(f"Wrote grammar checker features after {time.time() - start_time} sec")
            
    ##########################################################################################################################################       
            
    # 6. Speech duration measures
    # Metrics :
    # number of pauses, Total Pause Time, Mean Duration of Pauses ( Total time/ number of pauses )
    # The Standardized Pause Rate is the number of words in the sample divided by the number of pauses
    # The Total Phonation Time is the amount of time, in seconds, in the sample that contains speech events.
    # The Standardized Phonation Time is the number of words in the sample divided by the Total Phonation Time.
    # The Total Locution Time is the amount of time in the sample that contains both speech and pauses.
    # The Phonation Rate is the Total Phonation Time divided by the Total Locution Time.
    # The Verbal Rate is the number of words in the sample divided by the Total Locution Time.
    
    segments = read_transcription_json(data_path, podcast_id)["segments"] 
    
    total_duration = int(metadata["episode"]["duration"])
    total_segments = len(segments)
    number_of_words = features["n_tokens"]
    total_locution_time = total_duration
    
    pause_counter = 0
    total_pause_time = 0
    
    for segment in segments:
        
        # in each iteration check the pause between the current segment and the next
        # if we are in the last segment exit the for loop
        if int(segment["id"]) + 1 == total_segments:
            break 
        
        pause_time = segments[segment["id"]+1]["start"] - segment["end"] 
        
        # We follow Singh et al. [16] in setting a 1-second minimum for counting silence as pause
        if pause_time > 1:
            
            pause_counter += 1
            total_pause_time += pause_time
            
    # Error 2: Make sure we do not divide by 0
    if pause_counter != 0:
        standardized_pause_rate = number_of_words / pause_counter
        mean_pause_duration = total_pause_time / pause_counter
    elif pause_counter == 0:
        standardized_pause_rate = 0
        mean_pause_duration = 0
        
    total_phonation_time = total_duration - total_pause_time
    
    if total_locution_time != 0:
        phonation_rate = total_phonation_time / total_locution_time
        verbal_rate = number_of_words / total_locution_time
    elif total_locution_time == 0:
        phonation_rate = 0
        verbal_rate = 0
    
    if total_phonation_time != 0:
        standardized_phonation_time = number_of_words / total_phonation_time
    # if the total_phonation time is 0 the duration is of the podcast is either 0 or just pauses
    elif total_phonation_time == 0:
         standardized_phonation_time = 0

    pause_features = {}
    pause_features["number_of_pauses"] = pause_counter 
    pause_features["total_pause_time"] = total_pause_time
    pause_features["standardized_pause_rate"] = standardized_pause_rate
    pause_features["total_phonation_time"] = total_phonation_time
    pause_features["standardized_phonation_time"] = standardized_phonation_time
    pause_features["mean_pause_duration"] = mean_pause_duration
    pause_features["phonation_rate"] = phonation_rate
    pause_features["verbal_rate"] = verbal_rate
    
    write_features(output_path, podcast_id, pause_features)
    print(f"Wrote pause features after {time.time() - start_time} sec")
    
    
    ##########################################################################################################################################

    # 7. Check here if we have audiofeatures features (syllables from audio and gemaps) for the podcast
    
    syllables_audio_path = os.path.join(slow_features_path, podcast_id + ".pkl")
    gemaps_mid_path = os.path.join(slow_features_path, podcast_id + "_middle" + ".csv")

        
    if os.path.isfile(syllables_audio_path):
        
        with open(syllables_audio_path, 'rb') as f:
            syllables = pickle.load(f)
            
        syllables_audio_count = syllables["number_syllabels"]
    
    if os.path.isfile(gemaps_mid_path):
        
        gemaps_mid_arrays = {}
        with open(gemaps_mid_path, "r") as f:
            reader = csv.DictReader(f, delimiter=",")
            
            lines_mid = []
            for row in reader:
                lines_mid.append(row)

        for key in lines_mid[0].keys():
            if key != "start" and key != "end":
                gemaps_mid_arrays[key] = []

        for line in lines_mid:
            for key in line.keys():
                if key != "start" and key != "end":
                    if line[key] ==  "":
                        gemaps_mid_arrays[key].append(float(0))
                    else:
                        gemaps_mid_arrays[key].append(float(line[key]))
        
        mid_features = {}
        for key in gemaps_mid_arrays.keys():
            
            gemaps_mid_arrays[key] = np.array(gemaps_mid_arrays[key])
            mid_features[key + "_middle"] = np.mean(gemaps_mid_arrays[key])
        mid_features["syllables_audio_count"] = int(syllables_audio_count)
        
        write_features(output_path, podcast_id, mid_features)
        print(f"Wrote slow features after {time.time() - start_time} sec")
    
    
    ##########################################################################################################################################
    
    # 8. Reference metrics
    # This section only triggers if either the wdr or hinter den dingen or german test data is used
    
    if "wdr" in base_path or "hinter" in base_path or "german" in base_path or "nachrichten" in base_path or "all_manual" in output_path :
        
        hypothesis = transcribed_text
        reference = read_reference_txt(ref_path, podcast_id)
      
        # 8.1 Metrics
        transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        
        wer = jiwer.wer(reference, hypothesis, truth_transform=transforms, hypothesis_transform=transforms)
        mer = jiwer.mer(reference, hypothesis, truth_transform=transforms, hypothesis_transform=transforms)
        
        all_metrics = jiwer.compute_measures(truth=reference, hypothesis=hypothesis, truth_transform=transforms, hypothesis_transform=transforms)
        
        substitutions = all_metrics["substitutions"]
        deletions = all_metrics["deletions"]
        insertions = all_metrics["insertions"]
        
        # 8.2 Rogue score
        rouges_scores = scorer.score(reference, hypothesis)
        rouge1_prec = rouges_scores["rouge1"][0]
        rouge1_rec = rouges_scores["rouge1"][1]
        rouge2_prec = rouges_scores["rouge2"][0]
        rouge2_rec = rouges_scores["rouge2"][1]
        rougeL_prec = rouges_scores["rougeL"][0]
        rougeL_rec = rouges_scores["rougeL"][1]
        
        # 8.3 Bleu score
        bleu_score = sentence_bleu([word_tokenize(reference)], word_tokenize(hypothesis))
        
        # 8.4 Meteor score
        met_score = meteor_score([word_tokenize(reference)], word_tokenize(hypothesis)) 

        # 8.5 BERTScore 
        tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
        model = BertModel.from_pretrained("dbmdz/bert-base-german-uncased")
        inputs1 = tokenizer(hypothesis, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()
        similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

        # 8.6 NER Score 
        nlp_hypothesis = nlp(hypothesis)
        nlp_reference  = nlp(reference)
        
        reference_metrics = {}
        reference_metrics["wer"] = wer
        reference_metrics["mer"] = mer
        reference_metrics["substitutions"] = substitutions
        reference_metrics["deletions"] = deletions
        reference_metrics["insertions"] = insertions
        reference_metrics["rouge1_prec"] = rouge1_prec
        reference_metrics["rouge1_rec"] = rouge1_rec
        reference_metrics["rouge2_prec"] = rouge2_prec
        reference_metrics["rouge2_rec"] = rouge2_rec
        reference_metrics["rougeL_prec"] = rougeL_prec
        reference_metrics["rougeL_rec"] = rougeL_rec
        reference_metrics["bleu_score"] = bleu_score
        reference_metrics["meteor_score"] = met_score
        reference_metrics["BERTScore"] = float(str(similarity[0][0]))
        reference_metrics["hypothesis_ner_score"] = len(nlp_hypothesis.ents)
        reference_metrics["reference_ner_score"] = len(nlp_reference.ents)
    
        write_features(output_path, podcast_id, reference_metrics)
        print(f"Wrote reference metrics after {time.time() - start_time} sec")
    
    ##########################################################################################################################################

    # 9. Detect the language in the transcript 
    
    prob_is_german = detector.compute_language_confidence(transcribed_text, Language.GERMAN)
    language = detector.detect_language_of(transcribed_text)
    detected_language = language.iso_code_639_3.name
    confidence_values = detector.compute_language_confidence_values(transcribed_text)[:3]
    top_3_most_prob_lang = []
    for confidence in confidence_values:
        top_3_most_prob_lang.append((confidence.language.name,confidence.value))

    lang_features = {}
    lang_features["prob_is_german"] = prob_is_german
    lang_features["detected_language"] = detected_language
    lang_features["top_3_most_prob_lang"] = top_3_most_prob_lang
    
        
    write_features(output_path, podcast_id, lang_features)
    print(f"Wrote lang features after {time.time() - start_time} sec")

    ##########################################################################################################################################
    
    
    # 10. Detect if and how many times "Untertitelung des ZDF" occurs in the transcribed text
    
    phrase = "Untertitelung des ZDF"
    phrase_features = {}
    
    if phrase in transcribed_text:

        if "ZDF" not in metadata["podcast"]["itunesAuthor"]:
            amount  = find_phrase(transcribed_text, phrase)
            
            phrase_features["phrase_found"] = True
            phrase_features["phrase_amount"] = amount
        
        else:
             phrase_features["phrase_found"] = False
             phrase_features["phrase_amount"] = 0
    
    else:
        phrase_features["phrase_found"] = False
        phrase_features["phrase_amount"] = 0
    
    write_features(output_path, podcast_id, phrase_features)
    print(f"Wrote phrase features after {time.time() - start_time} sec")
    
    ##########################################################################################################################################
    
    
    # Finally move the transcription file to another dir (so it can still be manually analyzed)
    shutil.move(os.path.join(data_path, filename), os.path.join(processed_podcasts_path, filename))
     
    print(f"Finished processing {podcast_id} after {time.time() - start_time} sec ")
    counter += 1
    print(f"Counter is at {counter} podcasts")
    print("#"*70)
    
    
tool.close()   
print(f"Finished processing {counter} podcasts after {time.time() - start_time} sec")

