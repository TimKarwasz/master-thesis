import os
import json


def write_features(output_path, podcast_id, dictionary):
# therefore the method will firstly check if such a json file exists. If one exists, it will be read and the containing dictionary will be appended
# to make these json files distinguishable from the transcriptions they will have the suffix _features

    # make sure that we always work with a string 
    podcast_id = str(podcast_id)
    
    file_path = output_path + podcast_id + "_features.json"
    
    if os.path.isfile(file_path):
        
        # if the file exists we read the dict and then write to it
        with open(file_path, 'r') as f:

            file_data = json.load(f)
            file_data.update(dictionary)
        
        with open(file_path, 'w') as f:
            
             f.write(json.dumps(file_data, indent=4))
        
    else:
        
        # if it does not exist we create it and write to it 
        with open(file_path, 'w') as f:

            f.write(json.dumps(dictionary, indent=4))
            



def read_feature_json(data_path, podcast_id):
# this method returns the data dictionary of a given podcast id

    # make sure that we always work with a string 
    podcast_id = str(podcast_id)
    
    file_path = data_path + podcast_id + "_features.json"
    
    if os.path.isfile(file_path):
        
        with open(file_path, 'r') as f:

            file_data = json.load(f)
    
        return file_data
    
    else: 
        
        print(f"There is no json file for {podcast_id}")
        

def read_transcription_json(data_path, podcast_id):
# this method returns the transcription dictionary for a given podcast id

    # make sure that we always work with a string 
    podcast_id = str(podcast_id)
    
    file_path = data_path + podcast_id + ".json"
    
    if os.path.isfile(file_path):
        
        with open(file_path, 'r') as f:

            file_data = json.load(f)
    
        return file_data
    
    else: 
        
        print(f"There is no json file for {podcast_id}")
        


def read_reference_txt(data_path, podcast_id):
    
    podcast_id = str(podcast_id)
    
    file_path = data_path + podcast_id + "_ref.txt"
    
    if os.path.isfile(file_path):
        
        with open(file_path, 'r', encoding="utf8") as f:
            
            content = f.read()
            
        return content
    
    else: 
        
        print(f"There is no txt file for {podcast_id}")
        

def find_phrase(text, phrase, num_surrounding=15):
    words = text.split()
    phrase_words = phrase.split()
    phrase_len = len(phrase_words)

    results = []

    #print(f"total length of words : {len(words)}")
    # Scan through the text word by word
    for i in range(len(words) - phrase_len + 1):
        # Compare a window of words to the phrase
        window = words[i:i + phrase_len]
        if [w.lower().strip('.,!?;:"\'') for w in window] == [w.lower() for w in phrase_words]:
            start = max(0, i - num_surrounding)
            end = min(len(words), i + phrase_len + num_surrounding)
            snippet = words[start:end]
            results.append(' '.join(snippet))

    #print(f"Found {len(results)} snippets")
    return len(results)