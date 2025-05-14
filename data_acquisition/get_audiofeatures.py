# general modules
import os
import sys 
from pydub import AudioSegment
mysp=__import__("my-voice-analysis")
import audiofile
import opensmile
import pickle
import time 

day_number = "12"

start_time = time.time()
# this is the path to where we store all podcasts that we want the syllabels and the audio featuires from
podcast_path = "/work/tk81fevo-whisper2/tk81fevo-whisper-transcription-1739240423/data/slow_features/podcasts/"

# this the path where all the slow_features will be stored
metadata_path = "/work/tk81fevo-whisper2/tk81fevo-whisper-transcription-1739240423/data/slow_features/metadata/"

counter = 0
for filename in os.listdir(podcast_path):

    if "praat" in filename:
        continue 
    
    podcast_id = str(filename.split(".")[0])
    audio_path = podcast_path + filename
    
    print(f"Processsing {audio_path}")

    # firstly convert the file to .wav if it is not already in that format
    if not audio_path.endswith(".wav") and not audio_path.endswith(".TextGrid"):
        
        # if this block catches an error that means the audio file did contain an audio stream
        try:
            audio_file = AudioSegment.from_file(audio_path, format=audio_path.split(".")[-1])
        except Exception as e:
            print(f"Error:  {e}")
            continue
        
        audio_file.export(os.path.join(podcast_path, podcast_id + ".wav"), format="wav")
        
        # delete the old file
        os.remove(audio_path)
            
        # and update audio_path 
        audio_path = os.path.join(podcast_path, podcast_id + ".wav")
    
    # get the audio duration
    audio_file = AudioSegment.from_file(audio_path)
    duration = audio_file.duration_seconds
    # only try and get the syllabels if the podcast is shorter than 1 hour, otherwise
    # we run into a out of memory error
    print(f"Duration before the if : {duration}")
    
    if duration < 5400:
        print(f"Duration in the if : {duration}")
        #next also get the numberefore the of syllabels in the original audio here
        syl_dict = {}
        syl_dict["number_syllabels"] = str(mysp.myspsyl(podcast_id,podcast_path))

        with open(os.path.join(metadata_path, podcast_id + ".pkl"),  "wb") as f:
            pickle.dump(syl_dict, f)
            print("Dictionary saved successfully to file")
        
        try:
            os.remove(os.path.join(podcast_path, podcast_id + ".TextGrid"))
        except Exception as e:
            print(f"Error : {e}")
          
        
        # get the low level descriptors here for the middle 10sec
        if duration > 22: 
            middle = duration / 2
            signal, sampling_rate = audiofile.read(audio_path,offset=middle - 10, duration=10, always_2d=True) 
            smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.LowLevelDescriptors, num_workers=5, multiprocessing=True)
            df = smile.process_signal(signal,sampling_rate)
            df.to_csv(os.path.join(metadata_path, podcast_id + "_middle" + ".csv"))
        
            print(f"Got middle features after {time.time() -start_time} for {podcast_id}")

        counter += 1
        os.remove(audio_path)
    else:
        os.remove(audio_path)

        
print(f"Processed {counter} podcasts after {time.time() -start_time} sec")
print(f"Processed podcasts from day {day_number}")