# general modules
import os
import sys 
mysp=__import__("my-voice-analysis")
import pickle
import time

# my modules
from helper import download, read_csv
from get_podcastindex_metadata import get_metadata

"""
this is the path to the csv files which are crawled by the SAW
they are tab separated
the format is : FeedId  timestamp   title   audioUrl 
Note: there is also a mini datastore, which makes sure that that we do not download audio files twice
"""
start_time = time.time()

day_number = "10"

# where the feedlists are
csv_dir = "path/to/csv_dir/"

# this is the path to where we want to store the downloaded audio files
output_dir = "path/to/podcasts/"

# this the path where all the metadata will be stored
metadata_dir = "path/to/archive/2024-12-" + day_number + "_00-00_de/metadata/"

log_dir = "path/to/download_logs/"

found_counter = 0
missed_counter = 0

for filename in os.listdir(csv_dir):
    file = os.path.join(csv_dir, filename)
    print(f"Processing : {filename}")
    if os.path.isfile(file):
        data = read_csv(file, "\t")
        for row in data:
            try:
                feedId = row[0]
                title = row[2]
                url = row[3]
            except Exception as e:
                print(f"Error: {e}")
                continue
            #print(f"{feedId}, {title}, {url}")
            podcast_metadata, episode_metadata = get_metadata(feedId, title, url)

            #print(f"episode : {len(episode_metadata)} ; podcast : {len(podcast_metadata)}")
            # check that we have metadata for episode and podcast here
            if len(podcast_metadata) != 0 and len(episode_metadata) != 0:
                found_counter +=1
                
                audio_path = download(url, episode_metadata["id"], output_dir)
                
                # merge the episode and podcast dict into one:
                metadata_dict = {}
                metadata_dict["episode"] = episode_metadata
                metadata_dict["podcast"] = podcast_metadata
                
                with open(os.path.join(metadata_dir, str(episode_metadata["id"]) + ".pkl"),  "wb") as f:
                        pickle.dump(metadata_dict, f)
                        print("Dictionary saved successfully to file")            
                
            else:
                missed_counter += 1
                

total_time = time.time() - start_time
with open(os.path.join(log_dir, day_number + ".log"), "a") as f:
    f.write(str(total_time))


print(f"Found : {found_counter}")
print(f"Missed : {missed_counter}")

