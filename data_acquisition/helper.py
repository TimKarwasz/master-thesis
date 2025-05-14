import csv
import os
import requests

def download(url, episode_id, out_path):
    
    try:
        response = requests.get(url)
    except Exception as e :
        print(f"Error: {e}")
        return
        
    if response.status_code == 200:
        if ".mp3" in url.lower(): 
            extension = ".mp3"
        elif ".m4a" in url.lower():
            extension = ".m4a"
        elif ".wav" in url.lower():
            extension = ".wav"
        elif ".flac" in url.lower():
            extension = ".flac"
        else:
            print(url)
            extension = ".wav"
        
        with open(os.path.join(out_path, str(episode_id) + extension, ), "wb") as file:
            file.write(response.content)
        print("File downloaded successfully")
        return os.path.join(out_path, str(episode_id) + extension)
    else:
        print(f"Failed to download the audio file {url}")
        return None
    
    
def get_file_contents(filename):
    """ Given a filename,
        return the contents of that file
    """
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)

def read_csv(path_to_csv, delimiter):
    with open(path_to_csv, newline='') as csv_file:
        data = list(csv.reader(csv_file, delimiter = delimiter))
    return data
