# general modules
import sqlite3
import json
import podcastindex
import time

# my modules
from helper import get_file_contents


# the structure of the metadata in the podcast index sql database
"""
CREATE TABLE podcasts (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    lastUpdate INTEGER,
    link TEXT NOT NULL,
    lastHttpStatus INTEGER,
    dead INTEGER,
    contentType TEXT NOT NULL,
    itunesId INTEGER,
    originalUrl TEXT NOT NULL,
    itunesAuthor TEXT NOT NULL,
    itunesOwnerName TEXT NOT NULL,
    explicit INTEGER,
    imageUrl TEXT NOT NULL,
    itunesType TEXT NOT NULL,
    generator TEXT NOT NULL,
    newestItemPubdate INTEGER,
    language TEXT NOT NULL,
    oldestItemPubdate INTEGER,
    episodeCount INTEGER,
    popularityScore INTEGER,
    priority INTEGER,
    createdOn INTEGER,
    updateFrequency INTEGER,
    chash TEXT NOT NULL,
    host TEXT NOT NULL,
    newestEnclosureUrl TEXT NOT NULL,
    podcastGuid TEXT NOT NULL,
    description TEXT NOT NULL,
    category1 TEXT NOT NULL,
    category2 TEXT NOT NULL,
    category3 TEXT NOT NULL,
    category4 TEXT NOT NULL,
    category5 TEXT NOT NULL,
    category6 TEXT NOT NULL,
    category7 TEXT NOT NULL,
    category8 TEXT NOT NULL,
    category9 TEXT NOT NULL,
    category10 TEXT NOT NULL,
    newestEnclosureDuration INTEGER
);

"""


# start a connection the databank
connection = sqlite3.connect("path/to/podcastindex_feeds.db")

# setup a cursor to execute sql statements
cursor = connection.cursor()

# the above schema is used by the database in question here

attribute_names = [ "id", "url", "title", "lastUpdate", "link", "lastHttpStatus", "dead", "contentType", "itunesId", "originalUrl", "itunesAuthor", "itunesOwnerName", 
                    "explicit", "imageUrl", "itunesType", "generator", "newestItemPubdate", "language", "oldestItemPubdate", "episodeCount", "popularityScore",
                    "priority", "createdOn", "updateFrequency", "chash", "host", "newestEnclosureUrl", "podcastGuid", "description", "category1", "category2",
                    "category3", "category4", "category5", "category6", "category7", "category8", "category9", "category10", "newestEnclosureDuration" ]

"""
for index, attr in enumerate(row):
    print(attribute_names[index] + " : " + str(attr))
"""

config = {
    "api_key":  get_file_contents("apikey"),
    "api_secret":  get_file_contents("apisecret")
}

podcast_index = podcastindex.init(config)



def get_metadata(feedId, title, url):
    """
    as we only get the id of the feed from the crawled data we have to find the correct episode
    by matching the title (and/or the url)
    """
    try:
        episodes = podcast_index.episodesByFeedId(feedId, max_results=3000)  
    except Exception as e:
        print(f"Error : {e}")
        return dict(), dict()

    # to not bombard the api with requests we wait between request
    time.sleep(2)

    if len(episodes["items"]) == 0:
        print("Found no episodes for this podcast")
        return dict(), dict()

    
    for episode in episodes["items"]:
        # only get the episode if title and url match
        # I encountered the problem that some episodes of the crawled data are not on the podcastindex
        # anymore, hence there is no metadata for them which makes them useless for us

        if episode["enclosureUrl"] == url and episode["title"] == title:
    
            podcast_metadata = get_metadata_from_sql(feedId)
            if  len(podcast_metadata) == 0:
                print("No podcast found in the database")
                return dict(), dict()
            
            # the information from the sql comes as a tuple and has to be converted to a dict
            podcast_dict = {}
            for index, attr in enumerate(podcast_metadata):
                #print("Assigining " + str(podcast_metadata[index]) + " to " + str(attribute_names[index]))
                podcast_dict[attribute_names[index]] = podcast_metadata[index]

            # my current approach is to return two dicts: one with the metadata and
            # and one one with the episode specific metadata
            if podcast_dict is not None and episode is not None:
                return podcast_dict, episode
            else:
                return dict(), dict()
    
    # if we find no match, we return two empty dicts
    return dict(), dict()
            


def get_metadata_from_sql(feedId):
    row = cursor.execute("SELECT * FROM podcasts WHERE id is "+ feedId +";").fetchone()

    if len(row) == 0:
        return dict()

    return row


"""
episodes = podcast_index.episodesByFeedId(492, max_results=3000)

for episode in episodes["items"]:
    print(episode["title"])
"""