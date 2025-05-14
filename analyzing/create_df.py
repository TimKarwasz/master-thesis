import os
import json
import pandas as pd

# Code to generate the dataframes which are then used for the plots and the machine learning 
data_path = "path/to/created_features/files"


rows = []
# for every file in the dir
for filename in os.listdir(data_path):
    
    row = {}
    file_path = os.path.join(data_path, filename)

    if os.path.isfile(file_path):
        
        with open(file_path, 'r') as f:

            file_data = json.load(f)
            
        for key in file_data.keys():
            if key not in ["podcast", "episode"]:
                row[key] = file_data[key]
        
            # this adds a lot of rows
            """
            if key == "grm_checker_ruleIds":
                # some json fields have dict in them so we need to unpack them into seperate rows
                for meta_key in file_data[key].keys():
                    row["grm_rule_" + meta_key] = file_data[key][meta_key]
            
            if key == "grm_checker_categories":
                for meta_key in file_data[key].keys():
                    row["grm_category_" + meta_key] = file_data[key][meta_key]#
             """
                    
            if key in ["podcast"]:
                for meta_key in file_data[key].keys():
                    row["podcast_" + meta_key] = file_data[key][meta_key]
            
            if key in ["episode"]:
                for meta_key in file_data[key].keys():
                    if meta_key == "id":
                        row["uid"] = file_data[key][meta_key]
                    else:
                        row["episode_" + meta_key] = file_data[key][meta_key]           
                
        rows.append(row)
    

df = pd.DataFrame(rows)
print(df)
df.to_csv('df.csv', index=False)