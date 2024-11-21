import pandas as pd 
import os
import glob

"""
a script to extract oids (with ndet > 2) from the database by day of a year and save as .parquet 

"""



years = [2016,2017,2018,2019,2020,2021,2022,2023]
core_path = '/home/magdalena/Desktop/magister/SSL_DATASET/ZTF_SSL_Dataset' #where the parquest will save
parquets = []

for year in years:
    # Create a directory for the year
    year_dir = os.path.join(core_path, str(year))
    # Find the latest processed MJD
    existing_files = glob.glob(os.path.join(year_dir, 'oids_*.parquet'))
    for file in existing_files:
        parquets.append(file)
 
dfs = [pd.read_parquet(file) for file in parquets]
 
combined_df = pd.concat(dfs, ignore_index=False)
nomjd = combined_df.drop(columns = 'mjdstarthist')
nomjd = nomjd[nomjd['ndet']> 2]
nomjd.sort_values(by = 'ndet', ascending=False)
nomjd.to_parquet(f'{core_path}/clean_oids.parquet')
 
print('retained percentage:')
print(nomjd.shape[0]/combined_df.shape[0])



