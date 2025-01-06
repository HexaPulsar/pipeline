import pandas as pd 
import os
import glob
 
years = [#2016,
         #2017,
         #2018,
         2019,
         2020,
         2021,
         2022,
         2023]

core_path = '/home/magdalena/Desktop/magister/SSL_DATASET/ZTF_SSL_Dataset'
out_path = '/home/magdalena/Desktop/magister/SSL_DATASET/ZTF_SSL_Dataset/SSL/'
for year in years:
    # Create a directory for the year
    year_dir = os.path.join(core_path, str(year))
    # Find the latest processed MJD
    existing_files = glob.glob(os.path.join(year_dir, 'oids_*.parquet'))
     
    
    dfs = [pd.read_parquet(file) for file in existing_files]
    
    combined_df = pd.concat(dfs, ignore_index=False)
    print(combined_df.shape)
    nomjd = combined_df.drop(columns = 'mjdstarthist') if 'mhdstarthist' in combined_df.columns else combined_df
    nomjd = nomjd[nomjd['ndet']>=6]
    nomjd.sort_values(by = 'ndet', ascending=False)
    nomjd.to_parquet(f'{core_path}/{year}_plussixdet_oids_.parquet')
    print('retained n',nomjd.shape)
    print(f'retained percentage for year {year}:')
    print(nomjd.shape[0]/combined_df.shape[0])






