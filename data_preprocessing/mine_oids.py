# This script queries data from the Alerce database for a series of years and saves daily object IDs (oids) 
# with their corresponding number of detections (ndet) into Parquet files. The script calculates the 
# Modified Julian Date (MJD) for each day of the specified years and ensures the process resumes from 
# the last successfully processed date. 
#
# Main steps:
# 1. The calculate_mjd function computes the Modified Julian Date from a given date string.
# 2. The calculate_mjd_for_year function generates the MJD for every day of a given year.
# 3. For each year, the script checks if there are already processed files in the corresponding directory  and resumes processing from the next MJD.
# 4. The script connects to the Alerce database and retrieves object data based on the calculated MJD ranges.
# 5. The queried data is saved as Parquet files with filenames corresponding to the MJD.
# 6. If an error occurs during querying, the script logs the error and continues to the next date.


import pandas as pd
from datetime import datetime, timedelta
import sys
import pandas as pd
import numpy as np

from alerce.core import Alerce
client = Alerce()
import requests
import sqlalchemy as sa
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging
import glob
from sqlalchemy.exc import SQLAlchemyError  
import time

class OIDMiner:
    def __init__():
        pass

def calculate_mjd(date_string):
    # Parse the input string
    day = int(date_string[:2])
    month = int(date_string[2:4])
    year = int(date_string[4:])
    
    # Add 2000 to the year if it's less than 100
    if year < 100:
        year += 2000
    
    # Create a datetime object
    date = datetime(year, month, day)
    
    # Calculate the Julian Date
    jd = date.toordinal() + 1721424.5
    
    # Calculate the Modified Julian Date
    mjd = jd - 2400000.5
    
    # Return the MJD as a float
    return round(mjd, 2)

def calculate_mjd_for_year(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    mjd_list = []
    current_date = start_date
    while current_date <= end_date:
        date_string = current_date.strftime("%d%m%Y")
        mjd = calculate_mjd(date_string)
        mjd_list.append((current_date.strftime('%Y-%m-%d'), mjd))
        current_date += timedelta(days=1)
    
    return mjd_list

# Set up logging
logging.basicConfig(filename='query_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

 
 
 
years = [2016,2017,2018,2019,2020,2021,2022,2023]

core_path = '/home/magdalena/Desktop/magister/SSL_DATASET/ZTF_SSL_Dataset'
import time
from sqlalchemy.exc import OperationalError

for year in years:
    # Create a directory for the year
    year_dir = os.path.join(core_path, str(year))
    os.makedirs(year_dir, exist_ok=True)


    mjd_results = calculate_mjd_for_year(year)
    
    # Find the latest processed MJD
    existing_files = glob.glob(os.path.join(year_dir, 'oids_*.parquet'))
    if existing_files:
        latest_file = max(existing_files, key=os.path.getctime)
        latest_mjd = float(latest_file.split('_')[-1].split('.')[0])
        start_index = next(i for i, (_, mjd) in enumerate(mjd_results) if mjd > latest_mjd)
    else:
        start_index = 0

    for day in tqdm(range(start_index, len(mjd_results)-1)):
        try:
            client = Alerce()
            url = 'https://raw.githubusercontent.com/alercebroker/usecases/master/alercereaduser_v4.json'
            params = requests.get(url).json()['params']
            engine = sa.create_engine('postgresql+psycopg2://' + params['user'] \
                            + ':' + params['password'] + '@' + params['host'] \
                            + '/' + params['dbname'])
            conn = engine.connect()
            query = f"""SELECT oid, ndet
                FROM alerce."object" o
                WHERE mjdstarthist >= {mjd_results[day][1]} AND mjdstarthist < {mjd_results[day+1][1]}
                """
            df_day = pd.read_sql_query(query, conn)
            df_day = df_day.drop_duplicates('oid').set_index('oid')
            
            # Save the file in the year directory
            file_name = f'oids_{str(mjd_results[day][1]).replace(".0","")}.parquet'
            file_path = os.path.join(year_dir, file_name)
            df_day.to_parquet(file_path)
            conn.close()
        except Exception as e:
            # Log the error with the MJD value that failed
            conn.close()

            error_msg = f"Error querying for MJD {mjd_results[day][1]}: {str(e)}"
            logging.error(error_msg)
            print(error_msg)  # Also print to console for immediate feedback
            continue  # Move to the next iteration
