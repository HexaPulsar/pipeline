import io

import pandas as pd
import requests

from ..spark_init.pyspark_configs import *

# It's necessary to adjust this value as necessary
_BATCH_SIZE = 10000 # Number of simultaneous requests to CDS Xmatch service
_NSIDE_RES = 9  # Used just to order the ipix. Not sure if it actually affects that much though
_RETRIES = 3
_RETRY_INTERVAL = 1
CATALOG_MAP = {
    "allwise":"vizier:II/328/allwise"
}

def pandas_dataframe_chunks(dataframe, batch_size):
    list_dataframes = [dataframe[i:i+batch_size] for i in range(0,len(dataframe),batch_size)]
    return list_dataframes

def generate_parameters_request_xmatch(distMaxArcsec = 1, selection = 'best', RESPONSEFORMAT = 'csv', cat2 = 'allwise', colRA1 = 'ra_in', colDec1 = 'dec_in', cols2 = 'AllWISE,RAJ2000,DEJ2000,W1mag,W2mag,W3mag,W4mag,e_W1mag,e_W2mag,e_W3mag,e_W4mag,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag,eeMaj,eeMin,eePA,ID,ccf,ex,var,qph,pmRA,e_pmRA,pmDE,e_pmDE,d2M'):
    CDS_Catalogues_MAP = {
        'allwise': 'vizier:II/328/allwise',
    }
    catalog_xmatch = CDS_Catalogues_MAP[cat2]
    dict_parameters = {'request': 'xmatch', 
                       'distMaxArcsec': distMaxArcsec, 
                       'selection': selection, 
                       'RESPONSEFORMAT': RESPONSEFORMAT, 
                       'cat2': catalog_xmatch, 
                       'colRA1': colRA1, 
                       'colDec1': colDec1, 
                       'cols2': cols2}
    return dict_parameters

def request_xmatch(input_catalog, retries_count, xmatch_client, parameters, retry_interval):
    if retries_count > 0:
            try:
                result = xmatch_client.execute(
                    input_catalog,
                    parameters["input_type"],
                    parameters["catalog_alias"],
                    parameters["columns"],
                    parameters["selection"],
                    parameters["output_type"],
                    parameters["radius"],
                )
                return result
            
            except Exception as e:
                print(f"CDS xmatch client returned with error {e}")
                time.sleep(retry_interval)
                print("Retrying request")
                return request_xmatch(input_catalog, retries_count - 1, xmatch_client, parameters, retry_interval)

    if retries_count == 0:
            print(f"Retrieving xmatch from the client unsuccessful after {retries_count} retries. Shutting down.")
            raise Exception(f"Could not retrieve xmatch from CDS after {retries_count} retries.")
    return

def execute_xmatch_requests(list_dataframe_pandas, parameters, dataframe_crossmatch_chunks, retries_count = _RETRIES, retry_interval = _RETRY_INTERVAL):
    r = requests.Session()
    xmatch_batch_results = pd.DataFrame()
    processed_batch = pd.DataFrame() # The ones that must be joined with xmatch_batch_results
    unprocessed_batch = pd.DataFrame() # The ones that must be written to disk when CDS returns an error
    for i in range(len(list_dataframe_pandas)):
        retries_count = _RETRIES   # Update retries with original value, in case the last dataframe had to be resent to CDS
        print(f"Executing batch number {i} of {len(list_dataframe_pandas)}")
        print(f"Batch {i} has a size of {len(list_dataframe_pandas[i])} oids to process")
        catalog = list_dataframe_pandas[i]
        string_io = io.StringIO()
        new_columns = {}
        for c in catalog.columns:
            new_columns[c] = "%s_in" % c
            catalog = catalog.rename(columns=new_columns)
        catalog.to_csv(string_io)
        catalog_content = string_io.getvalue()
        while retries_count > 0:
            try:
                response = r.post('http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync', data=parameters, files={'cat1': catalog_content}, timeout=100)
                break
            except Exception as e:
                print(f"CDS xmatch client returned with error {e}")
                time.sleep(retry_interval)
                print("Retrying request")
                retries_count -= 1 # Update retries, so it will retry one time less

        if retries_count == 0:
            print(f"Retrieving xmatch from the client unsuccessful after {_RETRIES} retries. Writing unprocessed batch to disk and continuing with already finished requests")
            for j in range(i, len(list_dataframe_pandas)):
                unprocessed_batch = pd.concat([unprocessed_batch, dataframe_crossmatch_chunks[j]])
            return xmatch_batch_results, processed_batch, unprocessed_batch
            
            #raise Exception(f"Could not retrieve xmatch from CDS after {_RETRIES} retries.")
        xmatches = pd.read_csv(io.StringIO(response.text))
        processed_batch = pd.concat([processed_batch, dataframe_crossmatch_chunks[i]])
        xmatch_batch_results = pd.concat([xmatch_batch_results, xmatches])
    return xmatch_batch_results, processed_batch, unprocessed_batch


def execute_batchsizes(spark, dataframe_spark):
    print(f"Processing using a batch size of {_BATCH_SIZE} oids per request to CDS")
    dataframe_crossmatch = dataframe_spark.selectExpr("oid", "meanra as ra", "meandec as dec")
    dataframe_crossmatch = dataframe_crossmatch.dropDuplicates()
    api = AstroideAPI()
    reference_healpix = api.create_healpix_index(dataframe_crossmatch, _NSIDE_RES, "ra", "dec")
    reference_healpix = reference_healpix.sort('ipix')
    dataframe_crossmatch = reference_healpix.toPandas()



    list_dataframes = pandas_dataframe_chunks(dataframe_crossmatch, _BATCH_SIZE)
    dataframe_crossmatch_chunks = pandas_dataframe_chunks(dataframe_spark.toPandas(), _BATCH_SIZE)
    xmatch_parameters = generate_parameters_request_xmatch()
    xmatches_query = execute_xmatch_requests(list_dataframes, xmatch_parameters, dataframe_crossmatch_chunks) 

    xmatches = xmatches_query[0]
    print(f'Number of oids matches with CDS (Allwise): {len(xmatches)}')
    processed_batch = xmatches_query[1]
    unprocessed_batch = xmatches_query[2]
    xmatches_spark = spark.createDataFrame(xmatches).withColumnRenamed("oid_in", "oid").drop("ra_in").drop("dec_in").drop("col1")
    xmatches_spark = xmatches_spark.repartition('oid')
    dataframe_spark = spark.createDataFrame(processed_batch)
    dataframe_spark = dataframe_spark.repartition('oid')
    #! It's slower doing this method but also it's more protected against failures in execution
    if len(unprocessed_batch) > 0:
        unprocessed_batch = spark.createDataFrame(unprocessed_batch)
        unprocessed_batch = unprocessed_batch.repartition('oid')
    else:
        unprocessed_batch = None
    xmatch_step = dataframe_spark.join(xmatches_spark, on="oid", how="left")

    # Add the W1-W2, W2-W3 columns
    xmatch_step = xmatch_step.withColumn("W1-W2", xmatch_step.W1mag - xmatch_step.W2mag)\
                                               .withColumn("W2-W3", xmatch_step.W2mag - xmatch_step.W3mag)
    # Expand all the detections rows
    xmatch_step = xmatch_step.withColumn("expanded_columns", F.arrays_zip("aid", "ra", "dec", "forced", "sgscore1", "distpsnr1", "sgmag1", "srmag1"))\
       .withColumn("expanded_columns", F.explode("expanded_columns"))\
       .select('oid', 'meanra', 'meandec', 'angDist', 'AllWISE', 'RAJ2000', 'DEJ2000', 'eeMaj', 
               'eeMin', 'eePA', 'W1mag', 'W2mag', 'W3mag', 'W4mag', 'Jmag', 'Hmag', 'Kmag', 
               'e_W1mag', 'e_W2mag', 'e_W3mag', 'e_W4mag', 'e_Jmag', 'e_Hmag', 'e_Kmag', 'ID', 
               'ccf', 'ex', 'var', 'qph', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'd2M', 'W1-W2', 'W2-W3',
               
                F.col("expanded_columns.aid").alias("aid"),
                F.col("expanded_columns.ra").alias("ra_oid"),
                F.col("expanded_columns.dec").alias("dec_oid"), 
                F.col("expanded_columns.forced").alias("forced"),
                F.col("expanded_columns.sgscore1").alias("sgscore1"),
                F.col("expanded_columns.distpsnr1").alias("distpsnr1"),
                F.col("expanded_columns.sgmag1").alias("sgmag1"),
                F.col("expanded_columns.srmag1").alias("srmag1")
                )
    
    window_spec = Window.partitionBy("oid")
    # Replace the null values of the columns from alert, by using for each oid the data of one alert 
    xmatch_step = xmatch_step.withColumn("sgscore1", first("sgscore1", ignorenulls=True).over(window_spec)) \
                   .withColumn("distpsnr1", first("distpsnr1", ignorenulls=True).over(window_spec)) \
                   .withColumn("sgmag1", first("sgmag1", ignorenulls=True).over(window_spec)) \
                   .withColumn("srmag1", first("srmag1", ignorenulls=True).over(window_spec))
    
    xmatch_step = xmatch_step.withColumn("catid", lit("allwise"))\
                             .withColumn("class_catalog", lit(None).cast('string'))\
                             .withColumn("period", lit(None).cast('string')) 
    
    
    return xmatch_step, unprocessed_batch
