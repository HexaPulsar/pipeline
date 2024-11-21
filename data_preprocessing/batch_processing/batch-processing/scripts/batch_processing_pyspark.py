import time

from batch_processing.pipeline import (correction, create_dataframes,
                                       lightcurve, magstats, prv_candidates,
                                       sorting_hat, xmatch)
from batch_processing.pipeline.spark_init.pyspark_configs import *

start_time = time.time()

## First, we run the sorting hat in all of the avro parquets in the folder of avro parquets

df_sorting_hat = sorting_hat.run_sorting_hat_step('60292', spark)
print('Completed sorting hat step...')
print(f"Sorting hat execution time: {time.time() - start_time}")

df_prv_candidates = prv_candidates.extract_detections_and_non_detections_dataframe_reparsed(df_sorting_hat)
del df_sorting_hat
print(f"Prv_candidates execution time: {time.time() - start_time}")
print('Completed prv candidates step...')


non_detections_frame = create_dataframes.create_non_detections_frame(df_prv_candidates)
non_detections_frame.write.parquet("output_batch/non_detections/non_detections_parquet.snappy.parquet", compression = 'snappy', mode='overwrite')

detections_frame = create_dataframes.create_detections_frame(df_prv_candidates)
detections_frame.write.parquet("output_batch/detections/detections_parquet.snappy.parquet", compression = 'snappy', mode='overwrite')


forced_photometries_frame = create_dataframes.create_forced_photometries_frame(df_prv_candidates)
forced_photometries_frame.write.parquet("output_batch/forced_photometries/forced_photometries_parquet.snappy.parquet", compression = 'snappy', mode='overwrite')



non_detections_frame = spark.read.format("parquet").option("recursiveFileLookup", "true").load("output_batch/non_detections")
print('Completed non detections frame...')

detections_frame = spark.read.format("parquet").option("recursiveFileLookup", "true").load("output_batch/detections")
print('Completed detections frame...')

forced_photometries_frame = spark.read.format("parquet").option("recursiveFileLookup", "true").load("output_batch/forced_photometries")

print('Total number of detections: ', detections_frame.count())
print(' Total number of forced photometries: ', forced_photometries_frame.count())
print(' Total number of non detections: ', non_detections_frame.count())
print('Total number of detections summed: ', detections_frame.count() + forced_photometries_frame.count())


#del df_prv_candidates
print('Completed forced photometries frame...')
print(f"3 Frames separation execution time: {time.time() - start_time}")
print('Completed separation in 3 dataframes step, including writing results to parquet...')

dataframe_lightcurve = lightcurve.execute_light_curve(detections_frame, non_detections_frame, forced_photometries_frame)


del detections_frame, non_detections_frame, forced_photometries_frame
print('Completed light curve step including write time...') 
print(f"Light curve execution time: {time.time() - start_time}")


correction_output = correction.produce_correction(dataframe_lightcurve)
del dataframe_lightcurve
correction_dataframe = correction_output[0]

correction_dataframe.write.parquet("output_batch/corrections/corrections_parquet.snappy.parquet", compression = 'snappy', mode='overwrite')
del correction_dataframe



corrected_xmatch_keys =  ["oid",
                          "candid",
                          "pid",
                          "ra",
                          "dec",
                          "mjd",
                          "mag_corr",
                          "e_mag_corr_ext",
                          "mag",
                          "e_mag",
                          "fid",
                          "isdiffpos"]

forced_photometries_corrected = correction_output[1]
forced_photometries_corrected.write.parquet("output_batch/forced_photometries_corrected/", compression = 'snappy', mode='overwrite')
forced_phots_xmatch = forced_photometries_corrected.select(*corrected_xmatch_keys)
forced_phots_xmatch.write.parquet("output_batch/lightcurves_xmatch/forced_phots_xmatch/", compression = 'snappy', mode='overwrite')
del forced_photometries_corrected, forced_phots_xmatch

detections_corrected = correction_output[2]
detections_corrected2 = detections_corrected.withColumnRenamed('mag', 'magpsf')\
                                           .withColumnRenamed('e_mag', 'sigmapsf')\
                                           .withColumnRenamed('mag_corr', 'magpsf_corr')\
                                           .withColumnRenamed('e_mag_corr', 'sigmapsf_corr')\
                                           .withColumnRenamed('e_mag_corr_ext', 'sigmapsf_corr_ext')
                                           

detections_corrected2.write.parquet("output_batch/detections_corrected/", compression = 'snappy', mode='overwrite')
detections_xmatch = detections_corrected.select(*corrected_xmatch_keys)
detections_xmatch.write.parquet("output_batch/lightcurves_xmatch/detections_xmatch/", compression = 'snappy', mode='overwrite')
del detections_corrected, detections_xmatch

correction_dataframe = spark.read.format("parquet").option("recursiveFileLookup", "true").load("output_batch/corrections")
print('Completed correction dataframe...')
print(f"Correction step execution time: {time.time() - start_time}")

object_stats, magstats_stats = magstats.execute_magstats_step(correction_dataframe)
print('conteo de magstats es ', magstats_stats.count())
print('conteo de objstats es ', object_stats.count())


object_stats.write.parquet("output_batch/objectstats/objectstats/objstats_parquet.snappy.parquet", compression = 'snappy', mode='overwrite')#
magstats_stats.write.parquet("output_batch/magstats/magstats_parquet.snappy.parquet", compression = 'snappy', mode='overwrite')#
print('Total time including magstats write: ' + str(time.time() - start_time))


correction_dataframe_xmatch_data = spark.read.format("parquet").option("recursiveFileLookup", "true").load("output_batch/corrections")\
                            .select('oid', 'meanra', 'meandec', 'detections.aid', 'detections.ra', 'detections.dec', 'detections.forced', 'detections.sgscore1',
                                'detections.distpsnr1', 'detections.sgmag1', 'detections.srmag1')

xmatch_step_output = xmatch.execute_batchsizes(correction_dataframe_xmatch_data)
xmatch_step = xmatch_step_output[0]
unprocessed = xmatch_step_output[1]

print("Number of xmatch result rows: ", xmatch_step.count())

if unprocessed!=None:
    print("Wrote to disk: ", unprocessed.count(), " unprocessed light curves")
    unprocessed.write.parquet("output_batch/unprocessed/unprocessed_parquet.snappy.parquet", compression = 'snappy', mode='overwrite')


xmatch_step = xmatch_step.repartition("oid")
xmatch_step = xmatch_step.withColumnRenamed('angDist', 'dist')\
                                       .withColumnRenamed('AllWISE', 'oid_catalog')



xmatch_step.write.parquet("output_batch/xmatch/xmatch_parquet.snappy.parquet", compression = 'snappy', mode='overwrite')

print('Total time including xmatch write: ' + str(time.time() - start_time))
input("Press enter to terminate")
spark.stop()
exit()
