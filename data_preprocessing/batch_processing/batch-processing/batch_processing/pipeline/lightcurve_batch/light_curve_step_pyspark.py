from ..spark_init.pyspark_configs import *

# Otherwise remains with the same detections it outputs
SKIP_MJD_FILTER = False



def execute_light_curve(detections, non_detections, forced_photometries):
    all_dets_joined = detections.unionByName(forced_photometries, allowMissingColumns=True)
    all_dets_joined = all_dets_joined.repartition('oid')
    candids = all_dets_joined.groupBy("oid").agg(collect_list("candid").alias("candids"))  
    candids = candids.repartition('oid')  
    window_spec = Window.partitionBy("oid")
    all_dets_joined = all_dets_joined.withColumn("candids", collect_list("candid").over(window_spec))
    non_detections = non_detections.dropDuplicates(['oid', 'mjd', 'fid'])
    if not SKIP_MJD_FILTER:
        windowSpec = Window.partitionBy("oid")
        all_dets_joined = all_dets_joined.withColumn("last_mjds", F.max("mjd").over(windowSpec))
        all_dets_joined = all_dets_joined.filter(all_dets_joined["mjd"] <= all_dets_joined["last_mjds"]).drop('last_mjds')
    oid_detections_df = all_dets_joined.groupby('oid').agg(collect_list(struct(all_dets_joined.columns)).alias('detections'))
    oid_detections_df = oid_detections_df.repartition('oid')
    oid_non_detections_df = non_detections.groupby('oid').agg(collect_list(struct(non_detections.columns)).alias('non_detections'))
    oid_non_detections_df = oid_non_detections_df.repartition('oid')
    output_df = oid_detections_df.join(oid_non_detections_df, on='oid', how='left').join(candids, on='oid', how='left')
 
    return output_df
