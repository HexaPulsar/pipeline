from ..spark_init.pyspark_configs import *


#! All grouped_values carry a order by mjd
_THRESHOLD_ZTF = 13.2

# Load all parquets from a folder and its nested ones. Used to load light curve dataframes

def separate_dataframe_corr_df(correction_df):
    detections = correction_df.select(explode("detections").alias("exploded_data")).select("exploded_data.*")
    non_detections = correction_df.select(explode("non_detections").alias("exploded_data")).select("exploded_data.*")
    return detections, non_detections

def explode_detections(detections):
    detections = detections.select(explode("detections").alias("exploded_data")).select("exploded_data.*")
    return detections

def explode_non_detections(non_detections):
    non_detections = non_detections.select(explode("non_detections").alias("exploded_data")).select("exploded_data.*")
    return non_detections

def separate_dataframe_cobj(correction_df):
    detections = correction_df.select(explode("detections").alias("exploded_data")).select("exploded_data.*")
    return detections

#! Function to create the "object statistic" will choose only the detections non forced
#Functions to create index for detections, for now we first drop it, then create it again, because last correction step ran had still index as a column, to compare with pipeline results
def drop_dupes_detection(detections):
    detections = detections.repartition("oid")
    return detections.dropDuplicates(["candid", "oid"])


def get_non_forced(corrector_detections):
    return corrector_detections.filter(~col("forced"))

###################################### OBJECT STATS ###################################################### 

def first_mjd(detections):
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid")
    detections = detections.withColumn("first_mjd", F.min("mjd").over(window_spec))
    return detections

def last_mjd(detections):  
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid")
    detections = detections.withColumn("last_mjd", F.max("mjd").over(window_spec))
    return detections

def deltajd(detections):
    detections = detections.repartition("oid")
    detections = first_mjd(detections)
    detections = last_mjd(detections)
    detections = detections.withColumn("deltajd", detections["last_mjd"] - detections["first_mjd"])
    return detections

def calculate_ndets_oid(detections):
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid")
    detections = detections.withColumn("ndet", F.count("*").over(window_spec))
    return detections

def arcsec2dec_era_edec(non_forced):
    non_forced = non_forced.withColumn("e_ra_arcsec", col("e_ra") / 3600.0)
    non_forced = non_forced.withColumn("e_dec_arcsec", col("e_dec") / 3600.0)
    return non_forced

def create_weighted_columns(non_forced):
    non_forced = non_forced.withColumn("weighted_e_ra", 1.0 / (col("e_ra_arcsec") ** 2.0))
    non_forced = non_forced.withColumn("weighted_e_dec", 1.0 / (col("e_dec_arcsec") ** 2.0))
    return non_forced

def correct_coordinates(non_forced):
    non_forced = non_forced.repartition("oid")
    non_forced = arcsec2dec_era_edec(non_forced)
    non_forced = create_weighted_columns(non_forced)
    window_spec = Window.partitionBy("oid")

    non_forced = non_forced.withColumn("weighted_sum_ra", F.sum(F.col("ra") * F.col("weighted_e_ra")).over(window_spec)) \
                           .withColumn("weighted_sum_dec", F.sum(F.col("dec") * F.col("weighted_e_dec")).over(window_spec)) \
                           .withColumn("total_weight_e_ra", F.sum("weighted_e_ra").over(window_spec)) \
                           .withColumn("total_weight_e_dec", F.sum("weighted_e_dec").over(window_spec))
    
    corrected_coords = non_forced.withColumn("meanra", col("weighted_sum_ra") / col("total_weight_e_ra")) \
                                       .withColumn("sigmara", 3600.0 * sqrt(1 / col("total_weight_e_ra"))) \
                                       .withColumn("meandec", col("weighted_sum_dec") / col("total_weight_e_dec")) \
                                       .withColumn("sigmadec", 3600.0 * sqrt(1 / col("total_weight_e_dec"))) 
    corrected_coords = corrected_coords.drop("ra", "dec", "weighted_sum_ra", "weighted_sum_dec", "total_weight_e_ra", "total_weight_e_dec", "e_ra_arcsec", "e_dec_arcsec", "weighted_e_ra", "weighted_e_dec", "weighted_sum_ra", "weighted_sum_dec", "total_weight_e_ra", "total_weight_e_dec", "e_ra", "e_dec") 

    return corrected_coords

#! Assuming that the same mjd is true for all the hist columns (which should be true) can still do for each column but its going to be messier
def create_hist_columns(detections):
    df_with_max_mjd = detections.withColumn("max_mjd", max(when(col("ndethist").isNotNull(), col("mjd"))).over(Window.partitionBy("oid")))
    df_with_max_mjd = df_with_max_mjd.withColumn("recent_ndethist",when(col("mjd") == col("max_mjd"), col("ndethist")))\
                                     .withColumn("recent_ncovhist", when(col("mjd") == col("max_mjd"), col("ncovhist")))\
                                     .withColumn("recent_starthist", when(col("mjd") == col("max_mjd"), col("jdstarthist")- 2400000.5))\
                                     .withColumn("recent_endhist", when(col("mjd") == col("max_mjd"), col("jdendhist")- 2400000.5))
    window_spec_fill = Window.partitionBy("oid").orderBy(col("mjd").desc()).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    detections = df_with_max_mjd.withColumn("ndethist", first("recent_ndethist").over(window_spec_fill))\
                                .withColumn("ncovhist", first("recent_ncovhist").over(window_spec_fill))\
                                .withColumn("mjdstarthist", first("recent_starthist").over(window_spec_fill))\
                                .withColumn("mjdendhist", first("recent_endhist").over(window_spec_fill))
     
    detections = detections.drop("max_mjd", "recent_ndethist", "recent_ncovhist", "recent_starthist", "recent_endhist")
    return detections



#! Por corroborar en ambos casos
def select_corrected_stellar_firstmjd(detections):
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid").orderBy("mjd")
    first_corrected = F.first("corrected").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))
    first_stellar = F.first("stellar").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))
    detections = detections.withColumn("corrected", F.last(first_corrected, ignorenulls=True).over(window_spec)) \
                           .withColumn("stellar", F.last(first_stellar, ignorenulls=True).over(window_spec))    
    return detections

def calculate_diffpos(detections):
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid").orderBy("mjd")
    detections = detections.withColumn("diffpos", F.first("isdiffpos").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)) > 0)
    return detections


#! Multiple null values since jdendref is null in all prv detections and fp hists, but has values in alerts

def calculate_reference_change(detections):
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid").orderBy("mjd")
    detections = detections.withColumn("last_mjdendref", F.last("jdendref").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
    detections = detections.withColumn("reference_change", detections["last_mjdendref"] - 2400000.5 > detections["first_mjd"])
    detections = detections.drop("last_mjdendref")
    return detections


def calculate_hist_columns(detections):
    detections = detections.repartition("oid")
    detections.filter(col('oid')=='ZTF19abzlyqu')
    window_spec = Window.partitionBy("oid").orderBy("mjd")
    detections = detections.withColumn("ndethist", F.last("ndethist").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
    detections = detections.withColumn("ncovhist", F.last("ncovhist").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
    detections = detections.withColumn("mjdstarthist", F.last("jdstarthist").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)) - 2400000.5) 
    detections = detections.withColumn("mjdendref", F.last("jdendref").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)) - 2400000.5)
    detections = detections.drop("jdstarthist", "jdendref")
    return detections

def create_g_r_columns(detections):
    detections = detections.withColumn("g_r_max", lit(None).cast('double'))
    detections = detections.withColumn("g_r_max_corr", lit(None).cast('double'))
    detections = detections.withColumn("g_r_mean", lit(None).cast('double'))
    detections = detections.withColumn("g_r_mean_corr", lit(None).cast('double'))
    return detections

def create_step_id_column(detections, step_id):
    detections = detections.withColumn("step_id_corr", lit(step_id))
    return detections


def calculate_object_stats(detections):
    detections = detections.repartition("oid")
    detections = calculate_ndets_oid(detections)
    detections = deltajd(detections)
    detections = correct_coordinates(detections)
    detections = select_corrected_stellar_firstmjd(detections)
    detections = calculate_reference_change(detections)
    detections = calculate_diffpos(detections)
    detections = create_hist_columns(detections)
    detections = detections.dropDuplicates(["oid", "sid"])
    detections = create_g_r_columns(detections)
    step_id = "batch_processing_DD/MM/YYYY"  #! Temporary here (and name!
    detections = create_step_id_column(detections, step_id)
    detections = detections.select("oid", "ndethist", "ncovhist", "mjdstarthist", "mjdendhist", "ndet", "first_mjd", "last_mjd", "deltajd", "g_r_max", "g_r_max_corr", "g_r_mean", "g_r_mean_corr", "meanra", "meandec", "sigmara", "sigmadec", "sid", "fid", "stellar", "corrected", "diffpos", "reference_change", "step_id_corr")
    return detections


#################################### MAGSTATS #########################################

def calculate_ndets_magstats(detections):
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid", "sid", "fid")
    detections = detections.withColumn("ndet", F.count("*").over(window_spec))
    return detections

def firstmagstats(detections):
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid", "sid", "fid")
    detections = detections.withColumn("first_mjd", F.min("mjd").over(window_spec))
    return detections

def lastmagstats(detections):
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid", "sid", "fid")
    detections = detections.withColumn("last_mjd", F.max("mjd").over(window_spec))
    return detections

def calculate_mags_magstats(detections):
    detections = detections.repartition("oid")
    window = Window.partitionBy("oid", "fid", "sid")
    detections_with_stats = detections.withColumn("magmax", F.max("mag").over(window)) \
                                     .withColumn("magmin", F.min("mag").over(window)) \
                                     .withColumn("magmean", F.avg("mag").over(window)) \
                                     .withColumn("magsigma", F.stddev_pop("mag").over(window))

    # Recalculate median but this time we are implementing our own median since percentile approx is not working as expected with odd values
    window_spec_ordered = Window.partitionBy("oid", "fid", "sid").orderBy("mag")
    count_window = Window.partitionBy("oid", "fid", "sid")
    detections_with_stats = detections_with_stats.withColumn("rank", F.row_number().over(window_spec_ordered))
    detections_with_stats = detections_with_stats.withColumn("count", F.count("mag").over(count_window))
    
    detections_with_stats = detections_with_stats.withColumn(
    "median",
    F.when(
        F.col("count") % 2 == 1,  # Odd count
        F.when(F.col("rank") == (F.col("count") + 1) / 2, F.col("mag"))
    ).otherwise(  # Even count
        F.when(
            F.col("rank").isin(F.col("count") / 2, (F.col("count") / 2) + 1),  # Two middle values
            F.col("mag")
        )))

    detections_with_stats = detections_with_stats.withColumn(
    "median_calculated",
    F.when(
        F.col("count") % 2 == 0,  # Even count
        (F.first("median", ignorenulls=True).over(window) + F.last("median", ignorenulls=True).over(window)) / 2
    ).otherwise(
        F.first("median", ignorenulls=True).over(window)
    )
    )

    detections_with_stats = detections_with_stats.drop("rank", "count", "median").withColumnRenamed("median_calculated", "magmedian")
    return detections_with_stats

def calculate_first_last_mag(detections):
    window = Window.partitionBy("oid", "sid", "fid").orderBy("mjd")
    detections = detections.withColumn("magfirst", F.first("mag").over(window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
    detections = detections.withColumn("maglast", F.last("mag").over(window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
    return detections

def calculate_stellar_corrected_magstats(detections):
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid", "sid", "fid").orderBy("mjd")
    first_corrected = F.first("corrected").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))
    first_stellar = F.first("stellar").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))
    detections = detections.withColumn("corrected", F.last(first_corrected, ignorenulls=True).over(window_spec)) \
                           .withColumn("stellar", F.last(first_stellar, ignorenulls=True).over(window_spec))    
    return detections


def calculate_mags_corrected_magstats(detections):
    detections = detections.repartition("oid")
    window = Window.partitionBy("oid", "fid", "sid")
    detections_with_stats = detections.withColumn("magmax_corr", F.max("mag_corr").over(window)) \
                                     .withColumn("magmin_corr", F.min("mag_corr").over(window)) \
                                     .withColumn("magmean_corr", F.avg("mag_corr").over(window)) \
                                     .withColumn("magsigma_corr", F.stddev_pop("mag_corr").over(window))
    
    # Recalculate median but this time we are implementing our own median since percentile approx is not working as expected with odd values
    window_spec_ordered = Window.partitionBy("oid", "fid", "sid").orderBy("mag")
    count_window = Window.partitionBy("oid", "fid", "sid")
    detections_with_stats = detections_with_stats.withColumn("rank", F.row_number().over(window_spec_ordered))
    detections_with_stats = detections_with_stats.withColumn("count", F.count("mag_corr").over(count_window))
    
    detections_with_stats = detections_with_stats.withColumn(
    "median",
    F.when(
        F.col("count") % 2 == 1,  # Odd count
        F.when(F.col("rank") == (F.col("count") + 1) / 2, F.col("mag_corr"))
    ).otherwise(  # Even count
        F.when(
            F.col("rank").isin(F.col("count") / 2, (F.col("count") / 2) + 1),  # Two middle values
            F.col("mag_corr")
        )))

    detections_with_stats = detections_with_stats.withColumn(
    "median_calculated",
    F.when(
        F.col("count") % 2 == 0,  # Even count
        (F.first("median", ignorenulls=True).over(window) + F.last("median", ignorenulls=True).over(window)) / 2
    ).otherwise(
        F.first("median", ignorenulls=True).over(window)
    )
    )

    detections_with_stats = detections_with_stats.drop("rank", "count", "median").withColumnRenamed("median_calculated", "magmedian_corr")
    return detections_with_stats

def calculate_first_last_mag_corr(detections):
    detections = detections.repartition("oid")
    window = Window.partitionBy("oid", "fid", "sid")
    detections = detections.withColumn("magfirst_corr", F.first("mag_corr").over(window))
    detections = detections.withColumn("maglast_corr", F.last("mag_corr").over(window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
    return detections


def calculate_dubious_magstats(detections):
    detections = detections.withColumn("dubious_numeric", F.col("dubious").cast("int"))
    window = Window.partitionBy("oid", "fid", "sid")
    detections = detections.withColumn("ndubious", F.sum("dubious_numeric").over(window))
    detections = detections.drop("dubious_numeric", "dubious")    
    return detections

def calculate_saturation_rate(detections):
    detections = detections.withColumn("corrected_numeric", F.when(F.col("corrected"), 1).otherwise(0))
    window_spec = Window.partitionBy("oid", "sid", "fid")
    detections = detections.withColumn("total_saturation", F.sum("corrected_numeric").over(window_spec))
    detections = detections.withColumn("sat_count_row", F.when(F.col("mag_corr") < _THRESHOLD_ZTF, 1).otherwise(0))
    detections = detections.withColumn("sat_count", F.sum("sat_count_row").over(window_spec))
    detections = detections.withColumn("total_saturation", F.sum("corrected_numeric").over(window_spec)) \
                           .withColumn("saturation_rate", 
                                       F.when(F.col("total_saturation") != 0, 
                                              F.col("sat_count") / F.col("total_saturation")).otherwise(None))
    detections = detections.drop("corrected_numeric", "sat_count_row", "total_saturation", "sat_count")
    return detections

def calculate_dmdt(detections, non_detections):
    dt_min = 0.5   
    detections = detections.repartition("oid")
    window_spec = Window.partitionBy("oid", "sid", "fid").orderBy("mjd")
    detections = detections.withColumn("min_mjd", F.min("mjd").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
    detections = detections.withColumn("is_first_detection", F.col("mjd") == F.col("min_mjd"))
    detections = detections.drop("min_mjd")
    detections = detections.filter(col("is_first_detection")==True)
    non_detections = non_detections.repartition("oid")
    non_detections_dmdt = (
        non_detections.sort("mjd")
        .select("oid", "sid", "fid", col("mjd").alias("mjdnd"), "diffmaglim"))
    joined_df = detections.join(non_detections_dmdt, on=["oid", "sid", "fid"], how="left")

    joined_df = joined_df.withColumn("dt_first", col("mjd") - col("mjdnd"))
    joined_df = joined_df.withColumn("dm_first", col("mag") - col("diffmaglim"))
    joined_df = joined_df.withColumn("sigmadm_first", col("e_mag") - col("diffmaglim"))
    joined_df = joined_df.withColumn("dmdt_first", (col("mag") + col("e_mag") - col("diffmaglim")) / col("dt_first"))
    condition = (col("dt_first") > dt_min) | (isnull(col("dt_first")))
    joined_df = joined_df.withColumn("dmdt_first", when((col("dt_first") > dt_min) | isnull(col("dt_first")), col("dmdt_first")).otherwise(col("dmdt_first")))\
                         .withColumn("dt_first", when((col("dt_first") > dt_min) | isnull(col("dt_first")), col("dt_first")).otherwise(col("dt_first")))\
                         .withColumn("dm_first", when((col("dt_first") > dt_min) | isnull(col("dt_first")), col("dm_first")).otherwise(col("dm_first")))\
                         .withColumn("sigmadm_first", when((col("dt_first") > dt_min) | isnull(col("dt_first")), col("sigmadm_first")).otherwise(col("sigmadm_first")))

    joined_df = joined_df.repartition('oid')
    joined_df = joined_df.sort("dmdt_first").dropDuplicates(["oid", "sid", "fid"])
    drop_columns = ['e_mag', 'dubious', 'mjdnd', 'is_first_detection', 'mjd', 'mag_corr', 'diffmaglim','dec','mag','ra']
    joined_df = joined_df.drop(*drop_columns)
    return joined_df
    

def calculate_magstats(detections, non_detections):
    # Calculate various statistics
    detections = calculate_ndets_magstats(detections)
    detections = firstmagstats(detections)
    detections = lastmagstats(detections)
    detections = calculate_mags_magstats(detections)    
    detections = calculate_first_last_mag(detections) 
    detections = calculate_stellar_corrected_magstats(detections)
    detections = calculate_mags_corrected_magstats(detections)
    detections = calculate_first_last_mag_corr(detections)
    detections = calculate_dubious_magstats(detections)
    detections = calculate_saturation_rate(detections)
    detections = calculate_dmdt(detections, non_detections)

    step_id = "batch_processing_DD/MM/YYYY"  #! Temporary here (and name!
    detections = create_step_id_column(detections, step_id)
    return detections

def execute_magstats_step(correction_df):
    detections = correction_df.select('detections')
    non_detections =correction_df.select(col("non_detections"))
    detections = explode_detections(detections)
    # Columns to extract for magstats. Some of them are not necessary for magstats. Hist columns are not used in magstats, while others are not necessary for objectstats. 
    # TODO: refactor so we extract on read only the necessary data, using this list on read of correction (reduce data transfer)
    columns_keep_detections = ["oid", "candid", "forced", "mjd", "e_ra", "e_dec", "ra", "dec", "sid", "fid", "stellar", "corrected", "mag", "mag_corr", "dubious", "e_mag", "isdiffpos", "extra_fields.jdstarthist", "extra_fields.jdendref", "extra_fields.ndethist", "extra_fields.ncovhist", "extra_fields.jdendhist"] 

    detections = detections.select(*columns_keep_detections)
    detections = get_non_forced(detections)
    detections = detections.drop("forced")
    detections = drop_dupes_detection(detections)
    detections = detections.drop("candid")
    detections_objectstats = detections.drop(*["mag", "mag_corr", "dubious", "e_mag"])
    objectstats = calculate_object_stats(detections_objectstats)

    non_detections = explode_non_detections(non_detections)
    columns_keep_non_detections = ["oid", "sid", "fid", "mjd", "diffmaglim"]
    non_detections = non_detections.select(*columns_keep_non_detections)
    detections_magstats = detections.drop('jdendref', 'ndethist', 'ncovhist', 'jdstarthist', 'jdendhist')
    magstats = calculate_magstats(detections_magstats, non_detections)

    return objectstats, magstats
