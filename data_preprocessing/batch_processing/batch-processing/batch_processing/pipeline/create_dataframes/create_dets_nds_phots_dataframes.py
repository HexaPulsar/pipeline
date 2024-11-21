from ..spark_init.pyspark_configs import *


# Create the dataframe of the unique non_detections
def create_non_detections_frame(df):
    non_detections_frame = df.select(F.explode("non_detections").alias("exploded_data"))
    non_detections_frame = non_detections_frame.repartition(col('oid'))
    unique_non_detections = non_detections_frame.select("exploded_data.*")
    window_spec = Window.partitionBy(col("oid"), col("mjd")).orderBy(F.col("mjd_alert").desc(), F.col("mjd").desc())
    unique_non_detections = unique_non_detections.withColumn("rank", row_number().over(window_spec))
    unique_non_detections = unique_non_detections.filter(col("rank") == 1).drop(col("rank")).drop(col("mjd_alert"))
    return unique_non_detections


def create_forced_photometries_frame(df):
    fp_frame = df.select(F.explode("forced_photometries").alias("exploded_data"))
    fp_frame = fp_frame.select("exploded_data.*")
    #! Create is first corrected here, so the deduplicated data identifies correctly the first corrected photometry
    #! Any changes to the definition of the first phot corrected in the correction step will be reflected here
    window_spec = Window.partitionBy(col("oid"), col("fid")).orderBy(col("mjd_alert"), col("mjd"))
    fp_frame = fp_frame.withColumn("row_number", row_number().over(window_spec))
    fp_frame = fp_frame.withColumn(
        "first_corrected_mjd",
        when(col("row_number") == 1, True).otherwise(False)
    ).drop(col("row_number"))

    windowSpec = Window.partitionBy(col("oid"), col("fid"))
    
    fp_frame = fp_frame.withColumn("is_first_row", 
                                when((col("first_corrected_mjd") & (col("distnr") < 1.4)), lit(True))
                                .otherwise(lit(False)))
    
    fp_frame = fp_frame.withColumn("is_first_corrected",
        first(col("is_first_row"), ignorenulls=True).over(windowSpec)
    )

    fp_frame = fp_frame.drop("is_first_row", "first_corrected_mjd")

    #! Keeping latest forced photometry for each candid/oid pair. TO do so, we added alert mjd to each forced photometry
    window_spec = Window.partitionBy(col("oid"), col("candid")).orderBy(F.col("mjd_alert").desc(), F.col("mjd").desc())
    df_with_rank = fp_frame.withColumn("rank", row_number().over(window_spec))
    unique_fp = df_with_rank.filter(col("rank") == 1).drop(col("rank"))
    # Drop column alert mjd after being used for its purpose of deduplicating orderly 
    return unique_fp


# Now we want to get the detections dataframe. 
def create_detections_frame(df):
    # We'll first order the dataframe by MJD. This is to retain the newest alerts
    df = df.sort(col('mjd'), ascending = False) 
    alert_df = df.drop(col('detections'), col('non_detections'), col('forced_photometries'))

    # Adding the min mjd value of all detections oid fid pairs, to be used in the correction step at first corrected
    window_spec = Window.partitionBy(col("oid"), col("fid")).orderBy(col("mjd"))
    alert_df = alert_df.withColumn("row_number", row_number().over(window_spec))
    alert_df = alert_df.withColumn(
        "first_corrected_mjd",
        when(col("row_number") == 1, True).otherwise(False)
    ).drop(col("row_number"))    
    
    
    windowSpec = Window.partitionBy(col("oid"), col("fid"))
    
    alert_df = alert_df.withColumn("is_first_row", 
                                when((col("first_corrected_mjd") & (col("distnr") < 1.4)), lit(True))
                                .otherwise(lit(False)))
    
    alert_df = alert_df.withColumn("is_first_corrected",
        first(col("is_first_row"), ignorenulls=True).over(windowSpec)
    )

    alert_df = alert_df.drop(col("is_first_row"), col("first_corrected_mjd"))
    # Use rank to keep the latest alert, instead of just deduplicating
    # Run a second deduplication (just candid/oid in case some weird alerts at the same time with same data but different candid event happens)
    windowSpec = Window.partitionBy(col("oid"), col("candid")).orderBy(F.col("mjd_alert").desc(), F.col("mjd").desc())
    alert_df = alert_df.withColumn("rank", rank().over(windowSpec))
    alert_df = alert_df.filter(col("rank") == 1).drop("rank")
    alert_df = alert_df.drop_duplicates(["oid", "candid"])
    
    # Now we create a detections dataframe where we expand the detections and its extra fields. Same process we did with the alerts
    detections_frame = df
    exploded_detections = detections_frame.select(explode(detections_frame.detections).alias("exploded_data"))
    unnested_detections = exploded_detections.select("exploded_data.*")

    unique_detections = unnested_detections.sort(col('mjd'), ascending = False)

    window_spec = Window.partitionBy(col("oid"), col("fid")).orderBy(col("mjd"))
    unique_detections = unique_detections.withColumn("row_number", row_number().over(window_spec))
    unique_detections = unique_detections.withColumn(
        "first_corrected_mjd",
        when(col("row_number") == 1, True).otherwise(False)
    ).drop(col("row_number"))    

    windowSpec = Window.partitionBy(col("oid"), col("fid"))
    
    unique_detections = unique_detections.withColumn("is_first_row", 
                                when((col("first_corrected_mjd") & (col("distnr") < 1.4)), lit(True))
                                .otherwise(lit(False)))
    unique_detections = unique_detections.withColumn("is_first_corrected",
        first(col("is_first_row"), ignorenulls=True).over(windowSpec)
    )
    unique_detections = unique_detections.drop(col("is_first_row"), col("first_corrected_mjd"))

    #! Use rank to keep the latest detection, instead of just deduplicating
    windowSpec = Window.partitionBy(col("oid"), col("candid")).orderBy(F.col("mjd_alert").desc(), F.col("mjd").desc())
    unique_detections = unique_detections.withColumn("rank", rank().over(windowSpec))
    unique_detections = unique_detections.filter(col("rank") == 1).drop(col("rank"))

    #! Second deduplication to ensure no oddities happen i.e diferent parent candid, exact same data, including alert time
    unique_detections = unique_detections.drop_duplicates(['oid', 'candid'])

    # We rename the columns to identify after the join from which dataframe they come from. We rename oid and candid so they have the same name (it can be done without this step, but there were originally problems not doing it)

    unique_detections = unique_detections.selectExpr([f"{col} as {col}_detection" for col in unique_detections.columns])
    alert_df = alert_df.selectExpr([f"{col} as {col}_alert" for col in alert_df.columns])
    
    # Directly join the alert data with the detections dat
    detections_dataframe = alert_df.join(
    unique_detections,
    on=[col('oid_alert') == col('oid_detection'), col('candid_alert') == col('candid_detection')],    how='full'
    )

    # We update the alert data with detection data in the cases where it is required
    # Mjd alert for parent candid because the logic is slightly different (an alert has parent candid null)
    detections_dataframe = detections_dataframe.select(
    when(col('candid_alert').isNotNull(), col('candid_alert')).otherwise(col('candid_detection')).alias("candid"),
    when(col('oid_alert').isNotNull(), col('oid_alert')).otherwise(col('oid_detection')).alias("oid"),
    when(col('aid_alert').isNotNull(), col('aid_alert')).otherwise(col('aid_detection')).alias("aid"),
    when(col('dec_alert').isNotNull(), col('dec_alert')).otherwise(col('dec_detection')).alias("dec"),
    when(col('e_dec_alert').isNotNull(), col('e_dec_alert')).otherwise(col('e_dec_detection')).alias("e_dec"),
    when(col('e_mag_alert').isNotNull(), col('e_mag_alert')).otherwise(col('e_mag_detection')).alias("e_mag"),
    when(col('e_ra_alert').isNotNull(), col('e_ra_alert')).otherwise(col('e_ra_detection')).alias("e_ra"),
    when(col('fid_alert').isNotNull(), col('fid_alert')).otherwise(col('fid_detection')).alias("fid"),
    when(col('forced_alert').isNotNull(), col('forced_alert')).otherwise(col('forced_detection')).alias("forced"),
    when(col('has_stamp_alert').isNotNull(), col('has_stamp_alert')).otherwise(col('has_stamp_detection')).alias("has_stamp"),
    when(col('isdiffpos_alert').isNotNull(), col('isdiffpos_alert')).otherwise(col('isdiffpos_detection')).alias("isdiffpos"),
    when(col('mag_alert').isNotNull(), col('mag_alert')).otherwise(col('mag_detection')).alias("mag"),
    when(col('mjd_alert').isNotNull(), col('mjd_alert')).otherwise(col('mjd_detection')).alias("mjd"),
    when(col('mjd_alert').isNotNull(), col('parent_candid_alert')).otherwise(col('parent_candid_detection')).alias("parent_candid"),
    when(col('pid_alert').isNotNull(), col('pid_alert')).otherwise(col('pid_detection')).alias("pid"),
    when(col('ra_alert').isNotNull(), col('ra_alert')).otherwise(col('ra_detection')).alias("ra"),
    when(col('sid_alert').isNotNull(), col('sid_alert')).otherwise(col('sid_detection')).alias("sid"),
    when(col('tid_alert').isNotNull(), col('tid_alert')).otherwise(col('tid_detection')).alias("tid"),
    when(col('parsed_fid_alert').isNotNull(), col('parsed_fid_alert')).otherwise(col('parsed_fid_detection')).alias("parsed_fid"),
    when(col('unparsed_isdiffpos_alert').isNotNull(), col('unparsed_isdiffpos_alert')).otherwise(col('unparsed_isdiffpos_detection')).alias("unparsed_isdiffpos"),
    when(col('unparsed_jd_alert').isNotNull(), col('unparsed_jd_alert')).otherwise(col('unparsed_jd_detection')).alias("unparsed_jd"),
    when(col('aimage_alert').isNotNull(), col('aimage_alert')).otherwise(col('aimage_detection')).alias("aimage"),
    when(col('aimagerat_alert').isNotNull(), col('aimagerat_alert')).otherwise(col('aimagerat_detection')).alias("aimagerat"),
    when(col('bimage_alert').isNotNull(), col('bimage_alert')).otherwise(col('bimage_detection')).alias("bimage"),
    when(col('bimagerat_alert').isNotNull(), col('bimagerat_alert')).otherwise(col('bimagerat_detection')).alias("bimagerat"),
    when(col('chinr_alert').isNotNull(), col('chinr_alert')).otherwise(col('chinr_detection')).alias("chinr"),
    when(col('chipsf_alert').isNotNull(), col('chipsf_alert')).otherwise(col('chipsf_detection')).alias("chipsf"),
    when(col('classtar_alert').isNotNull(), col('classtar_alert')).otherwise(col('classtar_detection')).alias("classtar"),
    when(col('clrcoeff_alert').isNotNull(), col('clrcoeff_alert')).otherwise(col('clrcoeff_detection')).alias("clrcoeff"),
    when(col('clrcounc_alert').isNotNull(), col('clrcounc_alert')).otherwise(col('clrcounc_detection')).alias("clrcounc"),
    col("clrmed_alert").alias("clrmed"),
    col("clrrms_alert").alias("clrrms"),
    when(col('decnr_alert').isNotNull(), col('decnr_alert')).otherwise(col('decnr_detection')).alias("decnr"),
    when(col('diffmaglim_alert').isNotNull(), col('diffmaglim_alert')).otherwise(col('diffmaglim_detection')).alias("diffmaglim"),
    when(col('distnr_alert').isNotNull(), col('distnr_alert')).otherwise(col('distnr_detection')).alias("distnr"),
    col("distpsnr1_alert").alias("distpsnr1"),
    col("distpsnr2_alert").alias("distpsnr2"),
    col("distpsnr3_alert").alias("distpsnr3"),
    col("drb_alert").alias("drb"),
    col("drbversion_alert").alias("drbversion"),
    col("dsdiff_alert").alias("dsdiff"),
    col("dsnrms_alert").alias("dsnrms"),
    when(col('elong_alert').isNotNull(), col('elong_alert')).otherwise(col('elong_detection')).alias("elong"),
    col("exptime_alert").alias("exptime"),
    when(col('field_alert').isNotNull(), col('field_alert')).otherwise(col('field_detection')).alias("field"),
    when(col('fwhm_alert').isNotNull(), col('fwhm_alert')).otherwise(col('fwhm_detection')).alias("fwhm"),
    col("jdendhist_alert").alias("jdendhist"),
    col("jdendref_alert").alias("jdendref"),
    col("jdstarthist_alert").alias("jdstarthist"),
    col("jdstartref_alert").alias("jdstartref"),
    when(col('magap_alert').isNotNull(), col('magap_alert')).otherwise(col('magap_detection')).alias("magap"),
    when(col('magapbig_alert').isNotNull(), col('magapbig_alert')).otherwise(col('magapbig_detection')).alias("magapbig"),
    when(col('magdiff_alert').isNotNull(), col('magdiff_alert')).otherwise(col('magdiff_detection')).alias("magdiff"),
    when(col('magfromlim_alert').isNotNull(), col('magfromlim_alert')).otherwise(col('magfromlim_detection')).alias("magfromlim"),
    col("maggaia_alert").alias("maggaia"),
    col("maggaiabright_alert").alias("maggaiabright"),
    when(col('magnr_alert').isNotNull(), col('magnr_alert')).otherwise(col('magnr_detection')).alias("magnr"),
    when(col("magzpsci_alert").isNotNull(), col("magzpsci_alert")).otherwise(col("magzpsci_detection")).alias("magzpsci"),
    when(col("magzpscirms_alert").isNotNull(), col("magzpscirms_alert")).otherwise(col("magzpscirms_detection")).alias("magzpscirms"),
    when(col("magzpsciunc_alert").isNotNull(), col("magzpsciunc_alert")).otherwise(col("magzpsciunc_detection")).alias("magzpsciunc"),
    when(col("is_first_corrected_alert").isNotNull(), col("is_first_corrected_alert")).otherwise(col("is_first_corrected_detection")).alias("is_first_corrected"),
    when(col("mindtoedge_alert").isNotNull(), col("mindtoedge_alert")).otherwise(col("mindtoedge_detection")).alias("mindtoedge"),
    when(col("nbad_alert").isNotNull(), col("nbad_alert")).otherwise(col("nbad_detection")).alias("nbad"),
    col("ncovhist_alert").alias("ncovhist"),
    col("ndethist_alert").alias("ndethist"),
    col("neargaia_alert").alias("neargaia"),
    col("neargaiabright_alert").alias("neargaiabright"),
    col("nframesref_alert").alias("nframesref"),
    when(col("nid_alert").isNotNull(), col("nid_alert")).otherwise(col("nid_detection")).alias("nid"),
    col("nmatches_alert").alias("nmatches"),
    col("nmtchps_alert").alias("nmtchps"),
    when(col("nneg_alert").isNotNull(), col("nneg_alert")).otherwise(col("nneg_detection")).alias("nneg"),
    col("objectidps1_alert").alias("objectidps1"),
    col("objectidps2_alert").alias("objectidps2"),
    col("objectidps3_alert").alias("objectidps3"),
    when(col("pdiffimfilename_alert").isNotNull(), col("pdiffimfilename_alert")).otherwise(col("pdiffimfilename_detection")).alias("pdiffimfilename"),
    when(col("programid_alert").isNotNull(), col("programid_alert")).otherwise(col("programid_detection")).alias("programid"),
    when(col("programpi_alert").isNotNull(), col("programpi_alert")).otherwise(col("programpi_detection")).alias("programpi"),
    when(col("ranr_alert").isNotNull(), col("ranr_alert")).otherwise(col("ranr_detection")).alias("ranr"),
    when(col("rb_alert").isNotNull(), col("rb_alert")).otherwise(col("rb_detection")).alias("rb"),
    when(col("rbversion_alert").isNotNull(), col("rbversion_alert")).otherwise(col("rbversion_detection")).alias("rbversion"),
    when(col("rcid_alert").isNotNull(), col("rcid_alert")).otherwise(col("rcid_detection")).alias("rcid"),
    col("rfid_alert").alias("rfid"),
    when(col("scorr_alert").isNotNull(), col("scorr_alert")).otherwise(col("scorr_detection")).alias("scorr"),
    when(col("seeratio_alert").isNotNull(), col("seeratio_alert")).otherwise(col("seeratio_detection")).alias("seeratio"),
    col("sgmag1_alert").alias("sgmag1"),
    col("sgmag2_alert").alias("sgmag2"),
    col("sgmag3_alert").alias("sgmag3"),
    col("sgscore1_alert").alias("sgscore1"),
    col("sgscore2_alert").alias("sgscore2"),
    col("sgscore3_alert").alias("sgscore3"),
    when(col("sharpnr_alert").isNotNull(), col("sharpnr_alert")).otherwise(col("sharpnr_detection")).alias("sharpnr"),
    when(col("sigmagap_alert").isNotNull(), col("sigmagap_alert")).otherwise(col("sigmagap_detection")).alias("sigmagap"),
    when(col("sigmagapbig_alert").isNotNull(), col("sigmagapbig_alert")).otherwise(col("sigmagapbig_detection")).alias("sigmagapbig"),
    when(col("sigmagnr_alert").isNotNull(), col("sigmagnr_alert")).otherwise(col("sigmagnr_detection")).alias("sigmagnr"),
    col("simag1_alert").alias("simag1"),
    col("simag2_alert").alias("simag2"),
    col("simag3_alert").alias("simag3"),
    when(col("sky_alert").isNotNull(), col("sky_alert")).otherwise(col("sky_detection")).alias("sky"),
    col("srmag1_alert").alias("srmag1"),
    col("srmag2_alert").alias("srmag2"),
    col("srmag3_alert").alias("srmag3"),
    when(col("ssdistnr_alert").isNotNull(), col("ssdistnr_alert")).otherwise(col("ssdistnr_detection")).alias("ssdistnr"),
    when(col("ssmagnr_alert").isNotNull(), col("ssmagnr_alert")).otherwise(col("ssmagnr_detection")).alias("ssmagnr"),
    when(col("ssnamenr_alert").isNotNull(), col("ssnamenr_alert")).otherwise(col("ssnamenr_detection")).alias("ssnamenr"),
    col("ssnrms_alert").alias("ssnrms"),
    when(col("sumrat_alert").isNotNull(), col("sumrat_alert")).otherwise(col("sumrat_detection")).alias("sumrat"),
    col("szmag1_alert").alias("szmag1"),
    col("szmag2_alert").alias("szmag2"),
    col("szmag3_alert").alias("szmag3"),
    when(col("tblid_alert").isNotNull(), col("tblid_alert")).otherwise(col("tblid_detection")).alias("tblid"),
    col("tooflag_alert").alias("tooflag"),
    when(col("xpos_alert").isNotNull(), col("xpos_alert")).otherwise(col("xpos_detection")).alias("xpos"),
    when(col("ypos_alert").isNotNull(), col("ypos_alert")).otherwise(col("ypos_detection")).alias("ypos"),
    col("zpclrcov_alert").alias("zpclrcov"),
    col("zpmed_alert").alias("zpmed"))

    # We generate the struct for the extra fields again
    not_extrafields = ['aid','candid','dec', 'distnr','e_dec', 'e_mag', 'e_ra', 'fid', 'forced', 'has_stamp', "distpsnr1", "chinr", "sharpnr", "magnr", "sigmagnr", "sgscore1", 'isdiffpos','mag', 'mjd', 'is_first_corrected',  'oid', 'parent_candid', 'pid', 'ra', 'sid', 'tid', 'parsed_fid', 'unparsed_isdiffpos', 'unparsed_jd']
    df_extrafields = detections_dataframe.drop(*not_extrafields)
    sorted_columns_extra_fields = sorted(df_extrafields.columns)
    df_extrafields = df_extrafields.select(*sorted_columns_extra_fields)
    columns_extra_fields = df_extrafields.columns
    not_extrafields.append("extra_fields") # adding the extra fields to add to the detection dataframe
    detections_dataframe = detections_dataframe.withColumn("extra_fields", struct([col(c) for c in columns_extra_fields])).select(*[not_extrafields])
    sorted_columns_df = sorted(detections_dataframe.columns)
    detections_dataframe = detections_dataframe.select(*sorted_columns_df)
    detections_dataframe = detections_dataframe.repartition(col('oid'))
    return detections_dataframe
