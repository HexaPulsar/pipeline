from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, first, abs, lit
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

from .schema_avros import schema_avros


# First we must load all the parquets containing the avros
def load_dataframes(parquet_avro_dir, spark):
    parquetDataFrame = (
        spark.read.format("parquet")
        .option("recursiveFileLookup", "true")
        .schema(schema_avros)
        .load(parquet_avro_dir)
        .select(
            col("objectId").alias("objectId"),
            col("prv_candidates").alias("prv_candidates"),
            col("fp_hists").alias("fp_hists"),
            col("candidate.jd").alias("jd"),
            col("candidate.fid").alias("fid"),
            col("candidate.pid").alias("pid"),
            col("candidate.diffmaglim").alias("diffmaglim"),
            col("candidate.pdiffimfilename").alias("pdiffimfilename"),
            col("candidate.programpi").alias("programpi"),
            col("candidate.programid").alias("programid"),
            col("candidate.candid").alias("candid"),
            col("candidate.isdiffpos").alias("isdiffpos"),
            col("candidate.tblid").alias("tblid"),
            col("candidate.nid").alias("nid"),
            col("candidate.rcid").alias("rcid"),
            col("candidate.field").alias("field"),
            col("candidate.xpos").alias("xpos"),
            col("candidate.ypos").alias("ypos"),
            col("candidate.ra").alias("ra"),
            col("candidate.dec").alias("dec"),
            col("candidate.magpsf").alias("magpsf"),
            col("candidate.sigmapsf").alias("sigmapsf"),
            col("candidate.chipsf").alias("chipsf"),
            col("candidate.magap").alias("magap"),
            col("candidate.sigmagap").alias("sigmagap"),
            col("candidate.distnr").alias("distnr"),
            col("candidate.magnr").alias("magnr"),
            col("candidate.sigmagnr").alias("sigmagnr"),
            col("candidate.chinr").alias("chinr"),
            col("candidate.sharpnr").alias("sharpnr"),
            col("candidate.sky").alias("sky"),
            col("candidate.magdiff").alias("magdiff"),
            col("candidate.fwhm").alias("fwhm"),
            col("candidate.classtar").alias("classtar"),
            col("candidate.mindtoedge").alias("mindtoedge"),
            col("candidate.magfromlim").alias("magfromlim"),
            col("candidate.seeratio").alias("seeratio"),
            col("candidate.aimage").alias("aimage"),
            col("candidate.bimage").alias("bimage"),
            col("candidate.aimagerat").alias("aimagerat"),
            col("candidate.bimagerat").alias("bimagerat"),
            col("candidate.elong").alias("elong"),
            col("candidate.nneg").alias("nneg"),
            col("candidate.nbad").alias("nbad"),
            col("candidate.rb").alias("rb"),
            col("candidate.ssdistnr").alias("ssdistnr"),
            col("candidate.ssmagnr").alias("ssmagnr"),
            col("candidate.ssnamenr").alias("ssnamenr"),
            col("candidate.sumrat").alias("sumrat"),
            col("candidate.magapbig").alias("magapbig"),
            col("candidate.sigmagapbig").alias("sigmagapbig"),
            col("candidate.ranr").alias("ranr"),
            col("candidate.decnr").alias("decnr"),
            col("candidate.sgmag1").alias("sgmag1"),
            col("candidate.srmag1").alias("srmag1"),
            col("candidate.simag1").alias("simag1"),
            col("candidate.szmag1").alias("szmag1"),
            col("candidate.sgscore1").alias("sgscore1"),
            col("candidate.distpsnr1").alias("distpsnr1"),
            col("candidate.objectidps1").alias("objectidps1"),
            col("candidate.objectidps2").alias("objectidps2"),
            col("candidate.sgmag2").alias("sgmag2"),
            col("candidate.srmag2").alias("srmag2"),
            col("candidate.simag2").alias("simag2"),
            col("candidate.szmag2").alias("szmag2"),
            col("candidate.sgscore2").alias("sgscore2"),
            col("candidate.distpsnr2").alias("distpsnr2"),
            col("candidate.objectidps3").alias("objectidps3"),
            col("candidate.sgmag3").alias("sgmag3"),
            col("candidate.srmag3").alias("srmag3"),
            col("candidate.simag3").alias("simag3"),
            col("candidate.szmag3").alias("szmag3"),
            col("candidate.sgscore3").alias("sgscore3"),
            col("candidate.distpsnr3").alias("distpsnr3"),
            col("candidate.nmtchps").alias("nmtchps"),
            col("candidate.rfid").alias("rfid"),
            col("candidate.jdstarthist").alias("jdstarthist"),
            col("candidate.jdendhist").alias("jdendhist"),
            col("candidate.scorr").alias("scorr"),
            col("candidate.tooflag").alias("tooflag"),
            col("candidate.drbversion").alias("drbversion"),
            col("candidate.rbversion").alias("rbversion"),
            col("candidate.ndethist").alias("ndethist"),
            col("candidate.ncovhist").alias("ncovhist"),
            col("candidate.jdstartref").alias("jdstartref"),
            col("candidate.jdendref").alias("jdendref"),
            col("candidate.nframesref").alias("nframesref"),
            col("candidate.dsnrms").alias("dsnrms"),
            col("candidate.ssnrms").alias("ssnrms"),
            col("candidate.dsdiff").alias("dsdiff"),
            col("candidate.magzpsci").alias("magzpsci"),
            col("candidate.magzpsciunc").alias("magzpsciunc"),
            col("candidate.magzpscirms").alias("magzpscirms"),
            col("candidate.nmatches").alias("nmatches"),
            col("candidate.clrcoeff").alias("clrcoeff"),
            col("candidate.clrcounc").alias("clrcounc"),
            col("candidate.zpclrcov").alias("zpclrcov"),
            col("candidate.zpmed").alias("zpmed"),
            col("candidate.clrmed").alias("clrmed"),
            col("candidate.clrrms").alias("clrrms"),
            col("candidate.neargaia").alias("neargaia"),
            col("candidate.neargaiabright").alias("neargaiabright"),
            col("candidate.maggaia").alias("maggaia"),
            col("candidate.maggaiabright").alias("maggaiabright"),
            col("candidate.exptime").alias("exptime"),
            col("candidate.drb").alias("drb"),
        )
    )
    return parquetDataFrame


# We define the structure of the parsing dataframe function
def df_sorting_hat(df):
    df = _parse_ztf_df(df)
    df2 = alerce_id_generator(df)
    df3 = aid_replacer(df2)
    return df3


def _parse_ztf_df(df):
    df = _apply_transformations(df)
    # df = _parse_extrafields(df)
    return df


def _apply_transformations(df):
    df = df.select(
        F.col("objectID").alias("oid"),
        F.col("prv_candidates"),
        F.col("fp_hists"),
        F.col("jd").alias("unparsed_jd"),
        F.col("fid"),
        F.col("pid"),
        F.col("diffmaglim"),
        F.col("pdiffimfilename"),
        F.col("programpi"),
        F.col("programid"),
        F.col("candid").cast(StringType()),
        F.when(F.col("isdiffpos") == "t", 1)
        .when(F.col("isdiffpos") == "1", 1)
        .otherwise(-1)
        .alias("isdiffpos"),
        F.col("tblid"),
        F.col("nid"),
        F.col("rcid"),
        F.col("field"),
        F.col("xpos"),
        F.col("ypos"),
        F.col("ra"),
        F.col("dec"),
        F.col("magpsf").alias("mag"),
        F.col("sigmapsf").alias("e_mag"),
        F.col("chipsf"),
        F.col("magap"),
        F.col("sigmagap"),
        F.col("distnr"),
        F.col("magnr"),
        F.col("sigmagnr"),
        F.col("chinr"),
        F.col("sharpnr"),
        F.col("sky"),
        F.col("magdiff"),
        F.col("fwhm"),
        F.col("classtar"),
        F.col("mindtoedge"),
        F.col("magfromlim"),
        F.col("seeratio"),
        F.col("aimage"),
        F.col("bimage"),
        F.col("aimagerat"),
        F.col("bimagerat"),
        F.col("elong"),
        F.col("nneg"),
        F.col("nbad"),
        F.col("rb"),
        F.col("ssdistnr"),
        F.col("ssmagnr"),
        F.col("ssnamenr"),
        F.col("sumrat"),
        F.col("magapbig"),
        F.col("sigmagapbig"),
        F.col("ranr"),
        F.col("decnr"),
        F.col("sgmag1"),
        F.col("srmag1"),
        F.col("simag1"),
        F.col("szmag1"),
        F.col("sgscore1"),
        F.col("distpsnr1"),
        F.col("objectidps1"),
        F.col("objectidps2"),
        F.col("sgmag2"),
        F.col("srmag2"),
        F.col("simag2"),
        F.col("szmag2"),
        F.col("sgscore2"),
        F.col("distpsnr2"),
        F.col("objectidps3"),
        F.col("sgmag3"),
        F.col("srmag3"),
        F.col("simag3"),
        F.col("szmag3"),
        F.col("sgscore3"),
        F.col("distpsnr3"),
        F.col("nmtchps"),
        F.col("rfid"),
        F.col("jdstarthist"),
        F.col("jdendhist"),
        F.col("scorr"),
        F.col("tooflag"),
        F.col("drbversion"),
        F.col("rbversion"),
        F.col("ndethist"),
        F.col("ncovhist"),
        F.col("jdstartref"),
        F.col("jdendref"),
        F.col("nframesref"),
        F.col("dsnrms"),
        F.col("ssnrms"),
        F.col("dsdiff"),
        F.col("magzpsci"),
        F.col("magzpsciunc"),
        F.col("magzpscirms"),
        F.col("nmatches"),
        F.col("clrcoeff"),
        F.col("clrcounc"),
        F.col("zpclrcov"),
        F.col("zpmed"),
        F.col("clrmed"),
        F.col("clrrms"),
        F.col("neargaia"),
        F.col("neargaiabright"),
        F.col("maggaia"),
        F.col("maggaiabright"),
        F.col("exptime"),
        F.col("drb"),
        F.lit("ZTF").alias("tid"),
        F.lit("ZTF").alias("sid"),
        F.when(F.col("fid") == 1, "g")
        .when(F.col("fid") == 2, "r")
        .otherwise("i")
        .alias("parsed_fid"),
        (F.col("jd") - 2400000.5).alias("mjd"),
        (F.col("jd") - 2400000.5).alias("mjd_alert"),
        F.col("isdiffpos").alias("unparsed_isdiffpos"),
        F.when(F.col("fid") == 1, 0.065)
        .when(F.col("fid") == 2, 0.085)
        .otherwise(0.01)
        .alias("e_dec"),
        F.when(
            F.cos(F.radians(F.col("dec"))) != 0,
            F.col("e_dec") / F.abs(F.cos(F.radians(F.col("dec")))),
        )
        .otherwise(float("nan"))
        .alias("e_ra"),
    )

    return df


#! missing true value of 0.01


############################### SORTING HAT AID GENERATION ##############################################
def alerce_id_generator(df):
    # Fix negative Ra
    df = df.select(
        "*",
        when(col("ra") < 0, col("ra") + 360)
        .when(col("ra") > 360, col("ra") - 360)
        .otherwise(col("ra"))
        .alias("ra_fixed"),
    )

    # Calculate Ra components
    df = df.select(
        "*",
        ((col("ra_fixed") / 15).cast("bigint")).alias("ra_hh"),
        (
            ((col("ra_fixed") / 15) - ((col("ra_fixed") / 15).cast("bigint")))
            * 60
        )
        .cast("bigint")
        .alias("ra_mm"),
        (
            (
                (
                    (col("ra_fixed") / 15)
                    - ((col("ra_fixed") / 15).cast("bigint"))
                )
                * 60
                - (
                    (
                        (
                            (col("ra_fixed") / 15)
                            - ((col("ra_fixed") / 15).cast("bigint"))
                        )
                        * 60
                    ).cast("bigint")
                )
            )
            * 60
        )
        .cast("bigint")
        .alias("ra_ss"),
        (
            (
                (
                    (
                        (col("ra_fixed") / 15)
                        - ((col("ra_fixed") / 15).cast("bigint"))
                    )
                    * 60
                    - (
                        (
                            (
                                (col("ra_fixed") / 15)
                                - ((col("ra_fixed") / 15).cast("bigint"))
                            )
                            * 60
                        ).cast("bigint")
                    )
                )
                * 60
                - (
                    (
                        (
                            (
                                (col("ra_fixed") / 15)
                                - ((col("ra_fixed") / 15).cast("bigint"))
                            )
                            * 60
                            - (
                                (
                                    (
                                        (col("ra_fixed") / 15)
                                        - (
                                            (col("ra_fixed") / 15).cast(
                                                "bigint"
                                            )
                                        )
                                    )
                                    * 60
                                ).cast("bigint")
                            )
                        )
                        * 60
                    ).cast("bigint")
                )
            ).alias("ra_ff")
        ),
    )

    # Fix negative Dec
    df = df.select(
        "*",
        when(col("dec") >= 0, lit(1)).otherwise(lit(0)).alias("h"),
        abs(col("dec")).alias("dec_fixed"),
    )

    # Calculate Dec components
    # Calculate Dec components
    df = df.select(
        "*",
        (col("dec_fixed") / 15).cast("bigint").alias("dec_deg"),
        (
            ((col("dec_fixed") / 15) - (col("dec_fixed") / 15).cast("bigint"))
            * 60
        )
        .cast("bigint")
        .alias("dec_mm"),
        (
            (
                (
                    (
                        (col("dec_fixed") / 15)
                        - (col("dec_fixed") / 15).cast("bigint")
                    )
                    * 60
                )
                - (
                    (
                        (col("dec_fixed") / 15)
                        - (col("dec_fixed") / 15).cast("bigint")
                    )
                    * 60
                ).cast("bigint")
            )
            * 60
        )
        .cast("bigint")
        .alias("dec_ss"),
        (
            (
                (
                    (
                        (
                            (col("dec_fixed") / 15)
                            - (col("dec_fixed") / 15).cast("bigint")
                        )
                        * 60
                    )
                    - (
                        (
                            (col("dec_fixed") / 15)
                            - (col("dec_fixed") / 15).cast("bigint")
                        )
                        * 60
                    ).cast("bigint")
                )
                * 60
            )
            - (
                (
                    (
                        (
                            (
                                (col("dec_fixed") / 15)
                                - (col("dec_fixed") / 15).cast("bigint")
                            )
                            * 60
                        )
                        - (
                            (
                                (col("dec_fixed") / 15)
                                - (col("dec_fixed") / 15).cast("bigint")
                            )
                            * 60
                        ).cast("bigint")
                    )
                    * 60
                ).cast("bigint")
            )
        ).alias("dec_f"),
    )

    # Calculate the aid
    df = df.select(
        "*",
        (
            lit(1000000000000000000)
            + (col("ra_hh") * 10000000000000000)
            + (col("ra_mm") * 100000000000000)
            + (col("ra_ss") * 1000000000000)
            + (col("ra_ff") * 10000000000)
            + (col("h") * 1000000000)
            + (col("dec_deg") * 10000000)
            + (col("dec_mm") * 100000)
            + (col("dec_ss") * 1000)
            + (col("dec_f") * 100)
        )
        .cast(StringType())
        .alias("aid"),
    )

    # Drop intermediate columns
    df = df.drop(
        "ra_hh",
        "ra_mm",
        "ra_ss",
        "ra_ff",
        "ra_fixed",
        "h",
        "dec_deg",
        "dec_mm",
        "dec_ss",
        "dec_f",
        "dec_fixed",
    )

    return df


def aid_replacer(df):
    df = df.repartition(col("oid"))
    window_spec = Window.partitionBy(col("oid")).orderBy(col("mjd"))
    df = df.withColumn(
        "aid_first_mjd",
        first("aid").over(
            window_spec.rowsBetween(
                Window.unboundedPreceding, Window.unboundedFollowing
            )
        ),
    )
    df = df.drop(col("aid")).withColumnRenamed("aid_first_mjd", "aid")
    return df


############################### EXECUTE ##############################################


def run_sorting_hat_step(spark, avro_parquets_dir):
    df = load_dataframes(avro_parquets_dir, spark)
    df = df_sorting_hat(df)
    sorted_columns_df = sorted(df.columns)
    df = df.select(*sorted_columns_df)
    return df
