import argparse
import logging
import time

from pyspark.sql import SparkSession

from batch_processing.pipeline import (
    correction,
    create_dataframes,
    lightcurve,
    magstats,
    prv_candidates,
    sorting_hat,
    xmatch,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_arguments():
    parser = argparse.ArgumentParser(
        prog="Batch Processing",
        description="Script to run ZTF batch processing",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="""
            Parquet file or folder where parquet files are located.
            Path should not end with /
            Format for S3: s3a://bucket/PATH
        """,
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="""
            Folder where parquet files will be saved.
            Path should not end with /
            Format for S3: s3a://bucket/PATH
        """,
    )
    parser.add_argument(
        "-c",
        "--compression",
        default="snappy",
        help="Compression method to use on saved parquet files. Default is snappy",
    )
    args = parser.parse_args()
    return args


def create_spark_session():
    JARS = "libs/jars/healpix-1.0.jar, libs/jars/minimal_astroide-2.0.0.jar"
    return (
        SparkSession.builder.appName("Batch Processing Pipeline")
        .config("spark.sql.parquet.compression.codec", COMPRESSION)
        .config("spark.jars", JARS)
        .config("spark.sql.constraintPropagation.enabled", "true")
        .config("spark.driver.memory", "9g")
        .config("spark.executor.memory", "6g")
        .getOrCreate()
    )


def write_parquet(df, path):
    df.write.parquet(path, compression=COMPRESSION, mode="overwrite")


def select_compression(compression: str):
    possible_compressions = {
        "lz4",
        "uncompressed",
        "snappy",
        "gzip",
        "lzo",
        "brotli",
        "zstd",
    }
    if compression not in possible_compressions:
        raise ValueError(f"Compression {compression} not available in polars")
    return compression


class ContextExecutor:
    """
    Context manager for cleaner logs and exceptions management
    """

    def __init__(self, title: str):
        self.start_time: float = time.perf_counter()
        self.title: str = title
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.logger.info(f"{self.title} started")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.perf_counter() - self.start_time
        if exc_type is not None:
            # Log exception information
            self.logger.error(
                f"{self.title} finished with an error after {elapsed_time:.2f} seconds: {exc_value}"
            )
            self.logger.error(
                "Traceback:", exc_info=(exc_type, exc_value, traceback)
            )
        else:
            # Log successful completion
            self.logger.info(
                f"{self.title} finished with execution time of: {elapsed_time:.2f} seconds"
            )

        # Returning False allows the exception to propagate, change to True to suppress it
        return False


def main():
    with ContextExecutor("BATCH PROCESSING"):
        spark = create_spark_session()

        with ContextExecutor("SORTING HAT"):
            df_sorting_hat = sorting_hat.run_sorting_hat_step(
                spark, INPUT_PATH
            )

        with ContextExecutor("PRV CANDIDATES"):
            df_prv_candidates = prv_candidates.extract_detections_and_non_detections_dataframe_reparsed(
                df_sorting_hat
            )

        with ContextExecutor("NON DETECTIONS FRAMES"):
            non_detections_frame = (
                create_dataframes.create_non_detections_frame(
                    df_prv_candidates
                )
            )
            write_parquet(
                non_detections_frame, OUTPUT_PATH + "/non_detections"
            )

        with ContextExecutor("DETECTIONS FRAMES"):
            detections_frame = create_dataframes.create_detections_frame(
                df_prv_candidates
            )
            write_parquet(detections_frame, OUTPUT_PATH + "/detections")

        with ContextExecutor("FORCED PHOTOMETRIES FRAMES"):
            forced_photometries_frame = (
                create_dataframes.create_forced_photometries_frame(
                    df_prv_candidates
                )
            )
            write_parquet(
                forced_photometries_frame, OUTPUT_PATH + "/forced_photometries"
            )

        with ContextExecutor("DISCTING FP and ND"):
            forced_photometries_frame = forced_photometries_frame.distinct()
            non_detections_frame = non_detections_frame.distinct()

        with ContextExecutor("LIGHTCURVE"):
            dataframe_lightcurve = lightcurve.execute_light_curve(
                detections_frame,
                non_detections_frame,
                forced_photometries_frame,
            )

        with ContextExecutor("CORRECTIONS"):
            correction_output = correction.produce_correction(
                dataframe_lightcurve
            )
            (
                correction_dataframe,
                forced_photometries_corrected,
                detections_corrected,
            ) = correction_output

            write_parquet(correction_dataframe, OUTPUT_PATH + "/corrections")
            write_parquet(
                forced_photometries_corrected,
                OUTPUT_PATH + "/forced_photometries_corrected",
            )
            write_parquet(
                detections_corrected, OUTPUT_PATH + "/detections_corrected"
            )

        with ContextExecutor("MAGSTATS"):
            object_stats, magstats_stats = magstats.execute_magstats_step(
                correction_dataframe
            )
            write_parquet(object_stats, OUTPUT_PATH + "/objectstats")
            write_parquet(magstats_stats, OUTPUT_PATH + "/magstats")

        with ContextExecutor("XMATCH"):
            correction_dataframe_xmatch_data = correction_dataframe.select(
                "oid",
                "meanra",
                "meandec",
                "detections.aid",
                "detections.ra",
                "detections.dec",
                "detections.forced",
                "detections.extra_fields.sgscore1",
                "detections.extra_fields.distpsnr1",
                "detections.extra_fields.sgmag1",
                "detections.extra_fields.srmag1",
            )

            xmatch_step_output = xmatch.execute_batchsizes(
                spark, correction_dataframe_xmatch_data
            )
            xmatch_step, unprocessed = xmatch_step_output

            if unprocessed:
                logger.info("Wrote to disk: unprocessed light curves")
                write_parquet(unprocessed, OUTPUT_PATH + "/unprocessed")

            xmatch_step_result = xmatch_step.repartition("oid")
            write_parquet(xmatch_step_result, OUTPUT_PATH + "/xmatch")

        spark.stop()


if __name__ == "__main__":
    args = initialize_arguments()
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    COMPRESSION = select_compression(args.compression)
    main()
