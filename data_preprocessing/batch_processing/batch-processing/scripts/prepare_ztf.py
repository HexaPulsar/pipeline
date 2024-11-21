import argparse

from performance_timer import PerformanceTimer

from batch_processing.ztf import ZTFCrawler

with PerformanceTimer("[SETUP] Parse arguments"):
    parser = argparse.ArgumentParser(
        description="Script to convert folders with AVRO files to PARQUET files"
    )

    parser.add_argument("--mjd", type=str, help="MJD", required=True)

    parser.add_argument(
        "-i",
        "--input-avro-folder",
        type=str,
        help="Path where input avro files are located",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-parquet-folder",
        type=str,
        help="Path where output parquet files will be stored",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--batch-size",
        type=str,
        help="AVRO alerts per parquet file",
        default=50000,
    )
    args = parser.parse_args()

with PerformanceTimer("[SETUP] initialize ZTFCrawler"):
    ztf = ZTFCrawler(
        args.mjd,
        args.input_avro_folder,
        args.output_parquet_folder,
        int(args.batch_size),
    )

with PerformanceTimer("[ZTFCrawler] execute"):
    ztf.execute()
