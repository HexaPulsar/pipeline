import os
from typing import Any

import fastavro
import fastavro.schema
import polars

from .schema import schema, schema_without_fp
from .utils import configure_logger


class ZTFCrawler:
    """
    ZTFCrawler class for handling ZTF data processing tasks.
    """

    def __init__(
        self,
        mjd: int,
        input_avro_folder: str,
        output_parquet_folder: str,
        batch_size: int,
    ):
        """
        Initialize the ZTFCrawler class.

        :param config_path: Path to the configuration file in JSON format.
        """
        # Necessary variables
        self.mjd = mjd
        self.avros_folder = f"{input_avro_folder}"
        self.output_folder = f"{output_parquet_folder}"
        self.batch_size = batch_size

        # Configure logger
        self.logger = configure_logger()

    def execute(self):
        self._create_parquet_files()

    def _df(self, batch: list[dict[str, Any]]) -> polars.DataFrame:
        df = polars.from_dicts(batch)
        if "fp_hists" in df.columns:
            return polars.DataFrame(df, schema=schema)
        else:
            return polars.DataFrame(df, schema=schema_without_fp)

    def _create_parquet_files(self):
        """
        Create Parquet files from an avro generator.

        :param avro_generator: Generator that yields dictionaries.
        :param batch_size: Size of each batch.
        :param output_folder: Folder where the Parquet files will be saved.
        """
        avro_generator = self._avro_generator(self.avros_folder)
        # if batch_size == 0, process everything in one file
        if self.batch_size == 0:
            batch = list(avro_generator)
            self._write_batch_to_parquet(self._df(batch), self.output_folder, "0")

        # elif batch_size > 0, create files of that size.
        elif self.batch_size > 0:
            batch = []
            current_batch = 0
            for avro_df in avro_generator:
                batch.append(avro_df)
                if len(batch) == self.batch_size:
                    self._write_batch_to_parquet(
                        self._df(batch), self.output_folder, str(current_batch)
                    )
                    batch = []
                    current_batch += 1

            # Write the remaining items in the last batch (if any)
            if len(batch) > 0:
                self._write_batch_to_parquet(
                    self._df(batch), self.output_folder, str(current_batch)
                )
        else:
            raise ValueError("batch_size must be a number greater or equal to 0")
        del batch

    def _avro_generator(self, avros_directory: str, avro_schema=None):
        """
        Get Avro data generator for ZTF public data for a specific MJD.

        :param mjd: Modified Julian Date.
        :param avro_schema: Avro schema for data reading.
        :return: Avro data generator.
        """

        avro_paths = [
            os.path.join(avros_directory, avro) for avro in os.listdir(avros_directory)
        ]

        for avro_path in avro_paths:
            with open(avro_path, "rb") as encoded_avro:
                avro_reader = fastavro.reader(encoded_avro, avro_schema)
                avro = next(avro_reader)
                avro = self._drop_avro_columns(avro)
            yield avro

    def _write_batch_to_parquet(self, df, output_folder: str, name: str):
        """
        Write a batch of dictionaries to a Parquet file.

        :param batch: List of dictionaries.
        :param output_folder: Folder where the Parquet file will be saved.
        :param name: Param used to identify the parquet file.
        """
        compression = "snappy"
        parquet_filename = f"{name}.{compression}.parquet"
        parquet_path = os.path.join(output_folder, parquet_filename)
        df.write_parquet(parquet_path, compression=compression)
        self.logger.info(f"Parquet file written: {parquet_path}")

    def _drop_avro_columns(self, avro: dict) -> dict:
        columns = [
            "cutoutScience",
            "cutoutTemplate",
            "cutoutDifference",
            "candid",
        ]
        for column in columns:
            del avro[column]
        return avro
