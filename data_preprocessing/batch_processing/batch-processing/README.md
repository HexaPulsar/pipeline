# Batch Processing Module

The Batch Processing module is currently designed for handling large-scale data processing tasks, specifically focused on ZTF (Zwicky Transient Facility) data. This module includes classes for reading Parquet files (ParquetReader) and utility functions for various tasks related to ZTF data processing (ZTFCrawler).

## ParquetReader Class

### Overview

The ParquetReader class facilitates the reading and processing of Parquet files. It is configured using a JSON file and provides methods to iterate through dataframes within a specified date range.

### Configuration

The configuration for the `ParquetReader` class is provided in a JSON file. 
Below is an example configuration file (`read.config.json`):

```json
{
    "DataFolder": "/home/alerce/batch_processing/data",
    "RawParquetFolder": "parquet",
    "StartDate": "2023/12/26",
    "EndDate": "2023/12/27",
    "DateFormat": "%Y/%m/%d"
}
```

For more details on the configuration parameters and usage, refer to the Configuration README.

### Usage
Refer to the `read.py` example in the `/scripts` folder.

## ZTFCrawler Class

### Overview

The ZTFCrawler class provides utility functions for various tasks related to ZTF data processing, including downloading, untarring, and creating Parquet files.

### Configuration

The configuration for the `ZTFCrawler` class is provided in a JSON file. Below is an example configuration file (`raw.config.json`):

```json
{
    "UrlSourceBase": "https://ztf.uw.edu/alerts/public/",
    "UrlSourceFilePrefix": "ztf_public_",
    "UrlSourceFilePostfix": ".tar.gz",
    "DataFolder": "/home/alerce/batch_processing/data",
    "CompressedAvrosFolder": "compressed_avros",
    "UncompressedAvrosFolder": "uncompressed_avros",
    "RawParquetFolder": "parquet",
    "BatchSize": 2500,
    "ParquetCompression": "snappy",
    "DeleteData": "false",
    "OverwriteData": "false",
    "StartDate": "2023/12/26",
    "EndDate": "2023/12/27",
    "DateFormat": "%Y/%m/%d"
}
```

For more details on the configuration parameters and usage, refer to the Configuration README.

### Usage
Refer to the `prepare_ztf.py` example in the `/scripts` folder.

## Installation

`poetry install`