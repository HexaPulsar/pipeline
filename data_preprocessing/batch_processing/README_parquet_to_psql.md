# Parquet to PostgreSQL Converter
This script provides a solution to convert Parquet files to PostgreSQL databases using DuckDB as an intermediary.

# Requirements

- Python 3.8+
- DuckDB
- PostgreSQL

# Installing DuckDB (Below you can find the instructions for use DuckDB manually)

## Install the necessary dependencies

    apt update $$ apt install -y wget unzip python3 python3-pip

## Download DuckDB from Github
    wget https://github.com/duckdb/duckdb/releases/download/v1.0.0/duckdb_cli-linux-amd64zip

## You can also download it from the main page: 

    https://duckdb.org/docs/installation/?version=stable&environment=cli&platform=linux&download_method=direct&architecture=x86_64

## We decompress the zip

    unzip duckdb_cli-linux-amd64.zip

## Run duckdb if you want

    cd /path/to/duckdb
    ./duckdb

# Configuration

You have to put your local or remote PostgreSQL connection variables in the first lines on the .sh file.

Then you need to change the duckdb paths in the .sh file to your local paths.

# Folders configuration

The folders are configured to work with the following structure:

```
/path/to/parquets
├── folder1/
│   ├── file1.parquet
│   └── file2.parquet
└── folder2/
    └── file3.parquet
```

Is very important that the names of the folders coincide with the names of the tables in the database.

For example if you want to put a parquet or some parquets into a PostgreSQL db with two tables named "table1" and "table2" you have to put the parquets in the following structure:

```
/path/to/parquets
├── table1/
│   ├── file1.parquet
│   └── file2.parquet
└── table2/
    └── file3.parquet 
```

# Runnig the script

To run the script you have to execute the following commands:

- cd /path/to/parquet_to_psql

- ./parquet_to_psql.sh /path/to/parquets

# Using DuckDB

## Conection to DuckDB

    cd /path/to/duckdb
    ./duckdb

## Conection of PosgreSQL

## We can initiate postgresql with the following command:
    sudo -u postgres psql

## We can create a database with the following command:
    sudo -u postgres createdb dbname 

# Conection of DuckDB with PostgreSQL

## We can connect DuckDB with PostgreSQL with the following command:
	ATTACH 'host=localhost dbname=dbname user=user password=password' AS name_to_use_inside_duckdb (TYPE postgres);

## We can see all the tables with the following command:
	show all tables;
## We can see an specific table with the following command:
	select * from name_to_use_inside_duckdb.table_name


# If you want to create a new table from a parquet file in DuckDB:

    CREATE TABLE postgres_db.name_to_use_inside_duckdb  (
    “Type of variables per column”);

## We copy all the data from the parquet file to the table in DuckDB:

    copy postgres_db.name_to_use_inside_duckdb from 'your_file.parquet'

## We can see the data in the table with the following command:

    select * from postgres_db.name_to_use_inside_duckdb

# If you want to insert the data from a parquet file into an existing table in DuckDB:

    insert into name_to_use_inside_duckdb.existing_table select * from 'your_file.parquet';


