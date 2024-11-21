#!/bin/bash

set -x # Debug mod

export PATH=$PATH:/path/to/duckdb  # NEED TO CHANGE TO YOUR LOCAL path to duckdb

# PostgreSQL connection details
PG_HOST="localhost"
PG_PORT="5432"
PG_DB="your_database"
PG_USER="your_username"
PG_PASSWORD="your_password"

process_parquet_to_postgres() {
    local parquet_file="$1"
    local parent_folder=$(basename "$(dirname "$parquet_file")")
    local table_name=$(echo "$parent_folder" | sed 's/[^a-zA-Z0-9_]/_/g')

    echo "Processing $parquet_file into table $table_name"
    
    # Use full path to DuckDB (modify as needed) # NEED TO CHANGE TO YOUR LOCAL path to duckdb
    /path/to/duckdb/duckdb -c "

        INSTALL postgres;
        LOAD postgres;
        ATTACH 'host=${PG_HOST} dbname=${$PG_DB} user=${$PG_USER} password=${PG_PASSWORD}' AS pg (TYPE postgres);

        INSERT INTO pg.${table_name} 
        SELECT * FROM read_parquet('${parquet_file}');
    "

    if [ $? -eq 0 ]; then
        echo "Successfully loaded $parquet_file into PostgreSQL table $table_name"
    else
        echo "Failed to load $parquet_file"
    fi
}

process_folder() {
    local folder="$1"
    local start_time=$(date +%s)
    
    echo "Processing folder: $folder"
    
    # Find all .parquet files and sort them numerically
    find "$folder" -type f -name "*.parquet" | sort -V | while read parquet_file; do
        echo "Processing file: $parquet_file"
        process_parquet_to_postgres "$parquet_file"
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "Time taken to process $folder: $duration seconds"
}

# Main execution
main() {
    local parent_folder="$1"
    local priority_folder="object" # In this case, the object folder must be the first, so is procesed and then there is a loop for the other folders.
    local total_start_time=$(date +%s)

    if [ -z "$parent_folder" ]; then
        echo "Usage: $0 /main/folder/of/parquets" # (optional) NEED TO CHANGE TO YOUR LOCAL path
        exit 1
    fi

    if [ ! -d "$parent_folder" ]; then
        echo "Error: The specified path is not a directory or does not exist."
        exit 1
    fi

    # Process priority folder first
    priority_path="$parent_folder/$priority_folder"
    if [ -d "$priority_path" ]; then
        echo "Processing priority folder: $priority_folder"
        process_folder "$priority_path"
    else
        echo "Priority folder '$priority_folder' not found in $parent_folder"
    fi

    # Process all other folders
    for folder in "$parent_folder"/*; do
        if [ -d "$folder" ] && [ "$folder" != "$priority_path" ]; then
            process_folder "$folder"
        fi
    done

    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    echo "Total time taken: $total_duration seconds"
}

# Run the main function
main "$@"