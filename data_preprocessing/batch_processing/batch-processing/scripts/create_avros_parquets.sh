#!/bin/bash

#--------------------------------------- PREPARE METHODS ---------------------------------------
# Function to display usage instructions
usage() {
    echo "Usage: $0 <start_date> <end_date> <num_processes> <s3_bucket>"
    echo "Date format: YYYY-MM-DD"
    exit 1
}

# Check if correct number of arguments is provided
check_args() {
    if [ "$#" -ne 4 ]; then
        usage
    fi
}

# Function to initialize global variables
initialize_variables() {
    start_timestamp=$(date -d "$1" +%s)         # Initial MJD (as timestamp)
    end_timestamp=$(date -d "$2" +%s)           # Final MJD (as timestamp)
    current_timestamp=$start_timestamp          # Current MJD (as timestamp)
    num_processes=$3                            # Maximum processes allowed
    processes=0                                 # Number of processes currently running
    declare -a pids                             # Array to store processes id's
    python_script="scripts/prepare_ztf.py"      # Path to python script that creates parquet files from avro files
    base_p="/home/$(whoami)/tmp/batch-processing/raw_data"     # Base path
    bucket="$4"                                 # AWS S3 Bucket where parquet files will be saved
    logs_p="${base_p}/logs"                     # Path where processes logs will be saved
}

# Check if start_date is lower than end_date
validate_dates() {
    if [ $start_timestamp -gt $end_timestamp ]; then
        echo "Error: start_date cannot be greater than end_date"
        usage
    fi
}

create_directories() {
    local paths=("$@")
    for path in "${paths[@]}"; do
        mkdir -p "$path"
    done
}

# Function to run a command and log its start and finish
run_and_log() {
    local title="$1"
    shift
    local command="$@"
    local title_text=""

    # Check if $title is set and non-empty
    if [ -n "$title" ]; then
        title_text=" | $title"
    fi

    echo "[PID $$${title_text}] RUNNING $command"
    $command
    echo "[PID $$${title_text}] FINISHED $command"
}

upload_to_s3() {
    local files_path=$1
    local s3_path=$2
    local subpath=""
    aws s3 rm s3://${bucket}/${s3_path} --recursive
    # Iterate the folders of s3_path and create them in s3 bucket
    IFS="/" read -r -a components <<< "$s3_path"
    for (( i=0; i<${#components[@]}; i++ )); do
        subpath="${subpath}/${components[i]}"
        aws s3api put-object --bucket $bucket --key $subpath --content-length 0
    done
    aws s3 cp $files_path s3://${bucket}/${s3_path} --recursive
}

prepare_raw_parquets() {
    local mjd="$1"
    local tar_link="$2"
    local avro_folder=${base_p}/avro/${mjd}/
    local parquet_folder=${base_p}/parquet/${mjd}/
    local tar_path=${base_p}/${mjd}.tar.gz

    run_and_log "MJD $mjd" wget -q ${tar_link} -O ${base_p}/${mjd}.tar.gz
    local FILE_SIZE_B=$(du -b "$tar_path" | cut -f1)
    if [ "$FILE_SIZE_B" -gt 100 ]; then
        run_and_log "MJD $mjd" create_directories $avro_folder $parquet_folder
        run_and_log "MJD $mjd" tar -xzvf $tar_path -C $avro_folder 1>/dev/null
        run_and_log "MJD $mjd" rm $tar_path
        run_and_log "MJD $mjd" poetry run python ${python_script} --mjd ${mjd} -i $avro_folder -o $parquet_folder -s 10000
        run_and_log "MJD $mjd" rm -r $avro_folder
        run_and_log "MJD $mjd" upload_to_s3 $parquet_folder data/raw_data/${mjd}/
        run_and_log "MJD $mjd" rm -r $parquet_folder
    else # Log if tar is invalid
        echo "ERROR: Tar file $tar_path is corrupted or invalid" >&2
        run_and_log "MJD $mjd" rm $tar_path
        echo "EXITED WITH ERROR"
        exit 1
    fi
}

# Check if processes are still running. If not, decrease processes count
count_and_update_pids() {
    for pid in ${pids[*]}; do
        if ! ps -p $pid > /dev/null; then
            # Process has finished
            ((processes--))
            pids=(${pids[@]/$pid}) # Remove finished process from array
        fi
    done
}

#--------------------------------------- DEFINE MAIN METHOD --------------------------------------- 

main() {
    echo "STARTING OF PID $$"
    check_args $@
    initialize_variables $@
    validate_dates
    create_directories $logs_p
    while [ $current_timestamp -le $end_timestamp ]; do
        if [ $processes -lt $num_processes ]; then
            local current_date=$(date -d @$current_timestamp +"%Y%m%d")
            local current_mjd=$(( ($current_timestamp / 86400) + 40587 ))
            local tar_link="https://ztf.uw.edu/alerts/public/ztf_public_$current_date.tar.gz"

            if [ ! -e "$base_p/parquet/$current_mjd" ]; then # Check if current parquet exists (only checks if folder exists)
                run_and_log "MAIN" prepare_raw_parquets $current_mjd $tar_link 1>${logs_p}/${current_mjd}.out 2>${logs_p}/${current_mjd}.err &
                pids+=($!)
                ((processes++))
                echo "STARTED $current_mjd in PID $!"
            else
                echo "SKIPPING $current_mjd"
            fi
            current_timestamp=$(expr $current_timestamp + 86400)
        else
            sleep 30
        fi

        count_and_update_pids
        sleep 1 # Adjust sleep time as needed
    done

    wait
    echo "All downloads completed!"
}

#--------------------------------------- RUN MAIN IN ---------------------------------------
main $@

