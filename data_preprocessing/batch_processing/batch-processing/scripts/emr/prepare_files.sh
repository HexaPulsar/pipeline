#!/bin/bash

# Check input
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <base_path>"
  exit 1
fi

# Define variables
BASE_DIR=$1
DIST_DIR="${BASE_DIR}/dist"
VENV_PATH=$(python -m poetry run which python | sed 's|/bin/python||')
SITE_PACKAGES="${VENV_PATH}/lib/python3.9/site-packages"
SITE_PACKAGES64="${VENV_PATH}/lib64/python3.9/site-packages"

# Function to prepare Python dependencies with Poetry
prepare_poetry() {
  cd "${BASE_DIR}" || { echo "Failed to change directory to ${BASE_DIR}"; exit 1; }
  python -m poetry lock || { echo "Failed to create lock file."; exit 1; }
  python -m poetry install || { echo "Failed to install Poetry dependencies."; exit 1; }
  python -m build || { echo "Failed to build Poetry project."; exit 1; }
}

# Function to create a zip file with the content of a specified folder
zip_folder_content() {
  origin=$1
  destination=$2
  filename=$3
  cd "${origin}" || { echo "Failed to change directory to ${VENV_PATH}"; exit 1; }
  zip -r "${DIST_DIR}/python-venv.zip" . || { echo "Failed to zip ${VENV_PATH}"; exit 1; }
}

# Prepare files for Spark
prepare_venv() {
  # Get Python executable path from poetry
  PYTHON_PATH=$(python -m poetry run which python)

  # Install built wheel explictly
  ${PYTHON_PATH} -m pip install "${DIST_DIR}/batch_processing-0.1.0-py3-none-any.whl"

  # Zip python-venv
  zip_folder_content ${VENV_PATH} ${DIST_DIR} python-venv.zip
}

main() {
  prepare_poetry
  prepare_venv
}

main
echo "Script completed successfully."
