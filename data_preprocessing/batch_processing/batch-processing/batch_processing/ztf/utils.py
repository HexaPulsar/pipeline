import logging
import os
import sys
from datetime import datetime, timedelta

import polars


def date_to_mjd(date: datetime) -> str:
    """
    Receives a date and transforms it to MJD.
    Possible date formats:
    https://docs.python.org/3/library/datetime.html#format-codes

    :param date:  date string
    :param format: string indicating date format
    :return: the date as MJD
    """
    reference_date = datetime(1858, 11, 17)
    mjd = (date - reference_date).days
    return str(mjd)


def str_to_date(date: str, format: str = "%Y/%m/%d"):
    return datetime.strptime(date, format)


def dates_between_generator(StartDate, EndDate):
    for n in range(int((EndDate - StartDate).days) + 1):
        yield StartDate + timedelta(n)


def configure_logger():
    # Create a stream handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)

    # Set loggin level
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setLevel(logging.ERROR)

    # Create a formatter and set it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger


def _rm_directory_or_file(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        # Get the list of files in the folder
        entries = os.listdir(path)

        # Iterate through the files and remove them
        for entry in entries:
            entry_path = os.path.join(path, entry)
            try:
                if os.path.isfile(entry_path):
                    os.remove(entry_path)
                elif os.path.isdir(entry_path):
                    # Recursively call _rm_directory_or_file for subdirectories
                    _rm_directory_or_file(entry_path)

            except Exception as e:
                print(f"Error: {e}")

        # rm folder
        os.rmdir(path)


def drop_polars_columms(df: polars.DataFrame, columns: list[str]):
    for column in columns:
        df.drop_in_place(column)
    return df
