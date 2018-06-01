import datetime
import logging
import os
import tempfile

# Global variables
BATCH_NAME = None
BATCH_OUTPUT_FOLDER = None
TEMP_DIR = None


def get_batch_name():
    """
    Get the name of the current run. This is a unique identifier for each run of this application
    :return: The name of the current run. This is a unique identifier for each run of this application
    :rtype: str
    """
    global BATCH_NAME

    if BATCH_NAME is None:
        logging.info('Batch name not yet set. Setting batch name.')
        BATCH_NAME = str(datetime.datetime.utcnow()).replace(' ', '_').replace('/', '_').replace(':', '_')
        logging.info('Batch name: {}'.format(BATCH_NAME))
    return BATCH_NAME


def get_batch_output_folder():
    global BATCH_OUTPUT_FOLDER
    if BATCH_OUTPUT_FOLDER is None:
        BATCH_OUTPUT_FOLDER = os.path.join('../data/output', get_batch_name())
        os.mkdir(BATCH_OUTPUT_FOLDER)
        logging.info('Batch output folder: {}'.format(BATCH_OUTPUT_FOLDER))
    return BATCH_OUTPUT_FOLDER