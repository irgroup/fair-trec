import logging
import sys
import warnings


def init(log_path):

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)s]  %(message)s")

    logging.captureWarnings(True)

    rootLogger = logging.getLogger()
    wLogger = logging.getLogger('py.warnings')
    esLogger = logging.getLogger('elasticsearch')
    
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    esLogger.addHandler(fileHandler)
    wLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    esLogger.addHandler(consoleHandler)
    wLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)
    wLogger.setLevel(logging.INFO)
    esLogger.setLevel(logging.WARNING)

    return rootLogger
