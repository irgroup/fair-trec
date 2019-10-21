from argparse import ArgumentParser


def init():
    parser = ArgumentParser()
    parser.add_argument("-db", "--db-file", dest="db_path",
                        help="json config for database", metavar="FILE", default= "./config/db_config.json")
    parser.add_argument("-d", "--data-path", dest="data_path",
                        help="path to folder containing raw data", metavar="FOLDER", default = "./semanticscholar")
    parser.add_argument("-b", "--batch-size", dest="batch_size",
                        help="batch size when writing data to db", metavar="VALUE", default = 1000, type = int)
    parser.add_argument("-l", "--log-file", dest="log_path",
                        help="path to a log file", metavar="FILE", default = "./log.log")
    parser.add_argument("", "--reset", dest="reset", action='store_true',
                    help="remove data in db")
    return parser.parse_args()