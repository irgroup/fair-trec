import src.reranker.model as model
from src.interface.corpus import Corpus
from src.interface.iohandler import InputOutputHandler
import pandas as pd
from src.interface.features import FeatureEngineer
import evaluation.validate_run as validate

OUT = "./evaluation/submission_random.json"
QUERIES = "./evaluation/fair-TREC-evaluation-sample.json"
SEQUENCE = "./evaluation/fair-TREC-evaluation-sequences.csv"

corpus = Corpus()
ft = FeatureEngineer(corpus)

input = InputOutputHandler(corpus, 
                     fsequence= SEQUENCE,
                     fquery = QUERIES)
 
random = model.RandomRanker(ft)
random.predict(input)

input.write_submission(random, outfile = OUT)

args = validate.Args(queries=QUERIES, query_sequence_file = SEQUENCE, run_file=OUT)
validate.main(args)

