from src.interface.corpus import Corpus
from src.interface.inputhandler import InputHandler
import src.utils.io as io

"""
It has come to our attention that the training query set includes document IDs that cannot be resolved in the OpenCorpus download. 
We therefore recommend removing such IDs.
There is a two-step procedure you can follow to remove such documents:

    Remove all documents from query 'documents' sets that do not exist in OpenCorpus
    Drop all queries that, after this document removal, have fewer than 5 documents

This will result in dropping approximately 100 training queries.
"""

corpus = Corpus()
input = InputHandler(corpus, fquery = './training/fair-TREC-training-sample.json')

queries = input.get_queries()

doc_ids = queries.doc_id.drop_duplicates().to_list()
ids_available = corpus.get_docs_by_ids(doc_ids).doc_id.drop_duplicates().to_list()


ids_missing = [id for id in doc_ids if id not in ids_available]

queries = queries.loc[queries.doc_id.isin(ids_available)]

result_size = queries.groupby('qid').doc_id.count()


queries_remove = result_size[result_size < 5].keys().to_list()

queries_raw = io.read_jsonlines('./training/fair-TREC-training-sample.json')

queries_raw = [query for query in queries_raw if query['qid'] not in queries_remove]


io.write_jsonlines(queries_raw, './training/fair-TREC-training-sample-cleaned.json')

