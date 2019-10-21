from elasticsearch import Elasticsearch
from elasticsearch import helpers
import jsonlines
from pathlib import Path
import gzip
import src.utils.logger as logger
import re
import pandas as pd

log = logger.init("./log/create_corpus.log")
data_path = "./semanticscholar"

processed = pd.read_csv("./log/create_corpus.log", sep = ' ',   header = None)[9]
processed = [re.sub('semanticscholar/', '', obj) for obj in processed]


def doc_generator(reader):
        for doc in reader.iter(type=dict, skip_invalid=True): 
                author_names = []
                author_ids = []
                for obj in doc.get('authors'):
                        author_ids.extend(obj.get('ids'))
                        author_names.append(obj.get('name'))

                yield {
                        "_index": 'semanticscholar',
                        "_type": "document",
                        "_id" : doc.get('id'),
                        "title": doc.get('title'),
                        "paperAbstract": doc.get("paperAbstract"),
                        "entities": doc.get("entities"),
                        "author_names": author_names,
                        "author_ids": author_ids,
                        "inCitations": len(doc.get("inCitations")),
                        "outCitations": len(doc.get("outCitations")),
                        "year": doc.get("year"),
                        "venue": doc.get('venue'),
                        "journalName": doc.get('journalName'),
                        "journalVolume": doc.get('journalVolume'),
                        "sources": doc.get('sources'),
                        "doi": doc.get('doi')
                        }



es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])

path = Path(data_path)

for file in path.iterdir():
        if file.suffix == ".gz":
                if file.name not in processed:
                        log.info("process file %s", file)
                        with gzip.open(str(file)) as f:
                                reader = jsonlines.Reader(f)
                                for success, info in helpers.parallel_bulk(es, doc_generator(reader), 
                                                                           chunk_size=100, max_chunk_bytes = 1000*1000*25, request_timeout = 30):
                                        if not success: log.errror('doc failed', info)


                

