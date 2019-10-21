[![DOI](https://zenodo.org/badge/216515658.svg)](https://zenodo.org/badge/latestdoi/216515658)

# TREC 2019 Fair Ranking Track

- `config` contains configurations files
- `semanticscholar` contains the raw data files from https://api.semanticscholar.org/corpus/. The files' ending was changed to `json.gz` with the `./src/etl/rename.py` script
- `elasticsearch` contains the index data with the semanticscholar corpus
- `src` contains the modules
- `training` contains the provided training docs, group defintitions, a python script to generate query sequences, the training corpus for feature engineering
- `evaluation` contains the provided evaluation files and a python script to validate a submission

## requirements

- training, evaluation and submission files from the [fair-trec website](https://fair-trec.github.io/)
- [pandas](https://pandas.pydata.org/)
- [fairsearchdeltr](https://github.com/fair-search/fairsearch-deltr-python)
- elasticsearch [python client](https://elasticsearch-dsl.readthedocs.io/en/latest/)
- elasticsearch [docker instance](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)

#### build the ltr module for elasticsearch in docker

```bash
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.0.1
docker build -t elasticsearch-ltr ./config
```

#### start the elasticsearch db

```console
docker run -d --rm --name es \
-p 9200:9200 -p 9300:9300 \
-e "discovery.type=single-node" \
-e "http.cors.enabled=true" \
-e "http.cors.allow-origin=*" \
-e "http.cors.allow-headers=X-Requested-With,X-Auth-Token,Content-Type,Content-Length,Authorization" \
-e "http.cors.allow-credentials=true" \
-e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
-v `pwd`/elasticsearch:/usr/share/elasticsearch/data \
elasticsearch-ltr
```

#### push the data to es and remove missing ids

```console
python ./src/etl/data_to_es.py
python ./src/etl/remove_missing_ids.py
```

## example

#### random run

```console
python -m src.runs.random
```

#### lambdamart

```console
python -m src.runs.lambdamart
```
#### programmatic access

see the example scripts in `src/runs`


## some stats

- the corpus contains `46 947 044` unique documents
- the training sample contains `4641` documents (`4490` unique docs) and `652` queries
- the cleand training sample contains `557` queries, as some doc_ids are missing in the corpus (see `./src/etl/remove_missing_ids.py`)
- 3863 docs from the training sample are included in the corpus
- the length of each ranking ranges from `2` to `26` docs with an average of `7` docs
- on average arround `50.94%` of docs per query are not relevant

## modules

### runs

- contains the final run script that build on all other modules

### etl

- imports and maps the raw data to elasticsearch index

### interface

- processes the input training and group files (`inputhandler`)
- layer between program modules and elasticsearch (`corpus`)

### reranker

- contains the learning to rank model to rerank the document sets: [DELTR algorithm](https://github.com/fair-search/fairsearch-deltr-python) for training
- provides implementations of the evaluation measures (`evaluation.py`)
- provides module to generate features from the corpus (`features.py`)

### utils

- contains modules for command line args, logger file initialization and IO functionalities

### test

- contains test files and scripts
