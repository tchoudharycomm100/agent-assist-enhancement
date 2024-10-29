# python 3.11
# config.py

# DATA_DIR
DATA_DIR = "./data/CISI.ALL"

# Sentence Transformer
MODEL_NAME = "sentence-transformers/stsb-roberta-large"

# ES
HOST_URL = "http://elasticsearch:9200"
INDEX_NAME = "kb-data"
INDEX_SETTINGS = {
    "settings": {"index": {"number_of_shards": 2, "number_of_replicas": 1}},
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "standard"},
            "author": {"type": "text", "analyzer": "standard"},
            "abstract": {"type": "text", "analyzer": "english"},
            "embedding": {"type": "dense_vector", "dims": 1024},
        }
    },
}
