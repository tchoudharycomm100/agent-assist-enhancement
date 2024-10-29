import time
import math
import config
import logging

import numpy as np

from typing import List, Dict
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import ConnectionError
from sentence_transformers import SentenceTransformer


# logging
logging.basicConfig(filename="examploggerle.log", level=logging.INFO)


class TextEmbedding:
    """Class representing text embedding"""

    def __init__(self, model_name: SentenceTransformer) -> None:
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}'") from e

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Generates the text embedding for a given input text.

        Args:
            text (str): The input text for which to generate an embedding.

        Returns:
            List[float]: The embedding vector as a list of floats.
        """
        embedding = self.model.encode([text])
        return embedding.tolist()[0]

    def get_batch_text_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates the text embeddings for a batch of input texts.

        Args:
            texts (List[str]): The input texts for which to generate embeddings.
            batch_size (int): The batch size to use for processing. Default is 32.

        Returns:
            List[List[float]]: A list of embedding vectors for each input text.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            embeddings.extend(batch_embeddings.tolist())
            logging.info(f"Indexing batch {i//32+1}/{math.ceil(len(texts)/32)} done!")
        return embeddings


class Data:
    """Class representing knowledge data."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    def get_raw_data(self) -> List[Dict]:
        """
        Returns:
            List[Dict]: List of data dictionaries
        """
        data = []
        content = []

        with open(self.data_dir, encoding="utf-8") as file:
            try:
                for line in file:
                    if line.startswith(".I"):
                        content.append(('id', line[3:].strip()))

                    elif line.startswith(".T"):
                        content.append(('title', next(file).strip()))

                    elif line.startswith(".A"):
                        content.append(('author', next(file).strip()))

                    elif line.startswith(".W"):
                        abstract = []
                        next_line = next(file)

                        while True:
                            if next_line.startswith(".X"):
                                break

                            abstract.append(next_line.strip())
                            next_line = next(file)

                        content.append(("abstract", " ".join(abstract)))

                    temp_dict = dict(content)

                    if len(temp_dict.keys()) == 4:
                        data.append(temp_dict)
                        content = []

            except StopIteration:
                pass

        return data


def spin_es_cluster() -> Elasticsearch:
    """
    Function to spin up the ES cluster
    """
    es = Elasticsearch(hosts=[config.HOST_URL])

    for _ in range(100):
        try:
            # make sure the cluster is available
            es.cluster.health(wait_for_status="yellow")

        except ConnectionError:
            logging.warning("Elasticsearch is still not up!!!")
            time.sleep(2)

    logging.info("Elasticsearch is up!!!")
    return es


def create_index(es_client: Elasticsearch,
                 index_name: str,
                 index_settings: Dict,
                 refresh: bool = True) -> None:
    """
    Function to create the indices in the es

    Args:
        es_client (Elasticsearch): Elasticsearch instance
        index_name (str): name of the index
        index_settings (Dict): settings dictionary for the index
    """
    es_client.indices.create(
        index=index_name,
        body=index_settings
    )

    # get docs
    docs = Data(config.DATA_DIR).get_raw_data()

    # create embedding vectors
    text_embedding_obj = TextEmbedding(config.MODEL_NAME)
    abstracts = [doc['abstract'] for doc in docs]
    embeddings = text_embedding_obj.get_batch_text_embeddings(
        abstracts, batch_size=32)

    for i, doc in enumerate(docs):
        doc['embedding'] = embeddings[i]

    logging.info("Starting indexig!!!")
    requests = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": i,
            "id": doc["id"],
            "title": doc["title"],
            "author": doc["author"],
            "abstract": doc["abstract"],
            "embedding": doc["embedding"],
        }
        for i, doc in enumerate(docs)
    ]

    bulk(es_client, requests)

    if refresh:
        es_client.indices.refresh(index=index_name)

    logging.info("Indexing complete.")


def main() -> None:
    """
    Main function to call
    """
    es_client = spin_es_cluster()
    create_index(es_client, config.INDEX_NAME,
                 config.INDEX_SETTINGS, refresh=True)


if __name__ == "__main__":
    main()
