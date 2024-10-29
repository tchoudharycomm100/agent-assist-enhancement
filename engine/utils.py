import config

from typing import List, Dict
from elasticsearch import Elasticsearch
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer


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


def get_ranked_docs_from_query(es_client: Elasticsearch, text_query: str) -> List[Dict]:
    """
    Retrieves relevant documents for a given text query

    Args:
        es_client (Elasticsearch): Elasticsearch instance
        query (str): The input text to search

    Returns:
        List[Dict]: List of returned documents
    """
    text_embedding_obj = TextEmbedding(config.MODEL_NAME)
    embedding = text_embedding_obj.get_text_embedding(text_query)

    es_query = {
        "field": "embedding",
        "query_vector": embedding,
        "k": 20,
        "num_candidates": 100,
    }

    hits = es_client.search(
        index=config.INDEX_NAME,
        knn=es_query,
        source=[
            "id",
            "title",
            "abstract"
        ],
        size=9,
    )

    kb_docs = [x["_source"] for x in hits["hits"]["hits"]]
    ranked_docs = get_reranked_docs(text_query, kb_docs)
    return ranked_docs


def get_reranked_docs(query: str, docs: List[Dict]) -> List[Dict]:
    """
    Reranks a set of documents based on a given text query

    Args:
        query (str): The original input search query
        docs (List[Dict]): List of retrieved documents in dictionary format

    Returns:
        List[Dict]: List of reranked documents
    """

    ranker = Ranker(model_name=config.RANKER_MODEL_NAME, cache_dir="/opt")
    passages = transform_data(docs)
    rerankrequest = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerankrequest)

    for result in results:
        result['score'] = round(result['score'].item()*100, 3)

    return results


def transform_data(data: List[Dict]) -> List[Dict]:
    """
    Transforms a list of dictionaries by renaming 'abstract' to 'text' and 
    moving 'title' under a new 'meta' key.

    Args:
        data (List[Dict]): List of dictionaries to transform.

    Returns:
        List[Dict]: Transformed list of dictionaries.
    """
    transformed_data = []
    for item in data:
        transformed_item = {
            "id": item["id"],
            "text": item["abstract"],
            "meta": {
                "title": item["title"]
            }
        }
        transformed_data.append(transformed_item)
    return transformed_data


books = [
    {
        "id": 1,
        "abstract": "An in-depth exploration of the effects of technology on modern society.",
        "meta": {
            "title": "The Digital Frontier"
        },
        "score": 0.932
    },
    {
        "id": 2,
        "abstract": "A historical recount of the events that shaped the world in the 20th century.",
        "meta": {
            "title": "Shadows of the Past"
        },
        "score": 0.875
    },
    {
        "id": 3,
        "abstract": "A gripping fantasy novel set in a world where magic and science collide.",
        "meta": {
            "title": "The Mage's Dilemma"
        },
        "score": 0.75
    },
    {
        "id": 4,
        "abstract": "A comprehensive guide to mastering the art of storytelling in various forms.",
        "meta": {
            "title": "Narrative Alchemy"
        },
        "score": 0.625
    },
    {
        "id": 5,
        "abstract": "An insightful look into the psychology of decision-making and its implications.",
        "meta": {
            "title": "Choices and Consequences"
        },
        "score": 0.5
    },
    {
        "id": 6,
        "abstract": "A dystopian science fiction novel exploring a society ruled by corporate powers.",
        "meta": {
            "title": "Empire of Shadows"
        },
        "score": 0.375
    },
    {
        "id": 7,
        "abstract": "A thrilling detective story set in a city teetering on the brink of chaos.",
        "meta": {
            "title": "The Last Witness"
        },
        "score": 0.25
    },
    {
        "id": 8,
        "abstract": "An inspiring autobiography of a trailblazing female entrepreneur in the tech industry.",
        "meta": {
            "title": "Breaking Barriers"
        },
        "score": 0.125
    },
    {
        "id": 9,
        "abstract": "An engaging exploration of culinary arts and the stories behind iconic dishes.",
        "meta": {
            "title": "The Chef's Secret"
        },
        "score": 0.056
    }
]
