## Requirement: 
We want to use Semantic retrieval to replace full-text search by Lucene.

## Purpose: 
Improve the accuracy of agent assist suggestions and make search more intelligent

## Proposed:
To migrate from full-text search to semantic search we can create an retrieval and reranking system which could be outlined as follows:

1. Using a pre-trained LM to convert all the data (assuming docs) into embeddings.
2. Store these embeddings in to a vector db:
    - Need to decide if we need a dedicated Vector DB or a general purpose DB with vector capabilities.
    - Also how conservative we want the solution to be - managed or self-hosted DB.
3. Design a retriever that takes in a query (visitor question) and give back the possible docs.
4. Train a reranker that improves the retrieved result and puts the relevant docs over the less relevant ones.
5. Design an evaluation strategy with metrics like mAP, MRR and NDCG.
6. Repeat till we converge.