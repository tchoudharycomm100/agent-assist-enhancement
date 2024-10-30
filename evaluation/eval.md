## To evaluate a batch of queries locally follow the below steps.

- Step 1: Create a list of set of doc_ids for each query in the batch following the query order. These doc_ids come from the ground truths.
```
queries_ground_truth = [
    {'doc1', 'doc2', 'doc3'},
    {'doc1', 'doc2', 'doc3'}
]
```

- Step 2: Create a list of list of doc_ids for each query in the batch following the query order. These doc_ids come from the retrieved documents.
```
queries_retrieved_docs = [
    ['doc1', 'doc4', 'doc5', 'doc2', 'doc3'],
    ['doc1', 'doc2', 'doc3']
]
```

- Step 3: Call the mean_average_precision function havings the above two variables as arguments. Optionally, pass value of K to adjust P@K
```
mean_average_precision(queries_ground_truth, queries_retrieved_docs, k)
```