from typing import List, Set


def precision_at_k(ground_truth_docs: List[Set[int]],
                   retrieved_docs: List[List[int]],
                   k: int = 10) -> float:
    """
    Calculate the precision at K for a set of retrieved documents.

    Parameters:
    - ground_truth_docs (List[Set[int]]): A list of sets of relevant documents (ground truth) for each query.
    - retrieved_docs (List[List[int]]): A list of lists of retrieved documents in ranked order for each query.
    - k (int): The number of top documents to consider for precision.

    Returns:
    - float: Precision at K, which is the proportion of relevant documents in the top K retrieved documents.
    """
    # Ensure k does not exceed the number of retrieved documents
    k = min(k, len(retrieved_docs))

    # Calculate the number of relevant documents in the top K retrieved documents
    relevant_count = sum(1 for i in range(
        k) if retrieved_docs[i] in ground_truth_docs)

    # Precision at K is the proportion of relevant documents among the top K
    return relevant_count / k if k > 0 else 0.0


def mean_average_precision(queries_ground_truth: List[Set[int]],
                           queries_retrieved_docs: List[List[int]],
                           k: int = 10) -> float:
    """
    Calculate the Mean Average Precision (MAP) across multiple queries.

    Parameters:
    - queries_ground_truth (List[Set[int]]): A list containing sets of relevant documents for each query.
    - queries_retrieved_docs (List[List[int]]): A list containing lists of retrieved documents for each query.
    - k (int): The number of top documents to consider for precision calculations.

    Returns:
    - float: The Mean Average Precision (MAP) across all queries.
    """
    # Helper function to calculate Average Precision (AP) for a single query
    def average_precision(ground_truth_docs: Set[int], retrieved_docs: List[int], k: int) -> float:
        relevant_count = 0
        precision_sum = 0.0

        for i in range(min(k, len(retrieved_docs))):
            if retrieved_docs[i] in ground_truth_docs:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)

        if relevant_count == 0:
            return 0.0
        return precision_sum / relevant_count

    # Calculate Average Precision for each query and then the Mean Average Precision
    average_precisions = [
        average_precision(ground_truth_docs, retrieved_docs, k)
        for ground_truth_docs, retrieved_docs in zip(queries_ground_truth, queries_retrieved_docs)
    ]

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
