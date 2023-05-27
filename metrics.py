import torch
from dataset import MangaDataset
from manga109utils import BoundingBox
from tqdm import tqdm
from collections import defaultdict
from retrieve import TranscriptRetriever, SceneRetriever
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def evaluate_bounding_boxes(expected: list[list[BoundingBox]], pred: list[list[tuple[BoundingBox, float]]]) -> dict:
    metric = MeanAveragePrecision()
    for expected_item, pred_item in zip(expected, pred):
        expected_dict = {"boxes": torch.tensor([[bb.xmin, bb.ymin, bb.xmax, bb.ymax] for bb in expected_item]),
                         "labels": torch.tensor([bb.bbtype for bb in expected_item])}
        pred_dict = {"boxes": torch.tensor([[bb.xmin, bb.ymin, bb.xmax, bb.ymax]for bb, _ in pred_item]),
            "scores": torch.tensor([ score for _, score in pred_item]), 
            "labels": torch.tensor([ bb.bbtype for bb, _ in pred_item])
        }
        metric.update([pred_dict], [expected_dict])
    return metric.compute()

def retrieval_eval_metric(retriever, queries: list[str], expected: list[int], k_list:list[int]) -> dict:
    '''
    retriever: TranscriptRetriever or SceneRetriever
    queries: list of queries
    expected: list of expected indices for each query, length of expected should be equal to length of queries
    k_list: list of k used to calculate MRR@k and average success@k
    '''

    reciprocal_rank, success = defaultdict(list), defaultdict(list)

    for query, expect_idx in zip(queries, expected):
        # results = retriever.query(query, topk=max(k_list))
        results = retriever.query(query)[:max(k_list)]
        rank = results.index(expect_idx)+1 if expect_idx in results else -1
        for k in k_list:
            rr_at_k = 1 / rank if rank != -1 and rank <= k else 0
            success_at_k = 1 if rank != -1 and rank <= k  else 0
            reciprocal_rank[k].append(rr_at_k)
            success[k].append(success_at_k)

    result = {}
    for k in k_list:
        result[f"MRR@{k}"] = sum(reciprocal_rank[k]) / len(reciprocal_rank[k])
        result[f"avg success@{k}"] = sum(success[k]) / len(success[k])
    return result
    


        
