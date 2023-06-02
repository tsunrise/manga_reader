import torch
from manga109utils import BoundingBox
from collections import defaultdict
from typing import TypedDict, List
from retrieve import Retriever
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import csv
from manga109utils import Book, Page
from sentence_transformers import SentenceTransformer, util as sentence_util
import random
import openai
import dsp
from tqdm import tqdm
import os


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

def retrieval_eval_metric(retriever: Retriever, queries: list[str], expected: list[int], k_list:list[int]) -> dict:
    '''
    retriever: TranscriptRetriever or SceneRetriever
    queries: list of queries
    expected: list of expected indices for each query, length of expected should be equal to length of queries
    k_list: list of k used to calculate MRR@k and average success@k
    '''

    reciprocal_rank, success = defaultdict(list), defaultdict(list)

    for query, expect_idx in zip(queries, expected):
        results = retriever.query(query, top_k=max(k_list))
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

class QuerySet(TypedDict):
    queries: list[str]
    expected: list[int]

def create_scene_retrieval_query_set(csv_path: str) -> QuerySet:
    """
    Creates a query set for scene retrieval from a CSV file.
    """
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        queries = []
        expected = []
        for row in reader:
            if row['Scene Description']:
                queries.append(row['Scene Description'])
                expected.append(int(row['Page Number']))
    return {'queries': queries, 'expected': expected}


def _filter_query(queries: List[str], filter_model: SentenceTransformer, threshold: float = 0.5) -> List[str]:
    """
    Filters a list of queries by removing those that are too similar to each other.

    - `queries`: A list of queries to filter.
    - `model`: A SentenceTransformer model used to encode the queries.
    - `threshold`: The cosine similarity threshold below which queries are considered different.
    """
    embeddings = filter_model.encode(queries)
    cos_sim = sentence_util.cos_sim(embeddings, embeddings)
    second_largest = torch.topk(cos_sim, 2, dim=0)[0][1] # the largest is itself
    mask = second_largest <= threshold
    filtered_queries = [queries[i] for i in range(len(queries)) if mask[i]] 
    return filtered_queries

def create_transcript_retrieval_query_set_for_book(book: Book, filter_model: SentenceTransformer, size=50, threshold=0.5, seed=224, max_pages=None) -> QuerySet:
    """
    Creates a query set for transcript retrieval from a book.
    """
    page_num = book.n_pages
    transcript_2_page_id = {}
    duplicate_transcript = set()
    if max_pages:
        page_num = min(page_num, max_pages)
    for page_id in range(page_num):
        page = Page(book, page_id)
        bbs = page.get_bbs()
        text_bbs = bbs['text']
        transcripts = [bb.text for bb in text_bbs]
        for transcript in transcripts:
            if transcript in transcript_2_page_id:
                duplicate_transcript.add(transcript)
            transcript_2_page_id[transcript] = page_id
    all_transcripts = [transcript for transcript in list(transcript_2_page_id.keys()) if transcript not in duplicate_transcript]

    filtered_queries = _filter_query(all_transcripts, filter_model, threshold)
    rng = random.Random(seed)
    sampled_queries = rng.choices(filtered_queries, k=size)
    expected = [transcript_2_page_id[query] for query in sampled_queries]

    return {'queries': sampled_queries, 'expected': expected}

def paraphrase_query_set(query_set: QuerySet, model: str = "gpt-4") -> QuerySet:
    """
    Paraphrases a query set by replacing each query with its paraphrases.
    Model can be gpt-3.5-turbo or gpt-4.
    """
    lm = dsp.GPT3(model=model, api_key=os.environ.get("OPENAI_API_KEY"), model_type="chat")
    dsp.settings.configure(lm=lm)

    Sentence = dsp.Type(
            prefix="Original Sentence:", 
            desc="${the sentence to be paraphrased}"
        )

    Paraphrase = dsp.Type(
            prefix="Paraphrase:", 
            desc="${the paraphrased version of the sentence}"
        )

    paraphrase_template = dsp.Template(
            instructions="Please paraphrase the sentences in Japanese without using the original language.", 
            sentence=Sentence(), 
            paraphrase=Paraphrase()
        )

    queries = query_set["queries"]
    paraphrased_queries = []
    for query in tqdm(queries, desc="Generating Paraphrases"):
        states_ex = dsp.Example(sentence=query, demos=[]) # zero shot works well 
        states_ex, states_compl = dsp.generate(paraphrase_template)(states_ex, stage='paraphrase')
        paraphrased_queries.append(states_compl.paraphrase)
    
    # replace original queries with paraphrased ones 
    query_set["queries"] = paraphrased_queries
    return query_set

def translate_query_set(query_set: QuerySet, model: str = "gpt-4", src: str = "English", target: str=  "Japanese") -> QuerySet:
    """
    Translate a query set by replacing each query with its translation.
    Model can be gpt-3.5-turbo or gpt-4.
    """
    lm = dsp.GPT3(model=model, api_key=os.environ.get("OPENAI_API_KEY"), model_type="chat")
    dsp.settings.configure(lm=lm)

    Sentence = dsp.Type(
            prefix=f"Original Sentence in {src}:", 
            desc="${the sentence to be translated}"
        )

    Translation = dsp.Type(
            prefix=f"Translated Sentence in {target}:", 
            desc="${the translated version of the sentence.}"
        )

    translation_template = dsp.Template(
            instructions=f"Please translate the sentences from {src} to {target}.", 
            sentence=Sentence(), 
            translation=Translation()
        )

    queries = query_set["queries"]
    translated_queries = []
    for query in tqdm(queries, desc="Generating Translations"):
        states_ex = dsp.Example(sentence=query, demos=[]) # zero shot works well 
        states_ex, states_compl = dsp.generate(translation_template)(states_ex, stage='translate')
        translated_queries.append(states_compl.translation)
    
    # replace original queries with paraphrased ones 
    query_set["queries"] = translated_queries
    return query_set

def save_query_set(query_set: QuerySet, path: str) -> None:
    import json
    with open(path, 'w') as f:
        json.dump(query_set, f)

def load_query_set(path: str) -> QuerySet:
    import json
    with open(path, 'r') as f:
        query_set = json.load(f)
    return query_set

def query_sets_to_csv(english_query_set: QuerySet, japanese_query_set: QuerySet, output_file):
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "english", "japanese"])
        for i in range(len(english_query_set["queries"])):
            assert english_query_set["expected"][i] == japanese_query_set["expected"][i]
            writer.writerow([english_query_set["expected"][i], english_query_set["queries"][i], japanese_query_set["queries"][i]])

def csv_to_query_sets(csv_file):
    english_query_set: QuerySet = {"queries": [], "expected": []}
    japanese_query_set: QuerySet = {"queries": [], "expected": []}
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            english_query_set["expected"].append(row[0])
            english_query_set["queries"].append(row[1])
            japanese_query_set["expected"].append(row[0])
            japanese_query_set["queries"].append(row[2])

    return english_query_set, japanese_query_set


        
