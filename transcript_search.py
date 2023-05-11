import torch
import random
from tqdm import tqdm

from constant import MANGA109_ROOT
from dataset import MangaDataset
from manga109utils import Book, Page, Manga109Dataset
from manga_ocr import MangaOcr
from sentence_transformers import SentenceTransformer, util
from transcript_extractor import get_ocr_dataset

def search(query, all_transcript, model):
    transcript_embeddings = model.encode(all_transcript) # (num_pages, 512)
    query_embedding = model.encode(query) # (512,)
    cos_sim = util.cos_sim(query_embedding, transcript_embeddings)
    max_idx = torch.argmax(cos_sim).item()
    selected_transcript = all_transcript[max_idx]
    return max_idx, selected_transcript

def filter_query(queries, model, threshold=0.5):
    embeddings = model.encode(queries)
    cos_sim = util.cos_sim(embeddings, embeddings)
    second_largest = torch.topk(cos_sim, 2, dim=0)[0][1] # the largest is itself
    mask = second_largest <= threshold
    filtered_queries = [queries[i] for i in range(len(queries)) if mask[i]] 
    return filtered_queries

def create_query_set(book, model, size=50, threshold=0.5):
    page_num = book.n_pages
    transcript_2_page_id = {}
    duplicate_transcript = set()

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

    filtered_queries = filter_query(all_transcripts, model, threshold)
    sampled_queries = random.choices(filtered_queries, k=size)

    return sampled_queries, transcript_2_page_id

def evaluate_query_set(book, model, method, mocr=None, size=50, threshold=0.5):
    if not mocr:
        mocr = MangaOcr()

    ds = MangaDataset(preprocess=None, book=book, text_annotations=True, frame_annotations=True)
    all_predicted_transcript = get_ocr_dataset(ds, mocr, method)

    sampled_queries, transcript_2_page_id = create_query_set(book, model, size, threshold)

    accuracy = 0
    for query in tqdm(sampled_queries, desc="Search Query", total=len(sampled_queries)):
        correct_page_id = transcript_2_page_id[query]
        predicted_page_id, _ = search(query, all_predicted_transcript, model)
        accuracy += predicted_page_id == correct_page_id

    accuracy /= len(sampled_queries)
    return accuracy


if __name__=="__main__":
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    # demo: search a query in a book
    doll_gun = Book(title="DollGun")
    ds = MangaDataset(preprocess=None, book=[doll_gun], text_annotations=True, frame_annotations=True)
    mocr = MangaOcr()
    page_ocr_all_transcript = get_ocr_dataset(ds, mocr, method="entire_page")
    gt_bb_ocr_all_transcript = get_ocr_dataset(ds, mocr, method="ground_truth_bb")

    page_idx, selected_transcript = search("撃て！ 殺虫剤をくれてやれ くっそーー！", page_ocr_all_transcript, model)
    print(f"Entire Page OCR Result\npage: {page_idx}\ntranscript: {selected_transcript}")

    page_idx, selected_transcript = search("撃て！ 殺虫剤をくれてやれ くっそーー！", gt_bb_ocr_all_transcript, model)
    print(f"Entire Page OCR Result\npage: {page_idx}\ntranscript: {selected_transcript}")

    # demo: sample five books and evaluate the accuracy on its queryset
    all_books = Manga109Dataset(manga109_root_dir=MANGA109_ROOT).books
    sampled_books = random.sample(all_books, 5)
    for book_title in sampled_books:
        book = Book(title=book_title)
        gt_bb_accuracy = evaluate_query_set(book, model, "ground_truth_bb", mocr, size=50)
        baseline_accuracy = evaluate_query_set(book, model, "entire_page", mocr, size=50)
        print(f"book: {book_title} baseline_accuracy: {baseline_accuracy} gt_bb_accuracy: {gt_bb_accuracy}")