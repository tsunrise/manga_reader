import torch
from tqdm import tqdm

from dataset import MangaDataset
from manga109utils import Book
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

if __name__=="__main__":
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    doll_gun = Book(title="DollGun")
    ds = MangaDataset(book=[doll_gun], text_annotations=True, frame_annotations=True)

    mocr = MangaOcr()
    page_ocr_all_transcript = get_ocr_dataset(ds, mocr, method="entire_page")
    gt_bb_ocr_all_transcript = get_ocr_dataset(ds, mocr, method="ground_truth_bb")

    page_idx, selected_transcript = search("撃て！ 殺虫剤をくれてやれ くっそーー！", page_ocr_all_transcript, model)
    print(f"Entire Page OCR Result\npage: {page_idx}\ntranscript: {selected_transcript}")

    page_idx, selected_transcript = search("撃て！ 殺虫剤をくれてやれ くっそーー！", gt_bb_ocr_all_transcript, model)
    print(f"Entire Page OCR Result\npage: {page_idx}\ntranscript: {selected_transcript}")