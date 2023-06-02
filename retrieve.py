from abc import ABC, abstractmethod
from model.model import DetectionModel
from PIL.Image import Image
from typing import Optional, TypedDict
from manga109utils import BoundingBoxDict, Book
from constant import TEXT_LABEL, FRAME_LABEL
from manga_ocr import MangaOcr
from tqdm import tqdm
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import json


class LabeledCrops(TypedDict):
    data: Image
    page: int
    location: BoundingBoxDict

class LabeledText(TypedDict):
    text: str
    page: int
    location: BoundingBoxDict

class Retriever(ABC):
    @abstractmethod
    def index(self, images: list[Image]) -> None:
        """
        Index the given images.
        """
        pass

    def index_book(self, book: Book, max_pages=None) -> None:
        # load images
        images = [page.get_image() for page in book.get_page_iter(max_pages)]
        self.index(images)

    @abstractmethod
    def query(self, query: str, top_k: int = -1) -> list[int]:
        """
        Return a list of page indices that match the query, ranked by relevance.
        """
        pass

    @abstractmethod
    def save_index(self, path: Path) -> None:
        """
        Save the index to the given path (a directory)
        """
        path.mkdir(parents=True, exist_ok=True)
        assert path.is_dir()
        pass

    @abstractmethod
    def load_index(self, path: Path) -> None:
        """
        Load the index from the given path.
        """
        path.mkdir(parents=True, exist_ok=True)
        assert path.is_dir()
        pass

class TextSearchEngine(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def index(self, texts: list[LabeledText]) -> None:
        """
        Index the given texts.
        """
        pass

    @abstractmethod
    def query(self, query: str, top_k: int = -1) -> list[tuple[LabeledText, float]]:
        """
        Return a list of top LabeledTexts.
        """
        pass

    @abstractmethod
    def save_index(self, path: Path) -> None:
        """
        Save the index to the given path (a directory)
        """
        assert path.is_dir()
        pass

    @abstractmethod
    def load_index(self, path: Path) -> None:
        """
        Load the index from the given path.
        """
        assert path.is_dir()
        pass

class SentBertJapanese(TextSearchEngine):
    # pooling and encode function adapted from https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2
    def __init__(self, model_name_or_path = "sonoisa/sentence-bert-base-ja-mean-tokens-v2", device=None) -> None:
        super().__init__()
        from transformers import BertJapaneseTokenizer, BertModel
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def _encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings) # (n, 768)
    
    @torch.no_grad()
    def index(self, texts: list[LabeledText]) -> None:
        self.texts = texts
        self.text_embeddings = self._encode([text["text"] for text in texts])
        # normalize
        self.text_embeddings = torch.nn.functional.normalize(self.text_embeddings, p=2, dim=1)

    @torch.no_grad()
    def query(self, query: str, top_k: int = -1) -> list[tuple[LabeledText, float]]:
        query_embedding = self._encode([query])[0]
        # normalize
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0).unsqueeze(1)

        # compute cosine similarity
        cos_sim = torch.mm(self.text_embeddings, query_embedding)
        cos_sim = cos_sim.squeeze(1)
        # sort by cosine similarity
        if top_k == -1:
            top_k = len(self.texts)
        top_results, top_results_idx = torch.topk(cos_sim, k=top_k)
        top_results = top_results.tolist()
        top_results_idx = top_results_idx.tolist()
        
        return [(self.texts[idx], score) for idx, score in zip(top_results_idx, top_results)]

    def save_index(self, path: Path) -> None:
        super().save_index(path)
        torch.save(self.text_embeddings, path / "text_embeddings.pt")
        with open(path / "texts.json", "w") as f:
            json.dump(self.texts, f, indent=2, ensure_ascii=False)

    def load_index(self, path: Path) -> None:
        super().load_index(path)
        self.text_embeddings = torch.load(path / "text_embeddings.pt")
        with open(path / "texts.json", "r") as f:
            self.texts = json.load(f)


class SentBertMultilingual(TextSearchEngine):
    def __init__(self, model_name_or_path="sentence-transformers/distiluse-base-multilingual-cased-v2") -> None:
        
        super().__init__()
        self.model = SentenceTransformer(model_name_or_path)

    def index(self, texts: list[LabeledText]) -> None:
        self.texts = texts
        self.text_embeddings = self.model.encode([text["text"] for text in texts], convert_to_numpy=False, convert_to_tensor=True)

    def query(self, query: str, top_k: int = -1) -> list[tuple[LabeledText, float]]:
        query_embedding = self.model.encode(query, convert_to_numpy=False)
        # compute cosine similarity
        if top_k == -1:
            top_k = len(self.texts)
        cos_sim = util.cos_sim(self.text_embeddings, query_embedding).squeeze(1)
        top_results, top_results_idx = torch.topk(cos_sim, k=top_k)
        top_results = top_results.tolist()
        top_results_idx = top_results_idx.tolist()

        return [(self.texts[idx], score) for idx, score in zip(top_results_idx, top_results)]
    
    def save_index(self, path: Path) -> None:
        super().save_index(path)
        torch.save(self.text_embeddings, path / "text_embeddings.pt")
        with open(path / "texts.json", "w") as f:
            json.dump(self.texts, f, indent=2, ensure_ascii=False)

    def load_index(self, path: Path) -> None:
        super().load_index(path)
        self.text_embeddings = torch.load(path / "text_embeddings.pt")
        with open(path / "texts.json", "r") as f:
            self.texts = json.load(f)

class TextSearchMixin:
    def set_text_search_engine(self, text_search_engine: TextSearchEngine) -> None:
        self.text_search_engine = text_search_engine

    def query_plus(self, query: str, top_k: int = -1) -> list[tuple[LabeledText, float]]:
        return self.text_search_engine.query(query, top_k=top_k)        
    
    def query(self, query: str, top_k: int = -1) -> list[int]:
        results = self.query_plus(query, top_k=top_k)
        return [result[0]["page"] for result in results]
    
    def save_index(self, path: Path) -> None:
        super().save_index(path)
        self.text_search_engine.save_index(path)

    def load_index(self, path: Path) -> None:
        super().load_index(path)
        self.text_search_engine.load_index(path)

class EndToEndTranscriptRetriever(TextSearchMixin, Retriever):
    def __init__(self, detection_model: DetectionModel, manga_ocr_model: Optional[MangaOcr]=None, text_search_engine: Optional[TextSearchEngine]=None) -> None:
        super().__init__()

        self.detection_model = detection_model
        if not manga_ocr_model:
            manga_ocr_model = MangaOcr()
        self.manga_ocr_model = manga_ocr_model
        if not text_search_engine:
            text_search_engine = SentBertJapanese()
        self.set_text_search_engine(text_search_engine)

    def _get_crops(self, images: list[Image]) -> list[LabeledCrops]:
        bounding_boxes_raw = self.detection_model.predict(images)
        crops: list[LabeledCrops] = []
        for page in range(len(images)):
            bounding_boxes = bounding_boxes_raw[page]
            for bounding_box, score in bounding_boxes:
                if bounding_box.bbtype != TEXT_LABEL:
                    continue
                data = images[page].crop((bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax))
                crops.append({
                    "data": data,
                    "page": page,
                    "location": bounding_box.to_dict(),
                    "bb_score": score,
                })

        return crops
    
    def index(self, images: list[Image]) -> None:
        print("Detecting text regions...")
        crops = self._get_crops(images)
        # for each crop, get the text using mangaOCR
        labeled_texts: list[LabeledText] = []
        for crop in tqdm(crops, desc="OCR"):
            text = self.manga_ocr_model(crop["data"])
            labeled_texts.append({
                "text": text,
                "page": crop["page"],
                "location": crop["location"],
                "bb_score": crop["bb_score"],
            })
        # index the texts
        print("Indexing texts...")
        self.text_search_engine.index(labeled_texts)

class GoldBoundingBoxTextRetriever(TextSearchMixin, Retriever):
    def __init__(self, manga_ocr_model: Optional[MangaOcr]=None, text_search_engine: Optional[TextSearchEngine]=None) -> None:
        super().__init__()

        if not manga_ocr_model:
            manga_ocr_model = MangaOcr()
        self.manga_ocr_model = manga_ocr_model
        if not text_search_engine:
            text_search_engine = SentBertJapanese()
        self.set_text_search_engine(text_search_engine)

    def index(self, images: list[Image]) -> None:
        raise NotImplementedError("GoldBoundingBoxRetriever only works with Manga109 dataset.")
    
    def _get_crops(self, book: Book, max_pages=None) -> list[LabeledCrops]:
        crops: list[LabeledCrops] = []
        for page in tqdm(book.get_page_iter(max_pages=max_pages), desc="Cropping"):
            bbs = page.get_bbs()["text"]
            page_img = page.get_image()
            for bb in bbs:
                data = page_img.crop((bb.xmin, bb.ymin, bb.xmax, bb.ymax))
                crops.append({
                    "data": data,
                    "page": page.page_index,
                    "location": bb.to_dict(),
                    "bb_score": 1.0,
                })
        return crops

    def index_book(self, book: Book, max_pages=None) -> None:
        crops = self._get_crops(book, max_pages=max_pages)
        # for each crop, get the text using mangaOCR
        labeled_texts: list[LabeledText] = []
        for crop in tqdm(crops, desc="OCR"):
            text = self.manga_ocr_model(crop["data"])
            labeled_texts.append({
                "text": text,
                "page": crop["page"],
                "location": crop["location"],
                "bb_score": crop["bb_score"],
            })
        # index the texts
        print("Indexing texts...")
        self.text_search_engine.index(labeled_texts)


class GoldTextRetriever(TextSearchMixin, Retriever):
    def __init__(self, text_search_engine: Optional[TextSearchEngine]=None) -> None:
        super().__init__()
        if not text_search_engine:
            text_search_engine = SentBertJapanese()
        self.set_text_search_engine(text_search_engine)

    def index(self, images: list[Image]) -> None:
        raise NotImplementedError("GoldTextRetriever only works with Manga109 dataset.")
    
    def index_book(self, book: Book, max_pages=None) -> None:
        labeled_texts: list[LabeledText] = []
        for page in tqdm(book.get_page_iter(max_pages=max_pages)):
            bbs = page.get_bbs()["text"]
            for bb in bbs:
                labeled_texts.append({
                    "text": bb.text,
                    "page": page.page_index,
                    "location": bb.to_dict(),
                    "bb_score": 1.0,
                })
        print("Indexing texts...")
        self.text_search_engine.index(labeled_texts)

class LabeledScene(TypedDict):
    page: int
    location: BoundingBoxDict
    bb_score: float

class SceneSearchMixin(ABC):
    def set_encoders(self, image_encoder: SentenceTransformer, text_encoder: SentenceTransformer) -> None:
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def set_scenes(self, scenes: list[LabeledCrops], batch_size: int) -> None:
        all_embeddings = []
        iterator = range(0, len(scenes), batch_size)
        for batch_idx in tqdm(iterator, desc="Encoding scenes"):
            batch = scenes[batch_idx:batch_idx + batch_size]
            batch_embeddings = self.image_encoder.encode([scene["data"] for scene in batch], convert_to_numpy=False, convert_to_tensor=True)
            all_embeddings.extend(batch_embeddings)
        self.scene_embeddings = torch.stack(all_embeddings) # (n, 768)
        self.scenes = []
        for scene in scenes:
            self.scenes.append({
                "page": scene["page"],
                "location": scene["location"],
                "bb_score": scene["bb_score"],
            })

    def query_plus(self, scene_description: str, top_k: int = -1) -> list[tuple[dict, float]]:
        query_embedding = self.text_encoder.encode(scene_description, convert_to_numpy=False)
        # compute cosine similarity
        if top_k == -1:
            top_k = self.scene_embeddings.shape[0]
        cos_sim = util.cos_sim(self.scene_embeddings, query_embedding).squeeze(1)
        top_results, top_results_idx = torch.topk(cos_sim, k=top_k)
        top_results = top_results.tolist()
        top_results_idx = top_results_idx.tolist()
        return [(self.scenes[idx], score) for idx, score in zip(top_results_idx, top_results)]

    def query(self, scene_description: str, top_k: int = -1) -> list[int]:
        query_results = self.query_plus(scene_description, top_k=top_k)
        return [result[0]["page"] for result in query_results]
    
    def save_index(self, path: Path) -> None:
        super().save_index(path)
        with open(path / "scenes.json", "w") as f:
            json.dump(self.scenes, f, indent=2, ensure_ascii=False)
        torch.save(self.scene_embeddings, path / "scene_embeddings.pt")

    def load_index(self, path: Path) -> None:
        super().load_index(path)
        with open(path / "scenes.json", "r") as f:
            self.scenes = json.load(f)
        self.scene_embeddings = torch.load(path / "scene_embeddings.pt")

class EndToEndSceneRetriever(SceneSearchMixin, Retriever):
    def __init__(self, detection_model: DetectionModel, image_encoder=None, text_encoder=None, batch_size=8) -> None:
        super().__init__()

        self.detection_model = detection_model
        if not image_encoder:
            image_encoder = SentenceTransformer('clip-ViT-B-32')
        if not text_encoder:
            text_encoder = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        self.set_encoders(image_encoder, text_encoder)
        self.batch_size = batch_size

    def _get_crops(self, images: list[Image]) -> list[LabeledCrops]:
        bounding_boxes_raw = self.detection_model.predict(images)
        crops: list[LabeledCrops] = []
        for page in range(len(images)):
            bounding_boxes = bounding_boxes_raw[page]
            for bounding_box, score in bounding_boxes:
                if bounding_box.bbtype != FRAME_LABEL:
                    continue
                data = images[page].crop((bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax))
                crops.append({
                    "data": data,
                    "page": page,
                    "location": bounding_box.to_dict(),
                    "bb_score": score,
                })

        return crops
    
    def index(self, images: list[Image]) -> None:
        scenes = self._get_crops(images)
        self.set_scenes(scenes, batch_size=self.batch_size)

class GoldBoundingBoxSceneRetriever(SceneSearchMixin, Retriever):
    def __init__(self, image_encoder=None, text_encoder=None, batch_size=8) -> None:
        super().__init__()
        if not image_encoder:
            image_encoder = SentenceTransformer('clip-ViT-B-32')
        if not text_encoder:
            text_encoder = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        self.set_encoders(image_encoder, text_encoder)
        self.batch_size = batch_size

    def _get_crops(self, book: Book) -> list[LabeledCrops]:
        crops: list[LabeledCrops] = []
        for page in tqdm(book.get_page_iter(), desc="Get frames"):
            bounding_boxes = page.get_bbs()["frame"]
            img = page.get_image()
            for bounding_box in bounding_boxes:
                data = img.crop((bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax))
                crops.append({
                    "data": data,
                    "page": page.page_index,
                    "location": bounding_box.to_dict(),
                    "bb_score": 1.0,
                })

        return crops
    
    def index(self, images: list[Image]) -> None:
        raise NotImplementedError("GoldBoundingBoxSceneRetriever only works with Manga109 books")
    
    def index_book(self, book: Book, max_pages=None) -> None:
        scenes = self._get_crops(book)
        self.set_scenes(scenes, batch_size=self.batch_size)
    
