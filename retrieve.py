from abc import ABC, abstractmethod
from model.model import DetectionModel
from PIL.Image import Image
from typing import Optional, TypedDict
from manga109utils import BoundingBox
from constant import TEXT_LABEL
from manga_ocr import MangaOcr
from tqdm import tqdm
import torch
import numpy as np

class LabeledCrops(TypedDict):
    data: Image
    page: int
    location: BoundingBox

class LabeledText(TypedDict):
    text: str
    page: int
    location: BoundingBox

class TranscriptRetriever(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def query(self, dialogue: str, top_k: int = -1) -> list[LabeledText]:
        """
        Return a list of page indices of the retrieved transcripts.
        """
        pass


class SceneRetriever(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def query(self, scene_description: str, top_k: int = -1) -> list[int]:
        """
        Return a list of page indices of the retrieved scenes.
        """
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

class SentBertJapanese(TextSearchEngine):
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
        top_results = torch.topk(cos_sim, k=top_k)
        top_results = top_results[0].tolist()
        top_results = [(self.texts[i], score) for i, score in enumerate(top_results)]
        top_results = sorted(top_results, key=lambda x: x[1], reverse=True)
        return top_results
    

class SentBertMultilingual(TextSearchEngine):
    def __init__(self, model_name_or_path="sentence-transformers/distiluse-base-multilingual-cased-v2") -> None:
        super().__init__()
        



    



class EndToEndTranscriptRetriever(TranscriptRetriever):
    def __init__(self, detection_model: DetectionModel, manga_ocr_model: Optional[MangaOcr]=None) -> None:
        # from sentence_transformers import SentenceTransformer
        super().__init__()

        self.detection_model = detection_model
        if not manga_ocr_model:
            manga_ocr_model = MangaOcr()
        self.manga_ocr_model = manga_ocr_model
        # self.embedding_model = SentenceTransformer(embedding_model_path)

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
                    "location": bounding_box
                })

        return crops
    
    def index(self, images: list[Image]) -> None:
        print("Detecting text regions...")
        crops = self._get_crops(images)
        # for each crop, get the text using mangaOCR
        self.labeled_texts: list[LabeledText] = []
        for crop in tqdm(crops, desc="OCR"):
            text = self.manga_ocr_model(crop["data"])
            self.labeled_texts.append({
                "text": text,
                "page": crop["page"],
                "location": crop["location"]
            })

        
        
        

        