from typing import List
from torch.utils.data import Dataset
from manga109api import Parser
from dataclasses import dataclass
from pathlib import Path
from transformers import DetrImageProcessor
from tqdm import tqdm
from PIL import Image
MANGA109_ROOT = "datasets/Manga109_released_2021_12_30"

@dataclass
class Page:
    idx: int
    image: Path
    text_boxes: List[dict] # fields: @xmin, @xmax, @ymin, @ymax

class MangaDataset(Dataset):
    def __init__(self, format_as="facebook/detr-resnet-50"):
        super().__init__()
        api = Parser(MANGA109_ROOT)
        self.transform = DetrImageProcessor.from_pretrained(format_as)
        self.pages: List[Page] = []
        # load image paths and page annotations
        for book in tqdm(api.books, desc="Loading Annotations"):
            annotation = api.get_annotation(book)
            for page, val in enumerate(annotation["page"]):
                text_boxes = self._get_text_boxes(val)
                image_path = api.img_path(book, page)
                self.pages.append(Page(len(self.pages), image_path, text_boxes))

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        # to COCO format
        annotations = []
        for box in self.pages[index].text_boxes:
            xmin, ymin = box["@xmin"], box["@ymin"]
            width, height = box["@xmax"] - xmin, box["@ymax"] - ymin
            area = width * height
            annotations.append({
                "category_id": 0,
                "bbox": [xmin, ymin, width, height],
                "area": area,
                "iscrowd": 0,
            })
        
        return {
            "image": Image.open(self.pages[index].image),
            "target": {
                # coco format
                "image_id": self.pages[index].idx,
                "annotations": annotations,
            }
        }


    def collate_fn(self, batch):
        # batch transform images
        images = [x["image"] for x in batch]
        targets = [x["target"] for x in batch]
        
        self.transform.preprocess(images, targets, return_tensors="pt")

    @staticmethod
    def _get_text_boxes(annotation: dict):
        stack = [annotation]
        text_boxes = []
        while stack:
            frame = stack.pop()
            if "frame" in frame:
                stack.extend(frame["frame"])
            if "text" in frame:
                text_boxes.extend(frame["text"])
        return text_boxes