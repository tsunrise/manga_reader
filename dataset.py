from typing import List
from torch.utils.data import Dataset
from manga109api import Parser
from dataclasses import dataclass
from pathlib import Path
from transformers import DetrImageProcessor
from tqdm import tqdm
from PIL import Image
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
MANGA109_ROOT = "datasets/Manga109_released_2021_12_30"

@dataclass
class Page:
    idx: int
    image: Path
    text_boxes: List[dict] # fields: @xmin, @xmax, @ymin, @ymax

class MangaDataset(Dataset):
    def __init__(self, format_as="facebook/detr-resnet-50"):
        super().__init__()
        self.base_model_name = format_as
        self.transform = DetrImageProcessor.from_pretrained(format_as)
        self.pages: List[Page] = []

    def load_from_disk(self):
        api = Parser(MANGA109_ROOT)
        # load image paths and page annotations
        for book in tqdm(api.books, desc="Loading Annotations"):
            annotation = api.get_annotation(book)
            for page, val in enumerate(annotation["page"]):
                text_boxes = self._get_text_boxes(val) # type: ignore
                image_path = Path(api.img_path(book, page))
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
        images = [x["image"] for x in batch]
        targets = [x["target"] for x in batch]
        
        # batch transform images, now bbox is normalized [x_center, y_center, width, height]
        return self.transform.preprocess(images, targets, return_tensors="pt") # type: ignore
 # type: ignore # type: ignore
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

    def post_process(self, batch, outputs: List[DetrObjectDetectionOutput], threshold=0.5):
        image_sizes = [x['size'] for x in batch.labels]
        outputs = self.transform.post_process_object_detection(outputs, threshold, target_sizes=image_sizes) # type: ignore

    def gather(self, indices):
        pages = [self.pages[i] for i in indices]
        dataset = MangaDataset(self.base_model_name)

        dataset.pages = pages
        return dataset

    def train_test_split(self, train_size=0.8, seed=42):
        import numpy as np
        indices = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_indices = indices[:int(len(self) * train_size)]
        test_indices = indices[int(len(self) * train_size):]
        return self.gather(train_indices), self.gather(test_indices)

    @property
    def id2label(self):
        return {0: "text"}

    @property
    def label2id(self):
        return {"text": 0}
