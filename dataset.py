from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Union
from torch.utils.data import Dataset
from tqdm import tqdm

from manga109utils import Manga109Dataset, Page, BoundingBox, Book
import order_estimator
from PIL.Image import Image
import numpy as np
from constant import FRAME_LABEL, TEXT_LABEL

MANGA109_ROOT = "datasets/Manga109_released_2021_12_30"
order_estimator.interception_ratio_threshold = 0.25

def _get_ordered_annotations(panels: list[BoundingBox], page_width: int) -> list[BoundingBox]:
    est = order_estimator.BoxOrderEstimator(panels, pagewidth = page_width, initial_cut_option="two-page")
    return est.ordered_bbs

class ProcessedPage:
    def __init__(self, page: Page) -> None:
        self.page = page
        bbs = page.get_bbs()
        frame_annotations = _get_ordered_annotations(bbs["frame"], page.pagedims[0])
        self.frame_bbs = [bounding_box_to_coco(bb, FRAME_LABEL) for bb in frame_annotations]
        self.text_bbs = [bounding_box_to_coco(bb, TEXT_LABEL) for bb in bbs["text"]]

class COCOFormat(TypedDict):
    category_id: int
    bbox: list[float] # [xmin, ymin, width, height]
    area: float
    iscrowd: int

class COCOTarget(TypedDict):
    image_id: int
    annotations: list[COCOFormat]

class DatasetItem(TypedDict):
    image: Image
    target: COCOTarget

def bounding_box_to_coco(bb: BoundingBox, category_id: int) -> COCOFormat:
    return {
        "category_id": category_id,
        "bbox": [bb.xmin, bb.ymin, bb.width, bb.height],
        "area": bb.width * bb.height,
        "iscrowd": 0,
    }

def coco_to_bounding_box(coco: COCOFormat) -> BoundingBox:
    xmin, ymin, width, height = coco["bbox"]
    xmax, ymax = xmin + width, ymin + height
    return BoundingBox(xmin, xmax, ymin, ymax)

class ImageProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, images: list[Image], targets: Optional[list[COCOTarget]] = None) -> dict:
        return self.preprocess(images, targets)

    @abstractmethod
    def preprocess(self, images: list[Image], targets: Optional[list[COCOTarget]] = None) -> dict:
        pass

    @abstractmethod
    def post_process_to_bounding_box_list(self, output, target_sizes: list[tuple]) -> list[list[tuple[BoundingBox, float]]]:
        """
        Converts the output of the model to a list of bounding boxes for each image.
        `target_sizes`: a list of tuples (height, width) of the target images.
        Returns a list of lists of tuples (bounding box, score).
        """
        pass

class MangaDataset(Dataset):
    """
    A dataset of manga pages, annotated with bounding boxes for frames and text, using the COCO format.
    """
    def __init__(self, preprocess: ImageProcessor, book = None, root = MANGA109_ROOT, text_annotations = True, frame_annotations = True, exclude_books: Optional[list[str]] = None):
        super().__init__()
        dataset = Manga109Dataset(root)
        if book is None:
            books = dataset.get_book_iter()
        elif isinstance(book, Book):
            books = [book]
        elif isinstance(book, list):
            books = book
        else:
            raise TypeError(f"book must be Book or list[Book] or None, not {type(book)}")
        self.pages: list[ProcessedPage] = []
        exclude_books = set(exclude_books) if exclude_books is not None else set()
        if not (isinstance(book, list) and len(books) == 0):
            for book in tqdm(books, desc="Loading Annotations", total=109):
                if book.title in exclude_books:
                    print(f"Skipping {book.title} as they are used for end-to-end evaluation.")
                    continue
                for page in book.get_page_iter():
                    self.pages.append(ProcessedPage(page))

        self.need_text_annotations = text_annotations
        self.need_frame_annotations = frame_annotations
        self.transform = preprocess
        self.collate_fn = build_collate_fn(preprocess)

    def __len__(self):
        return len(self.pages)
    
    def __getitem__(self, index: int) -> DatasetItem:
        image = self.pages[index].page.get_image()
        annotations = []
        if self.need_text_annotations:
            annotations += self.pages[index].text_bbs
        if self.need_frame_annotations:
            annotations += self.pages[index].frame_bbs
        return {
            "image": image,
            "target": {
                "image_id": index,
                "annotations": annotations,
            }
        }
    
    def gather(self, indices: list[int]):
        pages = [self.pages[i] for i in indices]
        dataset = MangaDataset(self.transform, book=[], text_annotations=self.need_text_annotations, frame_annotations=self.need_frame_annotations)
        dataset.pages = pages
        return dataset

    def train_val_test_split(self, train_size=0.8, val_size=0.1, seed=42):
        import numpy as np
        indices = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_indices = indices[:int(len(self) * train_size)]
        val_indices = indices[int(len(self) * train_size):int(len(self) * (train_size + val_size))]
        test_indices = indices[int(len(self) * (train_size + val_size)):]
        return self.gather(train_indices), self.gather(val_indices), self.gather(test_indices)
    
def build_collate_fn(transform: ImageProcessor, with_labels = True):
    if with_labels:
        def collate_fn(batch):
            images = [item["image"] for item in batch]
            targets = [item["target"] for item in batch]
            batch = transform(images, targets)
            return batch
        return collate_fn
    else:
        def collate_fn(batch):
            images = [item["image"] for item in batch]
            batch = transform(images, None)
        return collate_fn
    