from abc import ABC, abstractmethod
from PIL.Image import Image
from manga109utils import BoundingBox
from transformers import PreTrainedModel, TrainingArguments, Trainer
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from dataset import MangaDataset, build_collate_fn, ImageProcessor
from dataclasses import dataclass
import torch

@dataclass
class TrainConfig:
    output_dir: str = "checkpoints"
    num_train_epochs: int = 10
    save_steps: int = 1000
    logging_steps: int = 50
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    learning_rate: float = 1e-4
    batch_size: int = 8
    seed: int = 42

class DetectionModel(ABC):

    def __init__(self, name_or_path=None) -> None:
        super().__init__()
        if name_or_path is None:
            name_or_path = self._default_pretrained_name_or_path()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.build_model(name_or_path).to(self.device)
        self.image_process = self.build_image_processor(name_or_path)
        self.collate_fn = build_collate_fn(self.image_process)

    def predict(self, images: list[Image], batch_size = 4) -> list[list[tuple[BoundingBox, float]]]:
        self.model.eval()
        target_sizes = [(image.height, image.width) for image in images]
        target_sizes_chunks = [target_sizes[i:i + batch_size] for i in range(0, len(target_sizes), batch_size)]
        dataloader = torch.utils.data.DataLoader(
            images,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.image_process
        )
        result_tensor = []
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
                result_tensor.append(outputs)

        result = []
        for outputs, target_sizes_chunk in zip(result_tensor, target_sizes_chunks):
            result += self.image_process.post_process_to_bounding_box_list(outputs, target_sizes_chunk)
        return result

    @abstractmethod
    def _default_pretrained_name_or_path(self) -> str:
        """
        Return the default pretrained name or path.
        """
        pass

    @abstractmethod
    def build_model(self, name_or_path) -> PreTrainedModel:
        """
        Return a pretrained model used for training.
        """
        pass

    @abstractmethod
    def build_image_processor(self, name_or_path) -> ImageProcessor:
        """
        Return a function that preprocesses a list of images, and optionally a list of bounding boxes as COCO format.
        """
        pass

    def train(self, config: TrainConfig, dataset: MangaDataset, resume_from_checkpoint: bool = False):

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_train_epochs,
            fp16=False,
            save_steps=config.save_steps,
            logging_steps=config.logging_steps,
            do_eval=config.do_eval,
            evaluation_strategy=config.evaluation_strategy,
            eval_steps=config.eval_steps,
            learning_rate=config.learning_rate,
            save_total_limit=4,
            remove_unused_columns=False,
            report_to="wandb",
            seed=config.seed,
            data_seed=config.seed)

        train_dataset, val_dataset, test_dataset = dataset.train_val_test_split(train_size=0.8, val_size=0.1, seed=config.seed)

        opt = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        self.model.train()
        # TODO: override Trainer.evaluate to inject custom metrics        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            optimizers=(opt, None), # linear scheduler with warmup
            data_collator=dataset.collate_fn,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint) 
        trainer.save_model()
        trainer.log_metrics( "eval", self.evaluate(val_dataset))
        trainer.log_metrics( "test", self.evaluate(test_dataset))

    def evaluate(self, val_dataset: MangaDataset):
        
        def collate_fn(batch):
            images = [item["image"] for item in batch]
            annotations = [item["target"]["annotations"] for item in batch]
            target = []
            for annotation in annotations:
                boxes = []
                labels = torch.tensor([a["category_id"] for a in annotation])
                for a in annotation:
                    xmin, ymin, width, height = a["bbox"]
                    boxes.append(torch.tensor([xmin, ymin, xmin+width, ymin+height])) # need bb in format (xyxy) for map evaluation
                stacked_boxes = torch.stack(boxes) if boxes!=[] else torch.tensor([])
                target.append({"boxes": stacked_boxes.to(self.device), "labels":labels.to(self.device)})
            return images, target
        
        metric = MeanAveragePrecision()
        loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluation"):
                images, target = batch
                processed_batch = self.image_process(images).to(self.device)
                outputs = self.model(**processed_batch)
                target_sizes = [(image.height, image.width) for image in images]
                preds = self.image_process.processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)   
                metric.update(preds, target)
        return metric.compute()
    