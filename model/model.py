from abc import ABC, abstractmethod
from PIL.Image import Image
from manga109utils import BoundingBox
from transformers import PreTrainedModel, TrainingArguments, Trainer
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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

    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.build_model().to(self.device)
        self.image_process = self.build_image_processor()
        self.collate_fn = build_collate_fn(self.image_process)

    def predict(self, images: list[Image], target_sizes: list[tuple]) -> list[list[BoundingBox, float]]:
        """
        `target_sizes`: a list of tuples (height, width) of the target images.
        """
        self.model.eval()
        batch = self.image_process(images).to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch)
            result = self.image_process.post_process_to_bounding_box_list(outputs, target_sizes)
        return result

    @abstractmethod
    def build_model(self) -> PreTrainedModel:
        """
        Return a pretrained model used for training.
        """
        pass

    @abstractmethod
    def build_image_processor(self) -> ImageProcessor:
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

        train_dataset, val_dataset = dataset.train_test_split(train_size=0.8, seed=config.seed)

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
    
    def evaluate(self, val_dataset: MangaDataset):
        self.model.eval()
        loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=val_dataset.collate_fn)
        metric = MeanAveragePrecision()
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(**batch)
                outputs.logits = outputs.logits.cpu()
                outputs.pred_boxes = outputs.pred_boxes.cpu()
                # target_sizes = [image["size"] for image in batch["labels"]]
                preds = self.image_process.processor.post_process_object_detection(outputs, threshold=0.5)
                target = [{"boxes": image["boxes"], "labels":image["class_labels"]} for image in batch["labels"]]
                metric.update(preds, target)
        return metric.compute()
