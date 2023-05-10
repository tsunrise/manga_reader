from dataclasses import dataclass
from transformers import TrainingArguments, Trainer
from constant import DETR_PRETRAINED
from dataset_legacy import MangaDataset
from model import get_model

@dataclass
class TrainConfig:
    output_dir: str = "checkpoints"
    num_train_epochs: int = 10
    save_steps: int = 200
    logging_steps: int = 50
    eval_steps: int = 100
    learning_rate: float = 1e-4
    seed: int = 42

    @staticmethod
    def from_toml(path: str):
        import toml
        with open(path, "r") as f:
            return TrainConfig(**(toml.load(f)["train"]))


def train(config: TrainConfig):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_eval_batch_size=8,
        num_train_epochs=config.num_train_epochs,
        fp16=False,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        learning_rate=config.learning_rate,
        save_total_limit=2,
        remove_unused_columns=False)

    model = get_model().to(device)

    dataset = MangaDataset(format_as=DETR_PRETRAINED)
    dataset.load_from_disk()

    train_dataset, val_dataset = dataset.train_test_split(train_size=0.8, seed=config.seed)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=dataset.collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    args = parser.parse_args()
    train(TrainConfig.from_toml(args.config))

if __name__ == "__main__":
    import os
    main()

    
