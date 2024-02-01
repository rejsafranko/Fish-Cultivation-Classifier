import numpy as np
import torch
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    ViTForImageClassification,
    ViTFeatureExtractor,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# Utility function which loads the data splits, expands the train set with transformations and encodes the labels.
def prepare_dataset(data_dir):
    train_dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")
    traineval_dataset = train_dataset.train_test_split(test_size=0.33)
    test_dataset = load_dataset("imagefolder", data_dir=data_dir, split="test")

    dataset = DatasetDict(
        {
            "train": traineval_dataset["train"],
            "val": traineval_dataset["test"],
            "test": test_dataset,
        }
    )

    transform = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        ]
    )

    transformed_train_dataset = dataset["train"].map(
        lambda example: {
            "image": transform(example["image"]),
            "label": example["label"],
        }
    )

    dataset["train"] = concatenate_datasets(
        [transformed_train_dataset, dataset["train"]]
    )

    return dataset


# Utility function which padds the inputs.
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


# Utility function which defines evaluation metrics.
def compute_metrics(eval_pred: EvalPrediction):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return {
        "acc": accuracy_score(eval_pred.label_ids, preds),
        "f1": f1_score(eval_pred.label_ids, preds, average="weighted"),
        "precision": precision_score(eval_pred.label_ids, preds, average="weighted"),
        "recall": recall_score(eval_pred.label_ids, preds, average="weighted"),
    }


def main():
    # Configure the Feature Extractor.
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        do_resize=False,
        patch_size=64,
    )

    feature_extractor.image_mean = [0.485, 0.456, 0.406]  # ImageNet mean.
    feature_extractor.image_std = [0.229, 0.224, 0.225]  # ImageNet std.

    # Nested function for applying the Feature Extractor transforms.
    def transform(example_batch):
        inputs = feature_extractor(
            [x.convert("RGB") for x in example_batch["image"]], return_tensors="pt"
        )
        inputs["labels"] = example_batch["label"]
        return inputs

    dataset = prepare_dataset("../../data")
    dataset = dataset.with_transform(transform)

    # ID to label mappings and vice-versa.
    id2label = {
        id: label for id, label in enumerate(dataset["train"].features["label"].names)
    }
    label2id = {label: id for id, label in id2label.items()}

    # Configure the model.
    # Set interpolate_pos_encoding=True in the source code.
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.config.hidden_dropout_prob = 0.5

    # Prepare the Trainer.
    training_args = TrainingArguments(
        output_dir="../../model/vit",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        num_train_epochs=20,
        fp16=True,
        save_steps=60,
        eval_steps=60,
        warmup_steps=500,
        logging_steps=60,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=feature_extractor,
    )

    # Train and save the model.
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
