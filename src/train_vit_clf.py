import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import (
    ViTForImageClassification,
    ViTFeatureExtractor,
    Trainer,
    TrainingArguments,
)
from argparse import ArgumentParser
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_id", type=str)
    return parser.parse_args()


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def create_dataloaders_and_mappings(data_path):
    dataset = load_dataset("imagefolder", data_dir=data_path)
    splits = dataset["train"].train_test_split(test_size=0.33)
    dataset["train"] = splits["train"]
    dataset["val"] = splits["test"]

    id2label = {
        id: label for id, label in enumerate(dataset["train"].features["label"].names)
    }

    label2id = {label: id for id, label in id2label.items()}

    return dataset, id2label, label2id


def compute_metrics(eval_pred):
    metric1 = load("accuracy")
    metric2 = load("precision")
    metric3 = load("recall")
    metric4 = load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric2.compute(
        predictions=predictions, references=labels, average="weighted"
    )["precision"]
    recall = metric3.compute(
        predictions=predictions, references=labels, average="weighted"
    )["recall"]
    f1 = metric4.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main(args):
    dataset, id2label, label2id = create_dataloaders_and_mappings(args.data_path)
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        args.model_id,
        do_resize=False,
        patch_size=64,
    )

    def transform(example_batch):
        inputs = feature_extractor(
            [x.convert("RGB") for x in example_batch["image"]], return_tensors="pt"
        )
        inputs["label"] = example_batch["label"]
        return inputs

    dataset = dataset.with_transform(transform)

    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path=args.model_id,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="../model/",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=6,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
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

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(dataset["test"])
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)
