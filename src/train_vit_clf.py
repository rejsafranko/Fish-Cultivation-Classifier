import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from argparse import ArgumentParser
from model.vitlightning import ViTLightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_id", type=str)
    return parser.parse_args()


def create_dataloaders_and_mappings(data_path):
    dataset = load_dataset("imagefolder", data_dir=args.data_path)
    splits = dataset["train"].train_test_split(test_size=0.1)
    dataset["train"] = splits["train"]
    dataset["val"] = splits["test"]

    id2label = {
        id: label for id, label in enumerate(dataset["train"].features["label"].names)
    }
    label2id = {label: id for id, label in id2label.items()}

    train_dataloader = DataLoader(dataset["train"], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset["val"], collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset["test"], collate_fn=collate_fn)

    dataloaders = {}
    dataloaders["train"] = train_dataloader
    dataloaders["val"] = val_dataloader
    dataloaders["test"] = test_dataloader

    return dataloaders, id2label, label2id


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main(args):
    dataloaders, id2label, label2id = create_dataloaders_and_mappings(args.data_path)
    num_labels = len(id2label)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=3, strict=False, verbose=False, mode="min"
    )

    model = ViTLightningModule(
        args.model_id, num_labels, id2label, label2id, dataloaders
    )

    # model = ViTLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

    trainer = Trainer(
        gpus=1, callbacks=early_stop_callback
    )  # default_root_dir="some/path/" za Colab

    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    args = parse_args()
    main(args)
