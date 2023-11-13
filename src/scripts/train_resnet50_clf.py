import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.utils import *
from tensorflow.keras import *
from sklearn.model_selection import train_test_split
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets, DatasetDict
import numpy as np
from sklearn import metrics


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

    X_train = np.array(dataset["train"]["image"])
    X_train = np.array([tf.convert_to_tensor(np.array(image)) for image in X_train])
    y_train = to_categorical(np.array(dataset["train"]["label"]), num_classes=3)

    X_valid = np.array(dataset["val"]["image"])
    X_valid = np.array([tf.convert_to_tensor(np.array(image)) for image in X_valid])
    y_valid = to_categorical(np.array(dataset["val"]["label"]), num_classes=3)

    X_test = np.array(dataset["test"]["image"])
    X_test = np.array([tf.convert_to_tensor(np.array(image)) for image in X_test])
    y_test = to_categorical(np.array(dataset["test"]["label"]), num_classes=3)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# Create model configuration.
def configure_model():
    base_model = ResNet50(
        input_shape=(512, 768, 3), weights="imagenet", include_top=False
    )
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(
        Dense(
            256,
            kernel_initializer="he_uniform",
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax", kernel_regularizer=regularizers.l2(0.001)))

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    print(model.summary())

    return model


def main(args):
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_dataset(args.data_dir)

    # Create a model.
    model = configure_model()

    # Define callbacks.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="/model/resnet50/",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=6
    )

    # Train the model.
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=4,
        steps_per_epoch=50,
        epochs=30,
        validation_data=(X_valid, y_valid),
        validation_steps=25,
        callbacks=[es, checkpoint_callback],
        verbose=1,
    )

    # Save the model.
    model.save("./model/resnet50")
