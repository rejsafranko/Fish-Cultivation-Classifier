import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import ViTForImageClassification
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import List, Callable, Optional


# Utility class for wrapping the Huggingface ViT so it can return outputs as Tensor.
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


# Utility function to reshape Huggingface ViT's activations for GradCAM algorithm.
def reshape_transform_vit_huggingface(x):
    activations = x[:, 1:, :]
    activations = activations.reshape(
        activations.shape[0], 32, 48, activations.shape[2]
    )
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations


# Utility function for category to index mapping.
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]


# Utility function to apply the GradCAM algorithm.
def run_grad_cam_on_image(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    targets_for_gradcam: List[Callable],
    reshape_transform: Optional[Callable],
    input_tensor: torch.nn.Module,
    input_image: Image,
    method: Callable = GradCAM,
):
    with method(
        model=HuggingfaceToTensorModelWrapper(model),
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    ) as cam:
        # Replicate the tensor for each of the categories we want to create Grad-CAM for.
        repeated_tensor = input_tensor[None, :].repeat(
            len(targets_for_gradcam), 1, 1, 1
        )

        batch_results = cam(input_tensor=repeated_tensor, targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(
                np.float32(input_image) / 255, grayscale_cam, use_rgb=True
            )
            visualization = cv2.resize(
                visualization,
                (visualization.shape[1] // 2, visualization.shape[0] // 2),
            )
            results.append(visualization)
        return np.hstack(results)


# Utility function to predict the top predicted categories.
def print_top_categories(model, img_tensor, top_k=1):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k:][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")


def main():
    # Load the pre-trained Vision Transformer model.
    model = ViTForImageClassification.from_pretrained("/model/vit")

    # Prepare GradCAM configurations.
    target_layer_dff = model.vit.layernorm
    target_layer_gradcam = model.vit.encoder.layer[-2].output
    targets_for_gradcam = [
        ClassifierOutputTarget(category_name_to_index(model, "divlje")),
        ClassifierOutputTarget(category_name_to_index(model, "tunakavez")),
        ClassifierOutputTarget(category_name_to_index(model, "uzgoj")),
    ]

    # Load the image and turn it to Tensor.
    image_path = "./data/test/divlje/divlje_test_2020_WS_001.JPG"  # Path to your problematic image
    image = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image)

    # Create Deep Feature Factorization visualization for the image.
    image = Image.fromarray(
        run_dff_on_image(
            model=model,
            target_layer=target_layer_dff,
            classifier=model.classifier,
            img_pil=image,
            img_tensor=image_tensor,
            reshape_transform=reshape_transform_vit_huggingface,
            n_components=3,
            top_k=3,
        )
    )

    plt.imshow(image)

    # Create GradCAM visualization for the image.
    image = Image.fromarray(
        run_grad_cam_on_image(
            model=model,
            target_layer=target_layer_gradcam,
            targets_for_gradcam=targets_for_gradcam,
            input_tensor=image_tensor,
            input_image=image,
            reshape_transform=reshape_transform_vit_huggingface,
        )
    )

    plt.imshow(image)

    # Print the predicted category.
    print_top_categories(model, image_tensor)


if __name__ == "__main__":
    main()
