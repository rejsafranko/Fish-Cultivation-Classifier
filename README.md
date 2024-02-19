# Fish Image Classification with Deep Models

<img alt="FER UniZG" src="https://github.com/rejsafranko/Fish-Image-Classification/blob/main/icon.jpg" height="50" width="100">

## Overview
This repository is part of a university seminar and focuses on classifying fish images using a private dataset from the Institute of Oceanography and Fisheries of Croatia. The project leverages advanced techniques, including data augmentation, a Vision Transformer (ViT), and a ResNet50 architecture to achieve accurate and robust classification results. Additionally, a Grad-CAM visualizer has been implemented for AI explainability, providing insights into the model's decision-making process.

## Features
Data Augmentation: Augmented the training dataset to enhance model generalization and improve performance.

Vision Transformer (ViT): Utilized a pre-trained Vision Transformer model for image classification, providing a novel approach compared to traditional convolutional neural networks.

ResNet50: Utilized a pre-trained ResNet50 model for comparison, showcasing the performance of different architectures on the given dataset.

Grad-CAM Visualizer: Integrated a Grad-CAM visualizer to interpret and visualize the model's attention, offering insights into the regions of the image influencing the classification decision.

## Project Structure

Explore the Jupyter notebooks or Python scripts to understand the implementation details and model experiments.

```notebooks/```: contains Jupyter notebooks meant for running the experiments on Google Colaboratory

```src/```: contains the script version of the experiments meant for running localy

```docs/```: contains the technical document describing and explaining the technologies and experiments

```models/```: meant for saving the fine-tuned models
