# Men vs. Women Image Classification

## Overview
This project aims to classify images of men and women using deep learning techniques. We leverage the power of convolutional neural networks by utilizing the pre-trained VGG16 architecture, originally trained on the ImageNet dataset, to extract meaningful features from the images. The model is then fine-tuned on a custom dataset to enhance its accuracy in distinguishing between male and female images.

## Prerequisites

Before running this project, ensure you have the following installed:

*   Python 3.x
*   TensorFlow (>= 2.0)
*   Keras
*   NumPy
*   pandas
*   Matplotlib
*   Kaggle API (optional, for downloading the dataset)

You can install these dependencies using pip:

```bash
pip install tensorflow numpy matplotlib pandas
```

## Downloading the Dataset
To download the dataset in your Google Colab environment, run the following command:
```bash
!kaggle datasets download -d saadpd/menwomen-classification
```
After downloading, unzip the dataset and organize it into appropriate folders for training, validation, and testing.


## Installation
Clone the repository or download the notebook.
Upload your kaggle.json file to authenticate your Kaggle account. 

## Results
The training process includes monitoring both training and validation accuracy and loss over epochs. The resulting plots provide a visual representation of the model’s performance and convergence during training. These plots help in diagnosing potential overfitting or underfitting issues and in fine-tuning hyperparameters for improved results.
