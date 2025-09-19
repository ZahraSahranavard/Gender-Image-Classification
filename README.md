# 📑 Gender Image Classification

![Bank Transaction Analysis](https://github.com/ZahraSahranavard/Gender-Image-Classification/blob/main/Image/Men-Women-Classification.jpg)

## 🔹 Project Overview
This project aims to classify images of men and women using Deep Learning techniques.
By leveraging the power of Convolutional Neural Networks (CNNs) and the pre-trained VGG16 architecture (originally trained on the ImageNet dataset), the model extracts meaningful features from input images. The model is then fine-tuned on a custom dataset to achieve high accuracy in distinguishing between male and female images.

## 📂 Dataset

The project uses the **Men-Women Classification** dataset from [Kaggle](https://www.kaggle.com/datasets/saadpd/menwomen-classification).
To download the dataset in your Google Colab environment, run the following command:
```bash
!kaggle datasets download -d saadpd/menwomen-classification
```

```bash
!unzip menwomen-classification.zip -d ./data
```

After downloading, unzip the dataset and organize it into appropriate folders for training, validation, and testing.

The dataset is organized into three subsets:

- **Training Set**: 1598 images  
- **Validation Set**: 400 images  
- **Test Set**: 800 images  

###  Folder structure:
```
men_vs_women_small/
├── train/
│ ├── men/
│ └── women/
├── validation/
│ ├── men/
│ └── women/
└── test/
├── men/
└── women/
```

## ⚙️ Prerequisites

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

## 🏗️ Model Design

The model is based on **VGG16** (pre-trained on ImageNet).
It consists of the following components:

- Frozen convolutional base for feature extraction

- Dense layer (256 units, ReLU) + Dropout (0.5)

- Final sigmoid output for binary classification
  
## 🔄 Training Phases
### 1. Feature Extraction

- VGG16 convolutional base frozen

- Only top classifier layers trained

### 2. Fine-Tuning

- Last 4 convolutional blocks of VGG16 unfrozen

- Optimizer: Adam (learning_rate = 1e-5)

- Callbacks used to save the best-performing model

## 📌 Key Learnings

- Transfer learning with pre-trained CNNs significantly improves accuracy on small datasets.

- Fine-tuning only a subset of convolutional layers avoids overfitting.

- Data augmentation is essential for robustness.

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📬 Contact
Developed by [Zahra Sahranavard](https://www.linkedin.com/in/zahra-sahranavard)  
For inquiries: zahra.sahranavard7622@iau.ir



