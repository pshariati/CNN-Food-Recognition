# CNN-Food-Recognition

## Food Image Classification using CNN

This project applies a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify food images into three categories: Bread, Soup, and Vegetable-Fruit. The goal is to demonstrate how deep learning techniques can automate image classification for visual datasets where manual labeling is not feasible.

## Project Overview

Image classification has become significantly more accessible with deep learning and the availability of large datasets. CNNs are among the most widely used architectures for computer vision tasks due to their ability to automatically extract hierarchical features from images.

This project simulates a scenario inspired by Clicks, a stock photography platform that receives thousands of food-related uploads daily. Given the high volume of images, manual labeling is time-consuming and error-prone. A CNN-based classifier provides a scalable, automated way to tag and organize such images.

## Objective

* Develop a CNN to classify images into three categories: Bread, Soup, and Vegetable-Fruit.
* Apply convolutional, pooling, and dense layers to extract and learn visual features.
* Evaluate model performance on unseen test data and visualize key metrics.

## Model Architecture

* Input Layer: 150×150×3 RGB images
* Convolutional Layers: 4 blocks with 256 → 128 → 64 → 32 filters (kernel size = 3×3 or 5×5)
* Activation Function: ReLU
* Pooling Layers: MaxPooling2D with stride (2,2)
* Dropout Layers: 25% dropout to prevent overfitting
* Dense Layers: Two fully connected layers (64 and 32 neurons)
* Output Layer: 3 neurons with Softmax activation (multi-class classification)
* Optimizer: Adam (learning rate = 0.001)
* Loss Function: Categorical Cross-Entropy

## Results

* Test Accuracy: 79%
* Validation Accuracy: stabilized after 23 epochs using EarlyStopping
* Key Techniques: MaxPooling, Dropout, and ModelCheckpoint imrpoved generalizaiton

Visual results — including training/validation accuracy curves and the confusion matrix — are available in the notebook.

## Dataset

The dataset contains labeled images of three food categories organized into Training and Testing folders:

```text
Food_Data/
│
├── Training/
│ ├── Bread/
│ ├── Soup/
│ └── Vegetable-Fruit/
│
└── Testing/
├── Bread/
├── Soup/
└── Vegetable-Fruit/
```

Each image is resized to 150 × 150 pixels during preprocessing.

## Tools and Libraries

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib, Seaborn

## How to run

1. Clone this repository:

git clone https://github.com/pshariati/CNN-Food-Recognition.git

cd CNN-Food-Recognition

2. Install dependencies:

pip install -r requirements.txt

3. Open the Jupyter notebook:

jupyter notebook CNN_Food_Classification.ipynb

4. Run all cells to train and evaluate the model.

## Contact

Developed by **Pejmon Shariati** 

[GitHub](https://github.com/pshariati) | [LinkedIn](https://www.linkedin.com/in/pejmonshariati)

