# Fake News Detection

This project implements a machine learning model to detect fake news articles. It uses text preprocessing techniques and a Logistic Regression model to classify news articles as real or fake.

## Features
- **Dataset Splitting:** The dataset is split into training and testing sets.
- **TF-IDF Vectorization:** Text data is transformed into numerical vectors using TF-IDF.
- **Model Training:** A Logistic Regression model is trained on the processed data.
- **Accuracy Evaluation:** Model performance is evaluated on both training and test datasets.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git

2. Navigate to the project directory:
```bash
cd fake-news-detection
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt

```
## How to Use
1. Open the Jupyter Notebook:
 ```
jupyter notebook "fake news detection.ipynb"
  ```
2.Run each cell in sequence to preprocess the data, train the model, and evaluate its performance.
3.Replace the dataset with your own if needed by following the input format.

## Code Overview
1. Imports
The project uses the following libraries:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import string

```
2. Data Splitting
The dataset is divided into training and testing sets:
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

```
3. Text Feature Extraction
TF-IDF Vectorization is applied to transform the text into numerical data:
```
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)

```
4. Model Training
The Logistic Regression model is trained using the TF-IDF features:
```
model = LogisticRegression()
model.fit(X_train_feature, Y_train)

```
5. Model Evaluation
Accuracy is calculated for both training and test datasets:
```
pred_train_data = model.predict(X_train_feature)
acc_train_data = accuracy_score(Y_train, pred_train_data)

pred_test_data = model.predict(X_test_feature)
acc_test_data = accuracy_score(Y_test, pred_test_data)

```
Results

-Training Accuracy: Achieved a high accuracy score on the training data.
-Testing Accuracy: Model performance is evaluated on unseen test data to check for generalization.
