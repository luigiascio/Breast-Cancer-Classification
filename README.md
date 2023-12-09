# Breast-Cancer-Classification
The project aims to classify breast cancer tumors as either malignant or benign based on various features. Leveraging machine learning techniques, it provides a predictive system that can assist in the early detection of breast cancer.  

## 1. Import the Dependencies
Ensure you have the necessary dependencies installed before running the code:

```python
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## 2. Data Collection & Processing
The dataset consists of various features related to breast cancer, including mean radius, mean texture, mean perimeter, and more. The 'label' column represents the target variable, with values 0 and 1 indicating Malignant and Benign cases, respectively.
