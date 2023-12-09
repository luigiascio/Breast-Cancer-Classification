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
```

## 2. Data Collection & Processing
The dataset consists of various features related to breast cancer, including mean radius, mean texture, mean perimeter, and more ([see variables description](./variables_description.md)). The 'label' column represents the target variable, with values 0 and 1 indicating Malignant and Benign cases, respectively.  

## 3. Statistical Measures about the Data

Performing initial statistical analysis on the dataset, including mean, standard deviation, minimum, maximum, median values and correlation:

|                    | mean radius | mean texture | mean perimeter | mean area | mean smoothness | mean compactness | mean concavity | mean concave points | mean symmetry | mean fractal dimension | ... | worst texture | worst perimeter | worst area | worst smoothness | worst compactness | worst concavity | worst concave points | worst symmetry | worst fractal dimension | label |
|---------------------|-------------|--------------|-----------------|-----------|-----------------|------------------|----------------|---------------------|---------------|------------------------|-----|---------------|------------------|------------|-------------------|-------------------|-----------------|------------------------|----------------|--------------------------|-------|
| **count**           | 569         | 569          | 569             | 569       | 569             | 569              | 569            | 569                 | 569           | 569                    | ... | 569           | 569              | 569        | 569               | 569               | 569             | 569                    | 569            | 569                      | 569   |
| **mean**            | 14.127      | 19.290       | 91.969          | 654.889   | 0.096           | 0.104            | 0.089          | 0.049               | 0.181         | 0.063                 | ... | 25.677        | 107.261          | 880.583    | 0.132             | 0.254             | 0.272           | 0.115                  | 0.290          | 0.084                    | 0.627 |
| **std**             | 3.524       | 4.301        | 24.299          | 351.914   | 0.014           | 0.053            | 0.080          | 0.039               | 0.027         | 0.007                 | ... | 6.146         | 33.603           | 569.357    | 0.023             | 0.157             | 0.209           | 0.066                  | 0.062          | 0.018                    | 0.484 |
| **min**             | 6.981       | 9.710        | 43.790          | 143.500   | 0.053           | 0.019            | 0.000          | 0.000               | 0.106         | 0.050                 | ... | 12.020        | 50.410           | 185.200   | 0.071             | 0.027             | 0.000           | 0.000                  | 0.157          | 0.055                    | 0.000 |
| **25%**             | 11.700      | 16.170       | 75.170          | 420.300   | 0.086           | 0.065            | 0.030          | 0.020               | 0.162         | 0.058                 | ... | 21.080        | 84.110           | 515.300   | 0.117             | 0.147             | 0.115           | 0.065                  | 0.250          | 0.071                    | 0.000 |
| **50%** (median)    | 13.370      | 18.840       | 86.240          | 551.100   | 0.096           | 0.093            | 0.062          | 0.034               | 0.179         | 0.062                 | ... | 25.410        | 97.660           | 686.500   | 0.131             | 0.212             | 0.227           | 0.100                  | 0.282          | 0.080                    | 1.000 |
| **75%**             | 15.780      | 21.800       | 104.100         | 782.700   | 0.105           | 0.130            | 0.131          | 0.074               | 0.196         | 0.066                 | ... | 29.720        | 125.400          | 1084.000  | 0.146             | 0.339             | 0.383           | 0.161                  | 0.318          | 0.092                    | 1.000 |
| **max**             | 28.110      | 39.280       | 188.500         | 2501.000  | 0.163           | 0.345            | 0.427          | 0.201               | 0.304         | 0.097                 | ... | 49.540        | 251.200          | 4254.000  | 0.223             | 1.058             | 1.252           | 0.291                  | 0.664          | 0.208                    | 


## 4. Machine Learning Classification
In this section, we leverage machine learning to classify breast cancer using Logistic Regression. Here's an overview:

### 4.1 Data Preparation
* Features and Target Separation: The dataset's features, including mean radius, texture, and more, are separated from the binary target variable (0 for malignant, 1 for benign).

* Data Splitting: The dataset is split into training and testing sets for model evaluation.
  
```python
X = data_frame.drop(columns = 'label', axis = 1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)
```
  
### 4.2 Model Training and Evaluation
* Logistic Regression Model: A Logistic Regression model is trained on the training data, addressing potential convergence warnings.

* Model Accuracy: The model's accuracy is assessed on both training (93.8%) and testing (93.9%) datasets, indicating robust generalization.

### 4.3 Predictive System
* Example Prediction: The trained model predicts the class of breast cancer for a sample, providing practical insights.
  
```python
#taking a data from the datase. it should be 1
input_data = (13.64,16.34,87.21,571.8,0.07685,0.06059,0.01857,0.01723,0.1353,0.05953,0.1872,0.9234,1.449,14.55,0.004477,0.01177,0.01079,0.007956,0.01325,0.002551,14.67,23.19,96.08,656.7,0.1089,0.1582,0.105,0.08586,0.2346,0.08025)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one datapoint
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshape)
print(prediction)

if (prediction[0] == 0):
    print('The Breast cancer is Malignant')
    
else:
    print('The Breast Cancer is Benign')
```
OUTPUT: [1]  
'The Breast Cancer is Benign'    
  
Note:
Due to a warning about feature names, users are encouraged to refer to scikit-learn documentation for potential adjustments.
This comprehensive machine learning pipeline aims to effectively classify breast cancer, offering a practical tool for medical diagnosis.
