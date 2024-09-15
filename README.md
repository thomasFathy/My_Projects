Diabetes Prediction using SVM
This project uses a Support Vector Machine (SVM) model to predict whether a patient suffers from diabetes based on several medical attributes. The goal is to classify patients as diabetic or non-diabetic based on the given features using machine learning techniques.

Table of Contents
Overview
Dataset
Model
Requirements
Installation
Usage
Results
Contributing
License


Overview
The project leverages an SVM classifier to predict diabetes in patients. The model is trained on a dataset of medical records that include features such as glucose levels, blood pressure, BMI, and more. By using the SVM algorithm, we aim to create a model that can accurately distinguish between diabetic and non-diabetic patients.

Dataset
We use the PIMA Indians Diabetes Dataset, which consists of the following features:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration
Blood Pressure: Diastolic blood pressure (mm Hg)
Skin Thickness: Triceps skinfold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age of the patient
Outcome: 0 for non-diabetic, 1 for diabetic (target variable)
The dataset is available here.

Model
The project employs a Support Vector Machine (SVM) classifier, which is a powerful supervised learning algorithm commonly used for classification tasks.

Kernel used: Radial Basis Function (RBF)
Optimization technique: GridSearchCV for hyperparameter tuning
The model is trained on 70% of the dataset and tested on the remaining 30%. The final performance is evaluated using accuracy, precision, recall, and F1-score.

Requirements
The project requires the following Python packages:

bash
Copy code
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook (optional)
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/thomasFathy/diabetes-prediction-svm.git
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Usage
To train the model and make predictions, run the Jupyter Notebook or the Python script:

bash
Copy code
python svm_diabetes_prediction.py
You can modify the dataset and parameters in the config section of the script.

Once the training is completed, the model will output the evaluation metrics and a confusion matrix.

Results
The model achieved an accuracy of approximately X% on the test dataset. The performance metrics are as follows:

Accuracy: X%
Precision: X%
Recall: X%
F1-score: X%
You can visualize the confusion matrix and ROC curve to better understand the model's performance.

Contributing
Feel free to submit a pull request or open an issue if you have any suggestions or improvements for this project.

License
This project is licensed under the MIT License - see the LICENSE file for details.
