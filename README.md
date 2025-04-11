

# Health Insurance Claim Prediction

This project predicts health insurance claim costs based on various user inputs such as age, sex, weight, smoker status, BMI, blood pressure, and more. The model is built using a **Random Forest Regressor** with **GridSearchCV** for hyperparameter tuning, and the app uses **Streamlit** to provide an interactive interface for users.

## Contents

1. **Model Creation and Training**
2. **Streamlit Application**
3. **Saving and Loading the Model**
4. **Evaluation Metrics**
5. **Installation and Setup**
6. **Model Accuracy**
7. **Future Improvements**

## 1. Model Creation and Training

The model is trained using the dataset `HI_data.csv`, which contains features such as:

- Age
- Sex
- Weight
- Number of Dependents
- Smoker status
- BMI
- Blood Pressure
- Diabetes
- Regular Exercise
- Claim Cost (Target variable)

### Training Process:
1. Data cleaning (removing irrelevant columns and handling missing values).
2. Label encoding for categorical features (`sex`, `smoker`, `diabetes`, and `regular_ex`).
3. Splitting the dataset into training and testing sets.
4. Imputation of missing values using the `SimpleImputer`.
5. Training a **Random Forest Regressor** with **GridSearchCV** for hyperparameter tuning.
6. Saving the trained model and imputer using **joblib**.

```python
joblib.dump(model, 'trained_model.pkl')  # Save the trained model
joblib.dump(imputer, 'imputer.pkl')  # Save the imputer
```

## 2. Streamlit Application

The **Streamlit** application (`app.py`) allows users to input various features via an interactive form and get a prediction for the claim cost.

### Input Features:
- **Age**
- **Sex**
- **Weight (kg)**
- **Number of Dependents**
- **Smoker (yes/no)**
- **BMI**
- **Blood Pressure (mmHg)**
- **Diabetes (yes/no)**
- **Regular Exercise (yes/no)**

### Functionality:
1. The user inputs data through the web interface.
2. The data is processed, including encoding categorical variables (e.g., sex, smoker, diabetes, regular exercise) and imputation of missing values.
3. The model predicts the claim cost based on the input data.
4. The prediction result is displayed on the webpage.

## 3. Saving and Loading the Model

- **Model (`trained_model.pkl`)**: The trained **Random Forest Regressor** model used for predicting claim costs.
- **Imputer (`imputer.pkl`)**: The imputer used to handle missing values during both training and prediction.

Both the model and the imputer are loaded into the Streamlit app using the following code:

```python
model = joblib.load('trained_model.pkl')  
imputer = joblib.load('imputer.pkl')
```

## 4. Evaluation Metrics

The model’s performance is evaluated using the following metrics:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-Squared (R2)**

Example code for evaluating the model:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")
```

## 5. Installation and Setup

To run this application, follow the steps below:

### Prerequisites:

1. **Python 3.x** - Make sure you have Python 3.7+ installed.
2. **Required Libraries**:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `scikit-learn`
   - `joblib`
   - `streamlit`

### Install Required Libraries:

You can install the required libraries using `pip`. Run the following command to install all dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn joblib streamlit
```

### 1. Train the Model:

First, run the code to train the model and save the model and imputer files (`trained_model.pkl` and `imputer.pkl`):

```bash
python model.py
```

This will generate the necessary files (`trained_model.pkl` and `imputer.pkl`).

### 2. Run the Streamlit Application:

Now, run the Streamlit application using the following command:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your default browser (usually at `http://localhost:8501`).

### 3. Input Data and Get Predictions:

On the web interface, input the required information (age, sex, weight, etc.), and click the "Predict Claim Cost" button. The app will display the predicted claim cost based on the model's prediction.

## 6. Model Accuracy

The model has an **R-squared (R²)** value of approximately **96%** after hyperparameter tuning using **GridSearchCV**. This indicates that the model can explain 96% of the variance in the target variable (claim cost) based on the features provided. This is a significant improvement over the initial performance, which had an R² value of around 71%.

### How to Improve Model Accuracy:
1. **Feature Engineering**:
   - Consider adding more features or creating interactions between existing features to capture non-linear relationships.
   - Additional customer-related data (e.g., medical history, lifestyle factors) could enhance the model's performance.
   
2. **Model Selection**:
   - Test different models (e.g., Gradient Boosting, XGBoost, etc.) to compare performance.
   - Consider ensemble techniques for further improvement.
   
3. **Data Quality**:
   - **Handling Missing Data**: Improve the way missing data is handled, possibly by exploring other imputation methods or using models that can handle missing values internally.
   - **Outlier Detection**: Investigate outliers that could be affecting the model's performance and decide whether to remove or treat them.




## Troubleshooting

- **Missing Model or Imputer Files**: Ensure that `trained_model.pkl` and `imputer.pkl` exist in the same directory as `app.py` or adjust the paths accordingly.
- **Streamlit Not Installed**: If Streamlit is not installed, use the following command to install it:
  
  ```bash
  pip install streamlit
  ```


## 📬 Contact

Built with ❤️ by D. Yuva Shankar Narayana
[LinkedIn(Yuva Shankar Narayana )](https://www.linkedin.com/in/yuva-shankar-narayana-16b09a314) |
---


