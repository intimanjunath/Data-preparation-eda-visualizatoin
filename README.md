# AutoML Pipeline with AutoVIML

## Project Overview
This project builds a regression model to predict Airbnb listing prices using an automated machine learning (AutoML) approach powered by AutoVIML. The pipeline covers data preprocessing, model training, evaluation, and deployment readiness.
---

## Steps in the Pipeline

### 1. Data Preprocessing
- Removed irrelevant or low-information columns:
  - `id`, `anomaly_flag`, `dbscan_cluster`
- Converted categorical and object columns to appropriate types:
  - `neighbourhood`, `stay_duration_bucket`, `price_category`
- Filled missing values in numerical and categorical columns.

### 2. Model Training
- Leveraged `AutoVIML` for automated feature selection and model training.
- Optimized hyperparameters using randomized search (`hyper_param='RS'`).

### 3. Model Evaluation
- Evaluated the model on a test dataset using key metrics:
  - **R-squared (R²)**: Indicates the proportion of variance explained.
  - **Root Mean Squared Error (RMSE)**: Measures prediction error.
  - **Mean Absolute Error (MAE)**: Average absolute error between predicted and actual values.

### 4. Predictions on New Data
- Used the trained model to predict prices for unseen data.
- Saved the predictions for analysis.

### 5. Saving Artifacts
- Trained model saved as `trained_model.pkl`.
- Selected features stored in `selected_features.json`.
- Evaluation metrics saved in `evaluation_metrics.json`.

## Usage Instructions

### Preprocessing and Model Training
Ensure the dataset is cleaned and properly formatted, then run the pipeline:
```bash
python run_pipeline.py
```

---

# **AutoML Pipeline with AutoVIML for Time Series Forecasting**

## **Project Overview**
This project builds a regression model to forecast air passenger traffic using an automated machine learning (AutoML) approach powered by **AutoVIML**. The pipeline covers data preprocessing, model training, evaluation, and predictions on unseen data, with a focus on time-series forecasting.

---

## **Steps in the Pipeline**

### **1. Data Preprocessing**
- Extracted key time-based features:
  - `Year`, `Month_Num`, `DayOfWeek`
- Filled missing values:
  - Used median imputation for numerical columns.
  - Added missing flags for columns with missing values (`Year_Missing_Flag`, `Month_Num_Missing_Flag`, etc.).
- Removed irrelevant or low-information columns:
  - `is_anomaly`, `Rolling_Mean`, `Rolling_Std`, `Cluster`
- Scaled numeric features using Min-Max Scaling for better model performance.

### **2. Model Training**
- Leveraged **AutoVIML** for:
  - Feature selection: Selected `Year`, `Month_Num`, and `DayOfWeek` as key predictors.
  - Model training with hyperparameter optimization (`hyper_param='RS'` for Randomized Search).
  - Ensemble learning to combine predictions from multiple regressors (e.g., Adaboost, Bagging, KNN, etc.).

### **3. Model Evaluation**
- Evaluated the ensembled model using key metrics:
  - **R-squared (R²)**: Proportion of variance explained by the model.
  - **Root Mean Squared Error (RMSE)**: Measures prediction error.
  - **Mean Absolute Error (MAE)**: Average absolute error between predicted and actual values.
- Visualized the results with actual vs. predicted scatterplots.

### **4. Predictions on Test Data**
- Used the trained ensemble model to predict air passenger traffic for unseen data.
- Saved the predictions for analysis and visualization.

---

## **Key Files**
- **Pipeline Code**:
  - `run_pipeline.py`: Main script to preprocess data, train the model, and generate predictions.
- **Model Artifacts**:
  - `trained_model.pkl`: Saved trained ensemble model for deployment or future use.
  - `selected_features.json`: List of the best features chosen by AutoVIML.
  - `evaluation_metrics.json`: Evaluation metrics of the trained model (R², RMSE, MAE).
- **Processed Data**:
  - `train_modified.csv`: Transformed training data after preprocessing and feature selection.
  - `test_modified.csv`: Transformed test data with predictions from individual models and ensembles.
- **Predictions**:
  - `test_predictions.csv`: Final predicted values for the test dataset.
