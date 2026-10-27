# Bike Sharing Demand Prediction

## 📌 Project Overview

This project uses machine learning regression techniques to predict hourly bike rental demand based on weather conditions, time-related features, and calendar information.

The project explores the dataset, performs exploratory data analysis (EDA), engineers time-based features, compares multiple regression models, and selects a Random Forest Regressor to predict bike rental counts.

The final model generates predictions for the test dataset and saves them in a submission-ready CSV file.

## 🎯 Objectives

- Explore bike rental patterns and identify demand trends.
- Analyze the relationships between rental demand, weather, and time.
- Engineer meaningful features from datetime information.
- Compare multiple machine learning regression algorithms.
- Evaluate model performance using RMSE and R².
- Analyze residuals and feature importance.
- Generate and validate predictions for the test dataset.

## 📂 Dataset

The project uses three CSV files:

| File | Description |
|---|---|
| `train.csv` | Historical data containing features and the target variable |
| `test.csv` | Data used to generate rental demand predictions |
| `sampleSubmission.csv` | Template defining the required submission format |

The training dataset contains **10,886 rows and 12 original columns**, while the test dataset contains **6,493 rows and 9 original columns**.

### Key Features

| Feature | Description |
|---|---|
| `datetime` | Date and time of the rental record |
| `season` | Season identifier |
| `holiday` | Indicates whether the day is a holiday |
| `workingday` | Indicates whether the day is a working day |
| `weather` | Weather condition category |
| `temp` | Temperature |
| `atemp` | Perceived temperature |
| `humidity` | Humidity level |
| `windspeed` | Wind speed |
| `casual` | Number of casual users |
| `registered` | Number of registered users |
| `count` | Total bike rentals — prediction target |

The `casual` and `registered` columns are excluded from model features because they are components of the target variable `count` and are not available in the test dataset.

## 🛠️ Technologies Used

- **Python** — programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computations and feature engineering
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing, regression models, and evaluation
- **Jupyter Notebook** — interactive development environment

## 🔍 Project Workflow

### 1. Data Loading and Exploration

The training, test, and sample submission files are loaded into Pandas DataFrames.

The initial analysis includes:

- Dataset dimensions and data types.
- Summary statistics.
- Missing-value checks.
- Average rental counts across seasons and hours.
- Correlation analysis between numerical features and rental demand.

No missing values are reported in the provided training or test datasets.

### 2. Feature Engineering

The `datetime` column is converted to a datetime format, and the following features are extracted:

- `year`
- `month`
- `day`
- `hour`
- `weekday`

To represent the cyclical nature of time, additional sine and cosine features are created:

- `hour_sin`
- `hour_cos`
- `month_sin`
- `month_cos`

These features help represent the relationship between adjacent hours and months, including the transition between the end and beginning of a cycle.

The final model uses **17 features**.

### 3. Exploratory Data Analysis and Visualization

The project uses visualizations to investigate:

- Distribution of bike rental counts.
- Average rentals across seasons.
- Average rentals under different weather conditions.
- Hourly rental patterns.
- Correlations among numerical variables.

The exploratory analysis shows that rental demand varies considerably throughout the day, with higher average demand during the late afternoon and around the morning commute.

### 4. Data Preprocessing

The target variable is defined as:

`count`

The features and target are separated before training.

The labeled training data is split into:

- **Training set:** 80% (8,708 rows)
- **Validation set:** 20% (2,178 rows)

A `StandardScaler` is fitted only on the training split and applied to the validation split for scale-sensitive models. Tree-based models use the original, unscaled features.

This approach helps avoid data leakage during feature scaling.

### 5. Model Training and Comparison

Four regression algorithms are evaluated:

- Linear Regression
- K-Nearest Neighbors (KNN) Regressor
- Decision Tree Regressor
- Random Forest Regressor

Model performance is compared using Root Mean Squared Error (RMSE). A lower RMSE indicates more accurate predictions on the validation set.

## 📊 Model Performance

The following results were obtained on the validation set:

| Model | Validation RMSE |
|---|---:|
| Linear Regression | 125.6603 |
| KNN Regressor | 90.0984 |
| Decision Tree | 54.0917 |
| **Random Forest** | **38.7298** |

Random Forest achieves the lowest validation RMSE among the four evaluated models and is selected as the final model.

Its validation \(R^2\) score is **0.9546**, indicating that the model explains a large proportion of the variation in the validation target values.

### 6. Model Evaluation and Interpretation

The selected model is evaluated further through:

- Residuals versus predicted values.
- Distribution of validation residuals.
- Random Forest feature importance.

The leading feature importances are:

| Feature | Importance |
|---|---:|
| `hour` | 0.3794 |
| `hour_sin` | 0.1409 |
| `year` | 0.0876 |
| `workingday` | 0.0694 |
| `hour_cos` | 0.0654 |
| `temp` | 0.0588 |

The hour of the day is the most important feature in the fitted Random Forest model, followed by its cyclical encoding and the year.

Feature importance describes how the model uses features for prediction; it does not establish causation.

### 7. Final Model and Prediction

After model selection, a new Random Forest Regressor is trained on the complete labeled training dataset.

The final model uses:

- `n_estimators = 100`
- `random_state = 42`
- `n_jobs = -1`

Predictions are generated for the test dataset. Negative predicted counts are clipped to zero to ensure non-negative rental estimates.

The predictions are inserted into a copy of the sample submission template, preserving the original datetime order.

The final output is saved as:

`my_submission.csv`

The notebook also reloads the generated file and validates its shape, datetime order, non-missing predictions, and non-negative rental counts.

## 🚀 Getting Started

### Prerequisites

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run the Project

1. Clone or download the repository.
2. Place `train.csv`, `test.csv`, and `sampleSubmission.csv` in the project directory.
3. Open the Jupyter Notebook.
4. Run the notebook cells in order.
5. Find the generated predictions in `my_submission.csv`.

Start Jupyter Notebook with:

```bash
jupyter notebook
```

## 📁 Project Structure

```text
Bike-Sharing-Demand-Prediction/
│
├── train.csv
├── test.csv
├── sampleSubmission.csv
├── Machine Learning in Python Mini Project2.ipynb
├── my_submission.csv
└── README.md
```

*Adjust the notebook filename to match the actual filename in your repository. The submission CSV is generated when the notebook runs successfully.*

## 🔮 Future Improvements

- Use time-based validation to better reflect future rental demand forecasting.
- Tune Random Forest hyperparameters using cross-validation.
- Explore gradient boosting models such as XGBoost or HistGradientBoosting.
- Investigate additional weather and calendar interactions.
- Compare performance using additional metrics such as MAE and RMSLE.
- Analyze prediction errors across different hours, seasons, and weather conditions.
