# Predicting Diabetes Progression with Regression Models

A Python machine learning project that compares linear regression and deep neural network regression models to predict a continuous diabetes progression score.

The goal was to connect Python, machine learning, and health-related data in a way that was more meaningful than using a random dataset.

## Project Overview

This project uses the scikit-learn diabetes regression dataset to predict a numeric diabetes progression score. The model does not diagnose diabetes. It predicts a continuous value from health-related patient measurements.

The project compares four models:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression: BMI only | 52.26 | 63.73 | 0.233 |
| Linear Regression: all inputs | 42.79 | 53.85 | 0.453 |
| DNN Regression: BMI only | 51.95 | 65.97 | 0.179 |
| DNN Regression: full model | 43.46 | 54.12 | 0.447 |

The best model in this run was the all-input linear regression model. The full DNN model was close, but it did not beat the simpler baseline model.

## Why This Project

Diabetes has affected people in my family, so this topic felt more meaningful to me than choosing a random dataset. I wanted to see how machine learning can be used to study health-related data and predict a numeric outcome.

This project helped me better understand:

- Regression modeling
- Train and test splitting
- Feature comparison
- Model evaluation
- TensorFlow/Keras workflow
- Why a more complex model is not always better

## Research Question

Can a regression model use health-related features to estimate diabetes progression?

A second question was:

Which model performs better on the test set, linear regression or deep neural network regression?

## Dataset

Dataset used: `sklearn.datasets.load_diabetes`

Dataset details:

- 442 patient records
- 10 health-related input features
- 1 numeric target value
- Target: diabetes progression score
- Task type: regression

Input features:

- Age
- Sex
- BMI
- Blood pressure
- Six blood serum measurements

## Regression vs Classification

This project uses regression because the model predicts a number.

Regression example:

```text
Predict diabetes progression score: 151
```

Classification would predict a category.

Classification example:

```text
Diabetes or no diabetes
```

This project is not a yes or no diagnosis model.

## Project Workflow

1. Load the diabetes dataset
2. Check for missing values
3. Clean the data
4. Split the data into training and testing sets
5. Inspect feature relationships
6. Separate features and labels
7. Create TensorFlow normalization layers
8. Train baseline linear regression models
9. Train TensorFlow/Keras DNN models
10. Compare model performance
11. Make predictions
12. Save graphs, metrics, predictions, and the trained DNN model

## Models Used

### Linear Regression

Linear regression was used as a baseline model. It helps show whether a simple model can perform well before using a deeper neural network.

Two linear regression models were tested:

- BMI only
- All 10 input features

### Deep Neural Network Regression

The DNN models were built with TensorFlow/Keras.

Full DNN structure:

```text
Input: 10 health-related features
Normalization layer
Dense layer: 64 units, ReLU
Dense layer: 64 units, ReLU
Output: 1 numeric prediction
```

## Key Terms

| Term | Meaning |
|---|---|
| MAE | Mean Absolute Error. Average amount the prediction was wrong |
| RMSE | Root Mean Squared Error. Shows larger mistakes more strongly |
| R² | R-squared. How much of the data pattern the model explained |
| DNN | Deep Neural Network |
| Regression | Predicting a number |
| Feature | Input variable used by the model |
| Target | Number being predicted |
| Epoch | One full training pass through the data |
| Dense 64 ReLU | A neural network layer with 64 learning units using ReLU activation |
| BMI | Body Mass Index, a number based on height and weight |

## Main Findings

The all-input linear regression model performed best in this run.

The results showed:

- BMI was useful, but limited alone
- Using all 10 features improved prediction
- The DNN model was close to the best result
- The more complex model did not automatically perform better
- The model found patterns, but predictions still had errors

## Example Prediction Output

The model compares actual progression scores against predicted scores.

Example:

| Actual | Predicted | Error |
|---:|---:|---:|
| 219.0 | 139.78 | -79.22 |
| 70.0 | 193.75 | 123.75 |
| 94.0 | 98.25 | 4.25 |

A smaller error means the prediction was closer to the real value.

## Visual Outputs

The Python script creates these output files:

- `target_distribution.png`
- `feature_correlation.png`
- `bmi_linear.png`
- `model_architecture.png`
- `project8_predicted_vs_actual.png`
- `project8_dnn_loss_curve.png`
- `project8_model_comparison.png`
- `project8_model_metrics.csv`
- `project8_predictions.csv`
- `project8_diabetes_dnn_model.keras`

## How to Run

### Option 1: Google Colab

1. Open Google Colab
2. Upload the Python file
3. Run this command:

```python
!python project8_diabetes_tensorflow_regression_FULL.py
```

### Option 2: Local Python

Install the required packages:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

Run the script:

```bash
python project8_diabetes_tensorflow_regression_FULL.py
```

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- TensorFlow
- Scikit-learn

## Responsible Use

This model is for learning and analysis only.

It should not be used to make health decisions. A real clinical tool would need a larger dataset, medical review, outside validation, and clearer feature context.

## References

- Centers for Disease Control and Prevention. National Diabetes Statistics Report
- scikit-learn. `load_diabetes` dataset documentation
- TensorFlow. Basic regression tutorial
- scikit-learn. Train/test split documentation
- scikit-learn. Regression metrics documentation

## Author

Nicholas Krot  
ENGR-125  
Python and TensorFlow/Keras Regression Project
