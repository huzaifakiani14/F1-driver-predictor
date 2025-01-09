# F1-driver-predictor
**Project Overview**

This project aims to predict the finishing positions of all Formula 1 drivers in a race using a neural network model. The goal is to develop a model that accurately forecasts driver performance and provides insights into the factors influencing their race outcomes. The project leverages TensorFlow and Keras for building and training the neural network, with SMOTE used to address class imbalance and feature engineering to enhance accuracy.

## Project Structure**

**1. Data Exploration and Preprocessing:**

*   **Dataset Examination:** The project uses the "Formula 1 World Championship (1950-2020)" dataset from Kaggle, containing information about races, results, drivers, and constructors. We begin by understanding the dataset structure, identifying relevant features, and checking for missing values.
*   **Feature Engineering:** New features are derived from existing ones to potentially improve model performance. This includes calculating driver-constructor points, rolling average points, and grid start vs. final position differences.
*   **Data Transformation:** Categorical features like driver and constructor IDs are encoded using Label Encoding. The 'fastestLapTime' is converted to seconds for numerical analysis.
*   **Handling Missing Values:** Missing values in the dataset are addressed by imputation, filling them with appropriate values (e.g., 0 for numerical features).
*   **Balancing the Dataset:** SMOTE (Synthetic Minority Over-sampling Technique) is applied to handle class imbalance, ensuring the model doesn't bias towards drivers with more race data.
*   **Feature Scaling:** Numerical features are scaled using StandardScaler to ensure consistent scaling across all variables.

**2. Model Building and Training:**

*   **Neural Network Architecture:** A sequential neural network model is built using TensorFlow and Keras. The model consists of dense layers with ReLU activation functions, dropout layers for regularization, and batch normalization for improved training stability.
*   **Hyperparameter Tuning:** The model's hyperparameters (e.g., number of layers, neurons per layer, dropout rates) are carefully tuned to optimize performance.
*   **Train-Test Split:** The dataset is split into training and testing sets to evaluate the model’s generalization capabilities.

**3. Model Evaluation and Interpretation:**

*   **Performance Metrics:** The model's performance is evaluated using accuracy and the F1-score, which are suitable metrics for multi-class classification problems with potential class imbalance.
*   **Confusion Matrix:** A confusion matrix is generated to visualize the model's predictions and identify areas where it might be making errors. (Optional)
*   **Feature Importance Analysis:** Techniques like permutation feature importance can be used to assess the relative importance of different features in the model's predictions. (Optional)

## Results

The model achieved the following performance on the test set:

*   **Test Accuracy:** 85.81%
*   **Test F1 Score:** 0.8519

This demonstrates the model's capability to forecast driver performance across various race conditions with a reasonable degree of accuracy and completeness.
