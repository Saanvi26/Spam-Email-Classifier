# Spam Detection Project

## Project Overview
This project implements a spam detection system using machine learning. The model is trained on email data to classify messages as either "spam" or "ham" (not spam) based on the content of the messages.

## Technologies Used
- **Python**: For data processing and model development.
- **Pandas**: For data manipulation and handling.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning algorithms and metrics.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: For feature extraction from the text data.
- **Logistic Regression**: As the classification model.

## Dataset
The dataset used for this project is a collection of email messages labeled as "spam" or "ham". The data is loaded from a CSV file (`mail_data.csv`) and preprocessed to handle any missing values.

## Model Training
The dataset is split into training and testing sets with an 80-20 split. TF-IDF vectorization is applied to convert the text data into numerical features. Logistic Regression is then used to train the model on the training data.

## Model Accuracy
The trained model achieves an accuracy of **96%** on the test data.

## How to Run the Project
1. Ensure you have the necessary Python packages installed:
   ```bash
   pip install numpy pandas scikit-learn
