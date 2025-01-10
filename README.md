# Phishing Detection using ML Techniques

This repository contains an implementation of a phishing detection system using machine learning techniques. The dataset used consists of URLs classified as 'bad' (phishing) or 'good' (legitimate). The project preprocesses URLs by converting them into numerical representations, transforms them into 2D arrays for image-like inputs, and trains a Convolutional Neural Network (CNN) to classify them.

## Project Overview
- **Programming Language**: Python
- **Frameworks and Libraries**: TensorFlow, NumPy, pandas, scikit-learn, Keras
- **Dataset**: [Kaggle Phishing Site URLs Dataset](https://www.kaggle.com/code/mohamedgalal1/phishing-detection-using-ml-techniques/data)
- **Objective**: Detect phishing URLs using machine learning techniques.

## Files in this Repository
- `592ML-keras-phishing-encode.ipynb`: The main notebook containing the implementation of the project.

## Key Features
1. **Data Preprocessing**:
    - Replaces labels ('bad' -> 0, 'good' -> 1).
    - Removes duplicate URLs.
    - Converts characters in URLs to integers using the Keras `Tokenizer` class.
    - Transforms the numerical sequences into 2D arrays for image-like input.
2. **Model Architecture**:
    - A CNN model built using TensorFlow's Keras Functional API.
    - The model is optimized using the Adam optimizer and evaluated using binary cross-entropy loss.
3. **Training and Validation**:
    - Splits the dataset into training and testing sets.
    - Uses 20% of the training data as a validation set.
    - Trains the model for 10 epochs with a batch size of 64.

## Dataset Information
- The dataset contains 507,195 URLs, with approximately 28% labeled as 'bad' (phishing).
- The dataset can be downloaded from the [Kaggle Dataset page](https://www.kaggle.com/code/mohamedgalal1/phishing-detection-using-ml-techniques/data).

## Steps to Run the Project
1. **Setup**:
    - Install the required Python packages: `TensorFlow`, `NumPy`, `pandas`, `scikit-learn`, and `Keras`.
    - Mount your Google Drive if using Google Colab.
2. **Data Preparation**:
    - Load the dataset and preprocess it using the provided code.
3. **Model Training**:
    - Define and compile the CNN model.
    - Train the model using the `fit` method.
4. **Evaluation**:
    - Use the testing set to evaluate the model's performance.

## Output
The model outputs the training and validation accuracy for each epoch. Additionally, the testing set's performance metrics are printed after evaluation.

## Future Enhancements
- Experiment with different model architectures and hyperparameters.
- Use more advanced text preprocessing techniques to improve performance.
- Extend the project to classify URLs into multiple categories, if applicable.

## Acknowledgements
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Kaggle Phishing Site URLs Dataset](https://www.kaggle.com/code/mohamedgalal1/phishing-detection-using-ml-techniques/data)

