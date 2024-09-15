Entity Extraction from Images - ML Hackathon
Overview

In the rapidly expanding digital marketplace, many products lack detailed textual descriptions. Accurate extraction of key product attributes like weight, volume, and dimensions directly from images is crucial for improving product listings and user experience. This project addresses this problem by developing a machine learning model to extract these attributes from product images.
Problem Statement

The goal is to build a model that can extract specific entity values from product images. These entities include attributes such as weight, volume, wattage, and dimensions, which are often displayed in images but may not be included in the textual descriptions of the products. The challenge involves:

    Image Processing: Extracting readable text from images where attributes are visually represented.
    Entity Recognition: Identifying and categorizing the extracted text to determine which attribute it represents and its value.
    Prediction and Formatting: Predicting and formatting the output to match specified formats and units.

Solution Approach
1. Data Preprocessing

Data Loading and Image Downloading

    Load Dataset: Read the training and test datasets to extract image links and entity labels.
    Download Images: Fetch images from URLs provided in the dataset and save them locally for processing.

Text Extraction and Cleaning

    OCR with Tesseract: Use Tesseract OCR to extract text from images. The extracted text is then cleaned to remove unwanted characters and white spaces.
    Preprocessing: Process and prepare the text for further analysis.

2. Feature Extraction

Optical Character Recognition (OCR)

    Tesseract OCR: Extracts raw text from images. Tesseract is a powerful open-source OCR tool that is used to convert image data into text.

Object Detection

    Faster R-CNN: Implement Faster R-CNN, a robust object detection algorithm, to identify relevant objects in the images. This step is crucial for localizing regions of interest where entity values might be present.

3. Model Building and Training

Text Classification Model

    Text Feature Extraction: Convert the cleaned text data into numerical features using tokenization and padding.
    Model Architecture: Use a Convolutional Neural Network (CNN) for text classification. The model learns to categorize text into predefined entity values.
    Training: Train the model using labeled data from train.csv, employing techniques like validation and hyperparameter tuning to optimize performance.

Combining Object Detection and OCR

    Region-Based Analysis: Use Faster R-CNN to detect objects and then apply OCR to specific regions within those objects. This combination allows for targeted extraction of text from regions where relevant information is likely to be found.

4. Prediction and Output Formatting

Prediction on Test Data

    Text Extraction: Extract and clean text from test images.
    Model Inference: Use the trained classification model to predict entity values based on extracted text.
    Output Formatting: Format predictions according to the specified guidelines, ensuring correct units and formatting.

Validation

    Sanity Check: Validate the output format to ensure it adheres to the required structure and conventions.

Unique Aspects of This Model

    Combination of OCR and Object Detection: This approach uniquely integrates OCR with object detection to improve text extraction accuracy. By using Faster R-CNN to localize text regions and then applying OCR, the model can effectively handle diverse and complex images.

    Custom Text Classification: The model utilizes a CNN for text classification, tailored to predict entity values from extracted text. This approach ensures that the predictions are both accurate and aligned with the entity values in the dataset.

    End-to-End Pipeline: The solution encompasses an end-to-end pipeline from data preprocessing and feature extraction to model training and prediction. This comprehensive approach ensures seamless integration and performance optimization across all stages of the process.

    Adaptability: The pipeline is designed to be adaptable to various types of products and images. By training on diverse datasets and using robust detection and extraction methods, the model can handle a wide range of product attributes.

Installation

To set up the environment and install the required dependencies, use:

bash

pip install -r requirements.txt

Ensure that you have the following installed:

    Python 3.x
    Tesseract OCR
    PyTorch (for Faster R-CNN)
    TensorFlow (for text classification)

Usage
Data Preparation

    Place the dataset files (train.csv, test.csv, sample_test.csv, and sample_test_out.csv) into the dataset/ directory.
    Ensure that the train.csv file contains labeled data with image_link, entity_name, and entity_value, and that test.csv contains image_link and group_id.

Running the Pipeline

Execute the main.py script to run the entire pipeline:

bash

python main.py

This script will:

    Download images from URLs provided in test.csv.
    Preprocess images by extracting and cleaning text.
    Train the text classification model using the prepared training data.
    Predict entity values for test images and save results to output.csv.
    Validate the output format using the sanity checker.

Output

The results will be saved in output.csv with columns:

    index: The unique identifier from the test set.
    prediction: The predicted entity value in the format "x unit".

Notes

    Tesseract Path: Ensure that the Tesseract executable path in src/utils.py is correctly set for your system.
    Model Fine-Tuning: Adjust model parameters and perform hyperparameter tuning based on your specific dataset and requirements.
    Error Handling: Properly handle exceptions and errors for robust performance in real-world scenarios.