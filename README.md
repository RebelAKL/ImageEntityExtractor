Entity Extraction from Images - ML Hackathon
Overview

This project aims to develop a machine learning model that extracts entity values from product images. The extracted information, such as weight, volume, and dimensions, is critical for digital marketplaces where textual descriptions might be missing. The solution uses Optical Character Recognition (OCR) with Tesseract and object detection with Faster R-CNN, along with a text classification model to predict entity values.

Project Structure



project/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_test.csv
│   └── sample_test_out.csv
├── src/
│   ├── constants.py
│   ├── sanity.py
│   └── utils.py
├── models/
│   ├── object_detection.py
│   └── text_classification.py
├── preprocessing/
│   └── preprocess.py
├── main.py
└── requirements.txt


Certainly! Here’s a sample README.md file that explains the working of the code for your GitHub repository.
Entity Extraction from Images - ML Hackathon
Overview

This project aims to develop a machine learning model that extracts entity values from product images. The extracted information, such as weight, volume, and dimensions, is critical for digital marketplaces where textual descriptions might be missing. The solution uses Optical Character Recognition (OCR) with Tesseract and object detection with Faster R-CNN, along with a text classification model to predict entity values.
Project Structure

css

project/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_test.csv
│   └── sample_test_out.csv
├── src/
│   ├── constants.py
│   ├── sanity.py
│   └── utils.py
├── models/
│   ├── object_detection.py
│   └── text_classification.py
├── preprocessing/
│   └── preprocess.py
├── main.py
└── requirements.txt

Directory and File Descriptions

    dataset/: Contains CSV files with training and test data. The train.csv file includes labeled data, while test.csv includes unlabeled data.
    src/: Contains utility functions and constants.
        constants.py: Defines allowed units for predictions.
        sanity.py: Contains a sanity checker to validate output formatting.
        utils.py: Includes functions for downloading images and extracting text with Tesseract.
    models/: Contains code for object detection and text classification.
        object_detection.py: Implements object detection using Faster R-CNN.
        text_classification.py: Implements a text classification model using TensorFlow/Keras.
    preprocessing/: Contains code for preprocessing the data.
        preprocess.py: Prepares training data by extracting and cleaning text from images.
    main.py: The entry point script that orchestrates the entire pipeline from data preprocessing to model training and prediction.
    requirements.txt: Lists the required Python packages.



    Installation

To set up the environment and install the required dependencies, use:

bash

pip install -r requirements.txt

Ensure that you have the following installed:

    Python 3.x
    Tesseract OCR
    PyTorch (for Faster R-CNN)
    TensorFlow (for text classification)


    Certainly! Here’s a sample README.md file that explains the working of the code for your GitHub repository.
Entity Extraction from Images - ML Hackathon
Overview

This project aims to develop a machine learning model that extracts entity values from product images. The extracted information, such as weight, volume, and dimensions, is critical for digital marketplaces where textual descriptions might be missing. The solution uses Optical Character Recognition (OCR) with Tesseract and object detection with Faster R-CNN, along with a text classification model to predict entity values.
Project Structure

css

project/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_test.csv
│   └── sample_test_out.csv
├── src/
│   ├── constants.py
│   ├── sanity.py
│   └── utils.py
├── models/
│   ├── object_detection.py
│   └── text_classification.py
├── preprocessing/
│   └── preprocess.py
├── main.py
└── requirements.txt

Directory and File Descriptions

    dataset/: Contains CSV files with training and test data. The train.csv file includes labeled data, while test.csv includes unlabeled data.
    src/: Contains utility functions and constants.
        constants.py: Defines allowed units for predictions.
        sanity.py: Contains a sanity checker to validate output formatting.
        utils.py: Includes functions for downloading images and extracting text with Tesseract.
    models/: Contains code for object detection and text classification.
        object_detection.py: Implements object detection using Faster R-CNN.
        text_classification.py: Implements a text classification model using TensorFlow/Keras.
    preprocessing/: Contains code for preprocessing the data.
        preprocess.py: Prepares training data by extracting and cleaning text from images.
    main.py: The entry point script that orchestrates the entire pipeline from data preprocessing to model training and prediction.
    requirements.txt: Lists the required Python packages.

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
    The train.csv file should contain labeled data with image_link, entity_name, and entity_value.
    The test.csv file should contain image_link and group_id.



    Certainly! Here’s a sample README.md file that explains the working of the code for your GitHub repository.
Entity Extraction from Images - ML Hackathon
Overview

This project aims to develop a machine learning model that extracts entity values from product images. The extracted information, such as weight, volume, and dimensions, is critical for digital marketplaces where textual descriptions might be missing. The solution uses Optical Character Recognition (OCR) with Tesseract and object detection with Faster R-CNN, along with a text classification model to predict entity values.
Project Structure

css

project/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_test.csv
│   └── sample_test_out.csv
├── src/
│   ├── constants.py
│   ├── sanity.py
│   └── utils.py
├── models/
│   ├── object_detection.py
│   └── text_classification.py
├── preprocessing/
│   └── preprocess.py
├── main.py
└── requirements.txt

Directory and File Descriptions

    dataset/: Contains CSV files with training and test data. The train.csv file includes labeled data, while test.csv includes unlabeled data.
    src/: Contains utility functions and constants.
        constants.py: Defines allowed units for predictions.
        sanity.py: Contains a sanity checker to validate output formatting.
        utils.py: Includes functions for downloading images and extracting text with Tesseract.
    models/: Contains code for object detection and text classification.
        object_detection.py: Implements object detection using Faster R-CNN.
        text_classification.py: Implements a text classification model using TensorFlow/Keras.
    preprocessing/: Contains code for preprocessing the data.
        preprocess.py: Prepares training data by extracting and cleaning text from images.
    main.py: The entry point script that orchestrates the entire pipeline from data preprocessing to model training and prediction.
    requirements.txt: Lists the required Python packages.

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
    The train.csv file should contain labeled data with image_link, entity_name, and entity_value.
    The test.csv file should contain image_link and group_id.

Running the Pipeline

Execute the main.py script to run the entire pipeline:

bash

python main.py

The script performs the following steps:

    Download Images: Downloads images specified in test.csv to the images/ directory.
    Preprocess Data: Extracts and cleans text from images using OCR.
    Train Model: Trains a text classification model based on the provided training data.
    Predict: Uses the trained model to predict entity values for the test data.
    Save Predictions: Saves predictions to output.csv.
    Validate Output: Checks the output file format using the sanity checker.

Output

The output will be saved in output.csv with the following columns:

    index: The unique identifier from the test set.
    prediction: The predicted entity value in the format "x unit".

Notes

    Tesseract Path: Ensure that the Tesseract executable path in src/utils.py is correctly set for your system.
    Model Fine-Tuning: You may need to adjust model parameters and perform hyperparameter tuning based on your specific dataset.
    Error Handling: Ensure proper handling of exceptions and errors in real-world scenarios.