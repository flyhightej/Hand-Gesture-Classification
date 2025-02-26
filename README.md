# Hand Gestures Classification Using Deep Learning Models

## Overview

This project focuses on recognizing American Sign Language (ASL) digits using image data and deep learning techniques. The goal is to build a model that can accurately classify images of hand signs representing digits from 0 to 9.

## Project Structure

*   **asl\_sign\_language.ipynb:** This Jupyter Notebook contains the complete code for the project, including data loading, preprocessing, model building, training, and evaluation.

## Key Technologies

*   **Python:** The primary programming language used for the project.
*   **NumPy:** Used for numerical computations and array manipulation.
*   **OpenCV (cv2):** Utilized for image processing tasks.
*   **Matplotlib:** Used for data visualization.
*   **Scikit-learn:** Employed for splitting the dataset into training and testing sets.
*   **Keras:** A high-level neural networks API, running on top of TensorFlow, used for building and training the convolutional neural network (CNN) model.
*   **TensorFlow:** An open-source machine learning framework used as the backend for Keras.

## Dataset

The dataset used in this project is the "American Sign Language Digits Dataset" from Kaggle. It contains images of hand signs for digits from 0 to 9.

*   **Source:** [https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset](https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset)

## Dependencies

To run the notebook, you will need to install the following Python packages:

&emsp; pip install numpy <br>
&emsp; pip install opencv-python <br>
&emsp; pip install matplotlib <br>
&emsp; pip install scikit-learn <br>
&emsp; pip install tensorflow <br>
&emsp; pip install keras <br>
&emsp; pip install kaggle <br>


## Usage

1.  **Download the Dataset:**  You will need a Kaggle account and API key to download the dataset.  Follow the instructions in the notebook to set up the Kaggle API.
2.  **Open the Notebook:** Open `asl_sign_language.ipynb` in Jupyter Notebook or Google Colab.
3.  **Run the Code:** Execute the cells in the notebook sequentially to load the data, preprocess it, build the CNN model, train it, and evaluate its performance.

## Model Architecture

The notebook implements a Convolutional Neural Network (CNN) model for image classification. The architecture consists of convolutional layers, pooling layers, and fully connected layers.

## Results

The notebook outputs the model's accuracy on the test set and displays visualizations of the training process.

## License

The dataset is available under the CC0-1.0 license. See Kaggle for details.
