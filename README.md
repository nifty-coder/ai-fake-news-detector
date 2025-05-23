# Fake News Detector

This project implements a fake news detection system using a Long Short-Term Memory (LSTM) neural network. It is trained on a combination of the LIAR: A Benchmark Dataset for Fake News Detection and a previous Fake News Detection dataset (True.csv and Fake.csv from Kaggle). The program provides a simple Turtle-based graphical user interface (GUI) for users to input news text and get a prediction of whether it is likely TRUE or FAKE.

## Overview

The goal of this project is to build a model that can classify news articles or statements as either true or fake. It leverages the power of LSTMs to understand the sequential nature of text and identify patterns indicative of misinformation. By training on diverse datasets, the model aims to achieve better generalization and accuracy.

## Features

* **Combined Dataset Training:** The model is trained on a combination of the LIAR dataset and a previous Fake News Detection dataset, exposing it to a wider range of text styles and veracity labels.
* **LSTM-based Classification:** An LSTM neural network is used as the core classification model, capable of learning long-range dependencies in text.
* **Text Preprocessing:** The input text is cleaned by removing non-alphanumeric characters, extra whitespace, and converting it to lowercase. Meaningful sentences (at least 5 words) are extracted using NLTK's sentence tokenizer.
* **Vocabulary Building:** A vocabulary is built from the combined training data to map words to numerical indices for the model.
* **Padding:** Sequences of different lengths are padded to a fixed maximum length to be processed by the LSTM.
* **Training and Validation:** The combined dataset is split into training and validation sets to train the model and evaluate its performance on unseen data.
* **Accuracy Evaluation:** The training and validation accuracy of the model are calculated and printed after training.
* **Interactive Turtle GUI:** A user-friendly graphical interface built with the Turtle library allows users to paste or type news text and get a real-time prediction (TRUE, FAKE, or UNSURE) along with a confidence score.
* **Continuous Prediction:** The Turtle UI allows for continuous input and prediction, waiting for 5 seconds after each result before prompting for new text.
* **Pre-trained Model Loading:** If a pre-trained model (`lstm_combined_news.pth`) is found, the program will load it directly, skipping the training process.

## Requirements

* Python 3.x
* PyTorch (`torch`)
* Pandas (`pandas`)
* Scikit-learn (`sklearn`)
* NLTK (`nltk`)
* Turtle (`turtle`)
* Requests (`requests`) - (Potentially for downloading NLTK data)
* `newspaper3k` (`newspaper`) - (Imported but not directly used in the final version of the code provided)

**Installation**

1.  **Install the required libraries:**
    ```bash
    pip install torch torchvision torchaudio pandas scikit-learn nltk
    ```
2.  **Download NLTK data:** Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    ```
3.  **Download the LIAR dataset:** Obtain the `train.tsv`, `valid.tsv`, and `test.tsv` files from the official LIAR dataset source (e.g., Kaggle) and place them in a `./dataset/` directory.
4.  **Download the previous Fake News Detection dataset:** Obtain the `Fake.csv` and `True.csv` files (e.g., from Kaggle) and place them in the `./dataset/` directory.
5.  **Ensure the `./dataset/` directory exists in the same location as your `main.py` script.**

## Usage

1.  **Run the `main.py` script:**
    ```bash
    python main.py
    ```
2.  **Training the model (first run or if `lstm_combined_news.pth` is not found):**
    * The script will first load and preprocess both datasets.
    * It will then train an LSTM model on the combined training data.
    * The training loss for each epoch and the final training and validation accuracy will be printed in the console.
    * The trained model and vocabulary will be saved as `lstm_combined_news.pth`.
3.  **Using the Turtle GUI (after training or loading a pre-trained model):**
    * A Turtle graphics window will appear with a prompt: "Enter News Text".
    * Paste or type the news article text you want to classify and press Enter.
    * The model's prediction (TRUE, FAKE, or UNSURE) along with a confidence score will be displayed in the Turtle window for 5 seconds.
    * The prompt will reappear, allowing you to enter more text for classification.
    * To exit the GUI, type `exit` in the input prompt or close the Turtle window.

## Model Architecture

The model consists of the following layers:

* **Embedding Layer:** Converts input word indices into dense vector representations.
* **LSTM Layer:** Processes the embedded sequences to capture contextual information.
* **Dropout Layer:** Helps prevent overfitting by randomly dropping out units during training.
* **Fully Connected Layer:** Maps the LSTM output to a single output neuron.
* **Sigmoid Activation:** Outputs a probability between 0 and 1, representing the likelihood of the news being true.

## Thresholds

The prediction is determined based on the output of the sigmoid function:

* Score > 0.7: Classified as **TRUE**.
* Score < 0.3: Classified as **FAKE**.
* Score between 0.3 and 0.7: Classified as **UNSURE**.

These thresholds can be adjusted in the `predict_news` function.

## Saving and Loading the Model

The trained model's state dictionary and the vocabulary are saved in a file named `lstm_combined_news.pth`. If this file exists when the script is run, the pre-trained model will be loaded, and the training step will be skipped.

## Potential Improvements

* **More Sophisticated Text Preprocessing:** Explore techniques like stemming, lemmatization, and handling stop words more effectively.
* **Pre-trained Word Embeddings:** Integrate pre-trained word embeddings like GloVe or Word2Vec to improve the model's understanding of word semantics.
* **More Complex Model Architectures:** Experiment with deeper LSTMs, bidirectional LSTMs, or other sequence models like Transformers.
* **Hyperparameter Tuning:** Optimize the model's hyperparameters (e.g., embedding dimension, hidden dimension, learning rate, number of epochs) using techniques like grid search or random search.
* **Evaluation Metrics:** Implement more comprehensive evaluation metrics beyond accuracy, such as precision, recall, F1-score, and AUC.
* **Feature Engineering:** Incorporate other features from the LIAR dataset (e.g., speaker information, subject) into the model.
* **Larger and More Diverse Datasets:** Train on a larger and more diverse collection of fake and true news data to improve generalization.
* **Improved Handling of Uncertain Cases:** Explore methods to provide more informative outputs for cases where the model is unsure.