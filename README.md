# Twitter-Sentimental-Analysis

## Overview
This project demonstrates sentiment analysis on Twitter data using a deep learning model. The goal is to classify tweets as positive, negative, or neutral. The analysis involves preprocessing the data, building a Word2Vec model, tokenizing text, training a deep learning model, and evaluating its performance.

## Dependencies
- **DataFrame**: `pandas`
- **Data Visualization**: `matplotlib`
- **Machine Learning Libraries**: `scikit-learn`, `gensim`
- **Deep Learning Libraries**: `keras`
- **Natural Language Processing**: `nltk`


## Data
The dataset used in this project contains tweets with the following columns:
- `target`: Polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- `ids`: ID of the tweet
- `date`: Date of the tweet
- `flag`: Query (lyx) or NO_QUERY if no query
- `user`: User who tweeted
- `text`: Text of the tweet

## Data Preprocessing
1. **Cleaning**: Remove links, users, and special characters from the text.
2. **Tokenization**: Split the text into words.
3. **Stopword Removal**: Remove common words like "and", "the", etc.
4. **Label Encoding**: Encode target labels (0 -> NEGATIVE, 2 -> NEUTRAL, 4 -> POSITIVE).

## Word2Vec Model
Train a Word2Vec model on the preprocessed text data. This model learns vector representations of words, capturing semantic meanings.

## Tokenization and Padding
Tokenize the preprocessed text and pad sequences to a fixed length for model input.

## Model Architecture
- **Embedding Layer**: Maps words to dense vectors of fixed size (pre-trained Word2Vec embeddings).
- **LSTM Layer**: Long Short-Term Memory layer with dropout for sequence prediction.
- **Dense Layer**: Output layer with sigmoid activation for binary classification.

## Training the Model
Train the model using the preprocessed data. Utilize callbacks for reducing learning rate on plateaus and early stopping.

## Evaluation
- **Confusion Matrix**: Visualize the performance of the model.
- **Classification Report**: Precision, recall, and F1-score for each class.
- **Accuracy Score**: Overall accuracy of the model.

## Usage
1. **Prediction**: Utilize the trained model to predict sentiment for new tweets.
2. **Model and Tokenizer Saving**: Save the trained model, Word2Vec model, tokenizer, and label encoder for future use.

### Files
- **`model.h5`**: Trained Keras model.
- **`model.w2v`**: Word2Vec model.
- **`tokenizer.pkl`**: Tokenizer for text processing.
- **`encoder.pkl`**: Label encoder for target labels.

For detailed implementation, refer to the provided code. Enjoy analyzing sentiments in Twitter data!
