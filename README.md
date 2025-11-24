# Next_word_prediction
AI Project
Next Word Prediction using LSTM (AI Project)

This project implements a Next Word Prediction model using Recurrent Neural Networks (LSTM) in TensorFlow/Keras.
The model is trained on a text dataset and predicts the most likely next words based on the user-provided input.

📌 Project Title

Next Word Prediction

📚 Subject

Artificial Intelligence

📂 Project Overview

The goal of this project is to implement a deep learning model that can predict the next word in a sequence.
This is a basic example of Language Modeling, a fundamental task in Natural Language Processing (NLP).

The model is trained using:

Tokenizer

N-gram training sequences

Embedding layer

LSTM layer

Dense softmax output layer

The system learns from the provided text data (e.g., pizza.txt) and generates predictions based on learned patterns.

🧠 How the Model Works
1. Text Preprocessing

The uploaded code uses a function:

def file_to_sentence_list(file_path):


It:

Reads the text file

Splits the text into sentences using regex

Removes unwanted spaces

Returns a clean list of sentences

This helps create a structured dataset.

2. Tokenization

The text dataset is converted into numerical tokens:

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)


Every word is assigned a unique integer

These tokens are used to create training sequences

3. Creating N-gram Sequences

For each sentence, n-gram sequences are generated:

Example sentence:

“Pizza has different varieties”

N-grams generated:

pizza has

pizza has different

pizza has different varieties

This helps the model learn how words follow each other.

4. Preparing the Training Data

All sequences are padded to equal length (pre-padding)

Data split:

X → Predictors (all words except last)

y → Label (last word)

The label is one-hot encoded for training:

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

5. Building the LSTM Model

The model architecture:

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len - 1))
model.add(LSTM(128))
model.add(Dense(total_words, activation='softmax'))


Layers Explanation:

Embedding Layer: Converts tokens into meaningful vectors

LSTM Layer (128 units): Remembers long-term word patterns

Dense Softmax Layer: Predicts the next word probability

6. Training the Model

The model is trained for 500 epochs:

model.fit(X, y, epochs=500, verbose=1)


Higher epochs allow the model to learn deeper relationships.

🔮 Next Word Prediction

After training, the model can generate new words:

seed_text = "Pizza have different "
next_words = 5


The model:

Converts the input seed to tokens

Pads to required length

Predicts the most probable next word

Appends it to the sentence

Repeats for next_words times

Example output:

Next predicted words: Pizza have different ...

📁 Files in the Project
File	Description
Untitled13.ipynb	Main Colab Notebook containing all code
pizza.txt	Training dataset (text file)
README.md	Project documentation
🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Regex

NLP preprocessing

🚀 How to Run the Project

Install dependencies:

pip install tensorflow numpy regex


Place your text dataset:

pizza.txt


Run the Colab Notebook or Python script.

Modify seed_text to generate your own predictions.

📌 Future Improvements

Add custom dataset upload

### Author : Arshad, Akarsh, Akriti

Improve accuracy with bigger text data

Use Bidirectional LSTM or Transformer

Deploy as a Flask/Streamlit web app
