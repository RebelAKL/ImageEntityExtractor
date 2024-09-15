import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split

def train_text_classification_model(texts, labels):
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences)
    
    # Define labels 
    label_set = list(set(labels))
    label_map = {label: i for i, label in enumerate(label_set)}
    y = [label_map[label] for label in labels]

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

   
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dense(len(label_set), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    return model, tokenizer, label_set
