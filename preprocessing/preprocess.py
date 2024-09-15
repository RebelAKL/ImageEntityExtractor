import pandas as pd
from src.utils import extract_text_from_image, clean_text

def prepare_training_data(train_csv):
    train_data = pd.read_csv(train_csv)
    texts = []
    labels = []
    
    for index, row in train_data.iterrows():
        image_path = f'images/image_{index}.jpg'
        text = extract_text_from_image(image_path)
        cleaned_text = clean_text(text)
        texts.append(cleaned_text)
        labels.append(row['entity_value'])  # Modify as needed
    
    return texts, labels
