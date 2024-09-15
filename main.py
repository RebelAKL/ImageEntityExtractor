import pandas as pd
from preprocessing.preprocess import prepare_training_data
from models.text_classification import train_text_classification_model
from src.utils import download_images, extract_text_from_image
from src.sanity import check_sanity

def main(train_csv, test_csv):
    
    train_data, test_data = pd.read_csv(train_csv), pd.read_csv(test_csv)
    download_images(test_data['image_link'])
    
    texts, labels = prepare_training_data(train_csv)
    
    
    model, tokenizer, label_set = train_text_classification_model(texts, labels)
    
    
    predictions = []
    for index, row in test_data.iterrows():
        image_path = f'images/image_{index}.jpg'
        text = extract_text_from_image(image_path)
        cleaned_text = clean_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        X_test = pad_sequences(sequence, maxlen=X.shape[1])
        pred = model.predict(X_test)
        entity_value = label_set[np.argmax(pred)]
        predictions.append({'index': row['index'], 'prediction': entity_value})
    
    
    output_df = pd.DataFrame(predictions)
    output_df.to_csv('output.csv', index=False)
    
    
    check_sanity('output.csv')

if __name__ == '__main__':
    main('dataset/train.csv', 'dataset/test.csv')
