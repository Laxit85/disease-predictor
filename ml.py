# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from nltk.corpus import stopwords
import re
import nltk
import joblib
import os
from langdetect import detect
from googletrans import Translator

# Download stopwords
nltk.download('stopwords')

# Load the data from multiple CSV files and concatenate
# Updated to load only the two specified datasets as per user request
data2 = pd.read_csv('disease_prediction_dataset.csv')
data = pd.concat([data2], ignore_index=True, join='outer')

# Debug print to check data columns and sample before cleaning
print("Columns in concatenated data:", data.columns)
print("Sample data before cleaning:")
print(data.head(10))

# Combine symptom columns into a single 'text' column for preprocessing
symptom_cols = [col for col in data.columns if col.startswith('symptom_')]
data['text'] = data[symptom_cols].fillna('').agg(' '.join, axis=1)

# Define numerical columns to be used as features
numerical_cols = ['age', 'blood_pressure', 'glucose_level', 'cholesterol', 'bmi']

# Data Cleaning
data.dropna(subset=['text', 'diagnosis'] + numerical_cols, inplace=True)  # Remove missing values in relevant columns
print(f"Data shape after dropping rows with missing values: {data.shape}")

# Rename 'diagnosis' column to 'label' for consistency
data.rename(columns={'diagnosis': 'label'}, inplace=True)

# Initialize custom Hindi stopwords
hindi_stopwords = set([
    'और', 'है', 'के', 'को', 'में', 'से', 'यह', 'कि', 'पर', 'नहीं', 
    'तो', 'किया', 'होगा', 'होगी', 'होगे', 'क्योंकि', 'अगर', 'लेकिन', 
    'जब', 'तब', 'साथ', 'भी', 'या', 'अभी', 'किसी', 'किस', 'उन', 
    'उनका', 'उनकी', 'हम', 'हमारा', 'हमारी', 'आप', 'आपका', 'आपकी', 
    'यहाँ', 'वहाँ', 'कहाँ', 'कब', 'क्यों', 'कैसे', 'क्या', 'कौन', 
    'जैसे'
])

# Multilingual stop words
stop_words = {
    'en': set(stopwords.words('english')),
    'hi': hindi_stopwords,
    # Add more languages as needed
}

# Initialize the translator
translator = Translator()

# Text Preprocessing Function
def preprocess_text(text):
    # Detect language
    lang = detect(text)
    lang = lang if lang in stop_words else 'en'  # Default to English if language is not supported

    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [word for word in text if word not in stop_words[lang]]  # Remove stop words
    return ' '.join(text)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Strip whitespace from preprocessed text
data['text'] = data['text'].str.strip()

# Remove empty or whitespace-only strings after preprocessing to avoid empty vocabulary error
empty_text_count = (data['text'] == '').sum()
print(f"Number of empty documents after preprocessing: {empty_text_count}")
data = data[data['text'] != '']

# Print some samples of preprocessed text for debugging
print("Sample preprocessed texts:")
print(data['text'].head(10))

# Custom transformer to select numerical columns
class NumericalFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols):
        self.numerical_cols = numerical_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.numerical_cols]

# Custom transformer to combine text features
class TextFeatureCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, symptom_cols):
        self.symptom_cols = symptom_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Combine symptom columns into a single text column
        combined_text = X[self.symptom_cols].fillna('').agg(' '.join, axis=1)
        return combined_text

# Feature Selection
# Prepare features: both text and numerical
X_text = data['text']
X_numerical = data[numerical_cols]
X = pd.concat([X_text, X_numerical], axis=1)  # Combine text and numerical features
y = data['label']  # Target variable

# Create preprocessing pipelines
# Text preprocessing pipeline
text_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer())
])

# Numerical preprocessing pipeline
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_pipeline, 'text'),
        ('num', numerical_pipeline, numerical_cols)
    ]
)

# Create a pipeline that combines preprocessing and model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
full_pipeline.fit(X_train, y_train)

# Model Evaluation
y_pred = full_pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model pipeline and feature columns
joblib.dump(full_pipeline, 'symptom_disease_model.pkl')
joblib.dump(numerical_cols, 'numerical_features.pkl')  # Save numerical feature names for later use

# Function to save new disease inputs and update Excel
def save_new_disease(symptoms, disease_name, age=None, blood_pressure=None, glucose_level=None, cholesterol=None, bmi=None):
    # Save to text file
    with open('new_diseases.txt', 'a') as file:
        # Include numerical features in the saved data
        numerical_data = f"{age or ''},{blood_pressure or ''},{glucose_level or ''},{cholesterol or ''},{bmi or ''}"
        file.write(f"{symptoms},{disease_name},{numerical_data}\n")
    
    # Save to Excel
    # Create a dictionary with all features
    data_dict = {'text': [symptoms], 'label': [disease_name]}
    if age is not None:
        data_dict['age'] = [age]
    if blood_pressure is not None:
        data_dict['blood_pressure'] = [blood_pressure]
    if glucose_level is not None:
        data_dict['glucose_level'] = [glucose_level]
    if cholesterol is not None:
        data_dict['cholesterol'] = [cholesterol]
    if bmi is not None:
        data_dict['bmi'] = [bmi]
    
    new_data = pd.DataFrame(data_dict)
    excel_file = 'Symptom2Disease.xlsx'
    
    if os.path.exists(excel_file):
        # Append to existing Excel file
        with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            new_data.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        # Create a new Excel file
        new_data.to_excel(excel_file, index=False)

# Function to validate user input
def is_valid_input(symptoms):
    # Check for empty input
    if not symptoms.strip():
        return False
    # Preprocess the input to check for meaningful content
    processed_symptoms = preprocess_text(symptoms)
    # Check if the processed symptoms contain meaningful words
    if len(processed_symptoms.split()) < 2:  # Require at least 2 meaningful words
        return False
    return True

# Function to predict disease based on user input (both text and numerical)
def predict_disease(symptoms, age=None, blood_pressure=None, glucose_level=None, cholesterol=None, bmi=None, top_n=3):
    # Load the model pipeline and numerical feature names
    model_pipeline = joblib.load('symptom_disease_model.pkl')
    numerical_cols = joblib.load('numerical_features.pkl')
    
    # Detect language and translate if necessary
    lang = detect(symptoms)
    if lang == 'hi':  # If input is in Hindi
        symptoms = translator.translate(symptoms, src='hi', dest='en').text

    # Validate input
    if not is_valid_input(symptoms):
        return "Error: Please enter valid symptoms. Ensure your input is meaningful and contains at least two words."

    # Preprocess the input symptoms
    processed_symptoms = preprocess_text(symptoms)

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'text': [processed_symptoms],
        'age': [age if age is not None else np.nan],
        'blood_pressure': [blood_pressure if blood_pressure is not None else np.nan],
        'glucose_level': [glucose_level if glucose_level is not None else np.nan],
        'cholesterol': [cholesterol if cholesterol is not None else np.nan],
        'bmi': [bmi if bmi is not None else np.nan]
    })
    
    # Fill missing numerical values with mean values from training data if needed
    # For simplicity, we'll fill with 0, but in a production system you might want to use mean values
    for col in numerical_cols:
        if input_data[col].isnull().any():
            input_data[col].fillna(0, inplace=True)
    
    # Get prediction probabilities
    prediction_probs = model_pipeline.predict_proba(input_data)
    
    # Get the indices of the top N predictions
    top_n_indices = prediction_probs[0].argsort()[-top_n:][::-1]
    
    # Get the corresponding disease names and their probabilities
    top_diseases = [(model_pipeline.classes_[i], prediction_probs[0][i]) for i in top_n_indices]
    
    return top_diseases

# Function to retrain the model with new data
def retrain_model():
    # Load new data from the file
    if os.path.exists('new_diseases.txt'):
        # Read the new data with numerical features
        new_data_lines = []
        with open('new_diseases.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    symptoms = parts[0]
                    disease_name = parts[1]
                    # Handle numerical features if they exist
                    numerical_data = parts[2:7] if len(parts) >= 7 else [''] * 5
                    # Create a row with all features
                    row_data = [symptoms, disease_name] + numerical_data
                    new_data_lines.append(row_data)
        
        # Create DataFrame from new data
        columns = ['text', 'label', 'age', 'blood_pressure', 'glucose_level', 'cholesterol', 'bmi']
        new_data = pd.DataFrame(new_data_lines, columns=columns)
        
        # Convert numerical columns to float
        for col in ['age', 'blood_pressure', 'glucose_level', 'cholesterol', 'bmi']:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
        
        # Append new data to the existing dataset
        global data  # Ensure we are modifying the global data variable
        data = pd.concat([data, new_data], ignore_index=True)
        
        # Proceed with the existing training steps
        data.dropna(subset=['text', 'label'], inplace=True)  # Remove rows with missing text or label
        data['text'] = data['text'].apply(preprocess_text)
        
        # Prepare features: both text and numerical
        X_text = data['text']
        X_numerical = data[['age', 'blood_pressure', 'glucose_level', 'cholesterol', 'bmi']]
        X = pd.concat([X_text, X_numerical], axis=1)  # Combine text and numerical features
        y = data['label']  # Target variable
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Retrain the model pipeline
        full_pipeline.fit(X_train, y_train)
        
        # Save the updated model
        joblib.dump(full_pipeline, 'symptom_disease_model.pkl')
        print("Model retrained with new data.")

# User Input Section
if __name__ == "__main__":
    user_symptoms = input("Please enter your symptoms: ")
    # For demonstration, you can also ask for numerical values
    # age = float(input("Please enter your age (or press Enter to skip): ") or 0)
    predicted_diseases = predict_disease(user_symptoms)
    
    if isinstance(predicted_diseases, str):  # Check if an error message was returned
        print(predicted_diseases)
    else:
        print("Predicted diseases and their probabilities:")
        for disease, prob in predicted_diseases:
            print(f"{disease}: {prob:.2f}")

