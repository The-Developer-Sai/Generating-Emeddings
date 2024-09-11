from flask import Flask, request, redirect, render_template_string
import os
import pandas as pd
import numpy as np
import docx
import time
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
import torch
import openai

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['EMBEDDINGS_FOLDER'] = os.path.join(BASE_DIR, 'embeddings')
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'doc', 'docx'}

# Set your OpenAI API Key
openai.api_key = 'OpenAi API Key'

# Load the BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Available embedding models
embedding_models = {
    'bert': 'BERT Embeddings',
    'openai': 'OpenAI GPT Embeddings'
}

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload a File</title>
    <h1>Upload a File</h1>
    <form action="/upload" method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <h1>Select an Embedding Model</h1>
    <form action="/select_model" method=post>
      <select name="model">
        <option value="bert">BERT Embeddings</option>
        <option value="openai">OpenAI GPT Embeddings</option>
      </select>
      <input type=submit value="Select Model">
    </form>
    '''

@app.route('/select_model', methods=['POST'])
def select_model():
    selected_model = request.form.get('model')
    return f'''
    <h1>Selected Model: {embedding_models[selected_model]}</h1>
    <form action="/upload_model" method=post enctype=multipart/form-data>
      <input type="hidden" name="model" value="{selected_model}">
      <input type=file name=file>
      <input type=submit value=Upload and Process>
    </form>
    '''

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    selected_model = request.form.get('model')
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")

        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'csv':
            return preprocess_csv(filepath, filename, selected_model)
        elif file_extension in {'doc', 'docx'}:
            return preprocess_doc(filepath, filename, selected_model)
        else:
            return f'File type {file_extension} is not supported.'

    return 'File type not allowed.'

def preprocess_csv(filepath, filename, model_name):
    start_time = time.time()
    try:
        df = pd.read_csv(filepath)
        df.fillna(method='ffill', inplace=True)  # Fill missing values
        
        # Compute embeddings based on the selected model
        if model_name == 'bert':
            embeddings = get_bert_embeddings(df)
        elif model_name == 'openai':
            embeddings = get_openai_embeddings(df)
        else:
            return 'Invalid model selected.'

        embeddings_filepath = os.path.join(app.config['EMBEDDINGS_FOLDER'], f'{filename}_embeddings.npy')
        np.save(embeddings_filepath, embeddings)
        time_taken = time.time() - start_time
        return f'CSV File uploaded and preprocessed using {embedding_models[model_name]}. Embeddings saved to {embeddings_filepath}. Time taken: {time_taken:.2f} seconds.'

    except Exception as e:
        return f'Error processing CSV file: {e}'

def preprocess_doc(filepath, filename, model_name):
    start_time = time.time()
    try:
        doc = docx.Document(filepath)
        text = ' '.join([para.text for para in doc.paragraphs if para.text.strip() != ''])
        
        if model_name == 'bert':
            embeddings = get_bert_text_embedding(text)
        elif model_name == 'openai':
            embeddings = get_openai_text_embedding(text)
        else:
            return 'Invalid model selected.'

        embeddings_filepath = os.path.join(app.config['EMBEDDINGS_FOLDER'], f'{filename}_embeddings.npy')
        np.save(embeddings_filepath, embeddings)
        time_taken = time.time() - start_time
        return f'DOC/DOCX File uploaded and preprocessed using {embedding_models[model_name]}. Embeddings saved to {embeddings_filepath}. Time taken: {time_taken:.2f} seconds.'

    except Exception as e:
        return f'Error processing DOC/DOCX file: {e}'

def get_bert_embeddings(df):
    # Assume we are embedding the first text column
    text_column = df.select_dtypes(include=['object']).iloc[:, 0].fillna('')
    embeddings = []
    for text in text_column:
        embeddings.append(get_bert_text_embedding(text))
    return np.array(embeddings)

def get_bert_text_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def get_openai_embeddings(df):
    # Assume we are embedding the first text column
    text_column = df.select_dtypes(include=['object']).iloc[:, 0].fillna('')
    embeddings = []
    for text in text_column:
        embeddings.append(get_openai_text_embedding(text))
    return np.array(embeddings)

def get_openai_text_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    # Check if upload and embeddings folders exist, and create them if they don't
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        print(f"Created directory: {app.config['UPLOAD_FOLDER']}")
    
    if not os.path.exists(app.config['EMBEDDINGS_FOLDER']):
        os.makedirs(app.config['EMBEDDINGS_FOLDER'])
        print(f"Created directory: {app.config['EMBEDDINGS_FOLDER']}")
    
    # Run the Flask application
    app.run(debug=True)
