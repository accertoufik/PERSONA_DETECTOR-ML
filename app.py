from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import logging
import io # Needed to read the file from memory
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Define all data cleaning functions from your notebook ---
# These will be used on the uploaded file

def clean_food_budget(value):
    if isinstance(value, str):
        value = value.lower().replace('rs', '').strip()
        if '-' in value:
            parts = value.split('-')
            try:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return float(value)

def clean_screen_time(value):
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ['', 'no']:
            return np.nan
        value = value.replace('hours', '').replace('hrs', '').replace('hr', '').strip()
        if '-' in value or 'to' in value:
            if 'to' in value:
                parts = value.split('to')
            else:
                parts = value.split('-')
            try:
                num_parts = []
                for part in parts:
                    part = part.strip()
                    if part.isdigit():
                        num_parts.append(float(part))
                    else:
                        return np.nan
                if len(num_parts) == 2:
                    return sum(num_parts) / 2
                else:
                    return np.nan
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return float(value)

def clean_gaming_hours(value):
    if isinstance(value, str):
        value = value.lower().strip()
        value = value.replace('hours', '').replace('hrs', '').replace('hr', '').strip()
        if '-' in value or 'to' in value:
            if 'to' in value:
                parts = value.split('to')
            else:
                parts = value.split('-')
            try:
                num_parts = []
                for part in parts:
                    part = part.strip()
                    if part.isdigit():
                        num_parts.append(float(part))
                    else:
                        return np.nan
                if len(num_parts) == 2:
                    return sum(num_parts) / 2
                else:
                    return np.nan
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return float(value)

def clean_social_media_minutes(value):
    if isinstance(value, str):
        value = value.lower().strip()
        value = value.replace('minutes', '').replace('hrs', '').replace('hours', '').replace('mins', '').strip()
        if '-' in value or 'to' in value:
            if 'to' in value:
                parts = value.split('to')
            else:
                parts = value.split('-')
            try:
                num_parts = []
                for part in parts:
                    part = part.strip()
                    if part.isdigit():
                        num_parts.append(float(part))
                    else:
                        return np.nan
                if len(num_parts) == 2:
                    return sum(num_parts) / 2
                else:
                    return np.nan
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return float(value)

def clean_weekly_hobby_hours(value):
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ['', 'not fixed', 'kk', 'no']:
            return np.nan
        value = value.replace('hours', '').replace('hrs', '').replace('h', '').strip()
        if '-' in value:
            parts = value.split('-')
            try:
                num_parts = []
                for part in parts:
                    part = part.strip()
                    if part.isdigit():
                        num_parts.append(float(part))
                    else:
                        return np.nan
                if len(num_parts) == 2:
                    return sum(num_parts) / 2
                else:
                    return np.nan
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return float(value)

def clean_books_read(value):
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ['', 'no']:
            return np.nan
        if '-' in value:
            parts = value.split('-')
            try:
                num_parts = []
                for part in parts:
                    part = part.strip()
                    if part.isdigit():
                        num_parts.append(float(part))
                    else:
                        return np.nan
                if len(num_parts) == 2:
                    return sum(num_parts) / 2
                else:
                    return np.nan
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return float(value)

def clean_fashion_spend(value):
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ['', 'not fixed', 'p']:
            return np.nan
        value = value.replace('rs', '').strip()
        if '-' in value:
            parts = value.split('-')
            try:
                num_parts = []
                for part in parts:
                    part = part.strip()
                    if part.isdigit():
                        num_parts.append(float(part))
                    else:
                        return np.nan
                if len(num_parts) == 2:
                    return sum(num_parts) / 2
                else:
                    return np.nan
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return float(value)

def clean_budget_per_trip(value):
    if isinstance(value, str):
        value = value.lower().strip()
        value = value.replace('rs', '').replace(',', '').strip()
        if '-' in value:
            parts = value.split('-')
            try:
                num_parts = []
                for part in parts:
                    part = part.strip()
                    if part.isdigit():
                        num_parts.append(float(part))
                    else:
                        return np.nan
                if len(num_parts) == 2:
                    return sum(num_parts) / 2
                else:
                    return np.nan
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return float(value)

# --- Main Processing Endpoint ---
@app.route('/upload_and_process', methods=['POST'])
def upload_and_process_file():
    app.logger.info("Received request to /upload_and_process")
    
    # Check if a file was uploaded
    if 'file' not in request.files:
        app.logger.warning("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        app.logger.warning("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # --- 1. Load Data ---
            app.logger.info(f"Processing file: {file.filename}")
            df = pd.read_csv(file)
            app.logger.info(f"Successfully loaded CSV with {len(df)} rows.")

            # --- 2. Select Relevant Features (from notebook) ---
            relevant_features = [
                'Age', 'Height', 'Weight', 'Eating Out Per week', 'Food Budget per meal', 
                'Sweet tooth level ', '  Binge frequency per week  ', 
                '  Screen Time Movies/series in hours per week  ', '  Gaming days per week  ', 
                '  Gaming hours per week  ', '  Daily Social Media Minutes  ', 
                '  Listening hours per day  ', '  Live concerts past year  ', 
                'Weekly_hobby_hours   ', '  Books read past year', '  Fashion spend per month', 
                'Budget per trip', '  Introversion extraversion  ', '  Risk taking  ', 
                '  Conscientiousness  ', '  Open to new experiences  ', '  Teamwork preference  '
            ]
            
            # Keep a copy of original data for merging later if needed
            df_original = df.copy() 
            df_processed = df[relevant_features].copy()

            # --- 3. Clean Data & Calculate Medians (from notebook) ---
            app.logger.info("Starting data cleaning...")
            df_processed['Age'] = df_processed['Age'].str.replace('Age-', '', regex=False).astype(int)
            
            obj_cols_to_clean = {
                'Food Budget per meal': clean_food_budget,
                '  Screen Time Movies/series in hours per week  ': clean_screen_time,
                '  Gaming hours per week  ': clean_gaming_hours,
                '  Daily Social Media Minutes  ': clean_social_media_minutes,
                'Weekly_hobby_hours   ': clean_weekly_hobby_hours,
                '  Books read past year': clean_books_read,
                '  Fashion spend per month': clean_fashion_spend,
                'Budget per trip': clean_budget_per_trip
            }
            
            for col, func in obj_cols_to_clean.items():
                df_processed[col] = df_processed[col].apply(func)
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
            
            df_processed.fillna(0, inplace=True) # Impute other numerics with 0
            app.logger.info("Data cleaned and imputed.")

            # --- 4. Scale Data ---
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_processed)
            app.logger.info("Data scaled.")

            # --- 5. Train Model & Predict ---
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(df_scaled)
            app.logger.info("KMeans model trained and predictions made.")

            # --- 6. Prepare Results ---
            # Add cluster IDs and Persona names to the cleaned dataframe
            df_processed['Cluster'] = clusters
            persona_map = { 0: "Moderate Engager", 1: "High-Spending Entertainer" }
            df_processed['Persona'] = df_processed['Cluster'].map(persona_map)
            
            # Optionally, merge back with original data to include non-feature columns
            # For this app, we'll just return the processed features and the persona
            
            # Convert results to JSON
            results_json = df_processed.to_json(orient='records')
            
            # Send JSON response
            # We wrap the JSON string in another JSON object
            return jsonify({'status': 'success', 'data': results_json})

        except Exception as e:
            app.logger.error(f"Error during processing: {e}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a .csv file.'}), 400

# --- Health Check Endpoint ---
@app.route('/', methods=['GET'])
def health_check():
    return "Student Persona Clustering API is running. Use the /upload_and_process endpoint to post a CSV."

# Run the server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)