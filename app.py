import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import atexit

app = Flask(__name__)
CORS(app)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)

# --- Global Variables ---
# We will train the model, scaler, and get stats when the app starts.
model_data = {}

# --- Data Cleaning Functions (from your notebook) ---
# We only need the ones for the 4 features we are using.

def clean_food_budget(value):
    """Cleans the 'Food Budget per meal' column."""
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

def clean_weekly_hobby_hours(value):
    """Cleans the 'Weekly_hobby_hours' column."""
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

# --- Persona Definitions ---
PERSONA_MAP = {
    0: {
        "name": "Balanced Budgeter",
        "description": "You have a balanced and moderate lifestyle. You enjoy hobbies and eating out occasionally, but you keep a close eye on your budget and don't have an over-the-top sweet tooth. You represent a sustainable and well-rounded approach to student life."
    },
    1: {
        "name": "Weekend Indulger",
        "description": "You're a 'work hard, play hard' type. You might be frugal on daily meals but aren't afraid to spend more when you do eat out. You have a significant sweet tooth and really value your free time, packing your hobby hours into the week. You live for those moments of indulgence."
    },
    2: {
        "name": "Premium Lifestyle",
        "description": "You enjoy the finer things and aren't afraid to spend on high-quality meals. You eat out often and have a major sweet tooth. Your lifestyle is active and full, with a high number of hobby hours. You're likely very social and live a fast-paced, premium life."
    },
}

def get_persona_name(cluster_label):
    """Maps a cluster label to a persona dictionary."""
    return PERSONA_MAP.get(cluster_label, {"name": "Explorer", "description": "Your unique lifestyle doesn't fit a common mold! You're charting your own path."})


# --- On-Startup Model Training ---
def train_model_on_startup():
    """
    Loads data, cleans it, trains the model, and stores it,
    the scaler, and stats in the global 'model_data' dict.
    """
    global model_data
    app.logger.info("Starting model training on startup...")
    
    try:
        df = pd.read_csv('Studentdata.csv')
        app.logger.info("Successfully loaded 'Studentdata.csv'")
    except FileNotFoundError:
        app.logger.error("FATAL: 'Studentdata.csv' not found.")
        app.logger.error("Please place 'Studentdata.csv' in the same folder as app.py")
        return

    # --- 1. Select & Clean 4 Relevant Features ---
    # Note: Your notebook had trailing spaces in some column names.
    # We use the keys here to select and rename for simplicity.
    feature_map = {
        'Eating Out Per week': 'eating_out_per_week',
        'Food Budget per meal': 'food_budget_per_meal_inr',
        'Sweet tooth level ': 'sweet_tooth_level', # Trailing space
        'Weekly_hobby_hours   ': 'weekly_hobby_hours'  # Trailing spaces
    }
    
    try:
        # Select only the columns we need
        df_processed = df[list(feature_map.keys())].copy()
        # Rename them to simple names
        df_processed = df_processed.rename(columns=feature_map)
    except KeyError as e:
        app.logger.error(f"FATAL: A column is missing from 'Studentdata.csv'. Error: {e}")
        return

    # --- 2. Apply Cleaning Functions ---
    df_processed['food_budget_per_meal_inr'] = df_processed['food_budget_per_meal_inr'].apply(clean_food_budget)
    df_processed['weekly_hobby_hours'] = df_processed['weekly_hobby_hours'].apply(clean_weekly_hobby_hours)
    
    # --- 3. Impute Missing Data ---
    # Calculate medians *after* cleaning
    medians = {
        'eating_out_per_week': df_processed['eating_out_per_week'].median(),
        'food_budget_per_meal_inr': df_processed['food_budget_per_meal_inr'].median(),
        'sweet_tooth_level': df_processed['sweet_tooth_level'].median(),
        'weekly_hobby_hours': df_processed['weekly_hobby_hours'].median(),
    }
    
    # Apply imputation
    for col, med in medians.items():
        df_processed[col].fillna(med, inplace=True)
        
    app.logger.info(f"Data cleaned. Imputed with medians: {medians}")

    # --- 4. Scale Data ---
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_processed)
    app.logger.info("Data scaled.")

    # --- 5. Train Model ---
    # Using 3 clusters as it gives more interesting personas
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    app.logger.info(f"KMeans model trained with {n_clusters} clusters.")
    
    # --- 6. Get Cluster Stats for Charts ---
    # We use the *unscaled* data for human-readable chart stats
    df_processed['cluster'] = kmeans.labels_
    
    # Calculate means for each cluster
    cluster_stats = df_processed.groupby('cluster').mean().reset_index().to_dict(orient='records')
    
    # Calculate global averages for comparison
    global_avg = df_processed.drop('cluster', axis=1).mean().to_dict()

    # --- 7. Store Everything in Memory ---
    model_data = {
        'model': kmeans,
        'scaler': scaler,
        'medians': medians,
        'columns': list(df_processed.drop('cluster', axis=1).columns), # Save exact column order
        'cluster_stats': cluster_stats,
        'global_avg': global_avg
    }
    
    app.logger.info("Model, scaler, and stats are trained and stored in memory.")
    app.logger.info("Backend is ready to accept predictions.")


# --- API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict_persona():
    """
    Receives 4 features from the frontend, scales them,
    and predicts the cluster/persona.
    """
    if 'model' not in model_data:
        return jsonify({"error": "Model not trained. Check server logs."}), 500
        
    try:
        data = request.json
        app.logger.info(f"Received data for prediction: {data}")
        
        # Get data from request
        eating_out = data.get('eating_out_per_week')
        food_budget = data.get('food_budget_per_meal_inr')
        sweet_tooth = data.get('sweet_tooth_level')
        hobby_hours = data.get('weekly_hobby_hours')

        # --- 1. Create DataFrame in the correct order ---
        # Use the saved column order from training
        input_data = pd.DataFrame({
            'eating_out_per_week': [eating_out],
            'food_budget_per_meal_inr': [food_budget],
            'sweet_tooth_level': [sweet_tooth],
            'weekly_hobby_hours': [hobby_hours]
        })
        input_data = input_data[model_data['columns']] # Enforce column order

        # --- 2. Impute missing (if any) with saved medians ---
        # This makes the API robust if a field is empty
        for col, med in model_data['medians'].items():
            input_data[col].fillna(med, inplace=True)
            # Ensure types are correct (request data is string)
            input_data[col] = pd.to_numeric(input_data[col])

        app.logger.info(f"Cleaned/Imputed data: {input_data.to_dict(orient='records')}")

        # --- 3. Scale data with the *fitted* scaler ---
        scaled_data = model_data['scaler'].transform(input_data)
        
        # --- 4. Predict with the *trained* model ---
        prediction = model_data['model'].predict(scaled_data)
        cluster_label = int(prediction[0])
        
        persona = get_persona_name(cluster_label)
        
        app.logger.info(f"Prediction: Cluster {cluster_label} ({persona['name']})")
        
        # --- 5. Return the result ---
        return jsonify({
            'cluster': cluster_label,
            'persona': persona,
            'user_data': input_data.to_dict(orient='records')[0] # Send back the cleaned data
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get_persona_stats', methods=['GET'])
def get_stats():
    """
    Simple endpoint to send the pre-calculated
    cluster stats to the frontend for charting.
    """
    if 'cluster_stats' not in model_data:
        return jsonify({"error": "Stats not ready. Check server logs."}), 500
        
    return jsonify({
        "cluster_stats": model_data['cluster_stats'],
        "global_avg": model_data['global_avg'],
        "persona_map": PERSONA_MAP,
        "labels": model_data['columns'] # Send feature names for charts
    })

@app.route('/')
def index():
    """Serves the main HTML page."""
    # This is not strictly needed if using Live Server,
    # but it's good practice for deployment.
    return render_template('index.html')

# --- Run the App ---
if __name__ == '__main__':
    # Train the model *before* starting the server
    train_model_on_startup()
    # Run the Flask server
    app.run(debug=True, port=5000)