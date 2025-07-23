from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model and encoders
MODEL_PATH = "dt1.pkl"
SCALER_PATH = "sc.pkl"
attack_mapping = {
    0: "Survived",
    1: "Dead"
}
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
scaling_cols=['Age',
 'Weight (Kg)',
 'Height (cms)',
 'Genero',
 'Diagnosis',
 'Heart Rate',
 'oxygen saturation',
 'Respiratory Rate',
 'Systolic Blood Pressure',
 'Diastolic Blood Pressure, ',
 'Mean Blood Pressure',
 'Hour event in Minutes']

# Preprocessing functions
def preprocess_data(df, scaler):
    # Drop unnecessary columns if present
    #testing=pd.read_csv("testing.csv")
    scaling_cols=['Age',
 'Weight (Kg)',
 'Height (cms)',
 'Genero',
 'Diagnosis',
 'Heart Rate',
 'oxygen saturation',
 'Respiratory Rate',
 'Systolic Blood Pressure',
 'Diastolic Blood Pressure, ',
 'Mean Blood Pressure',
 'Hour event in Minutes']

    scaled_values = scaler.transform(df[scaling_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=scaling_cols)
    df[scaling_cols] = scaled_df
    df.drop(columns='Outcome',inplace=True)

    return df

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('predict', filename=file.filename))
    return render_template('index.html')

@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path) if filename.endswith('.csv') else pd.read_excel(file_path)
    df_processed = preprocess_data(df, scaler)
    
    predictions = model.predict(df_processed)
    #df['Prediction'] = predictions
    df['Prediction'] = [attack_mapping.get(pred, "Unknown") for pred in predictions]
    results_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    df.to_csv(results_path, index=False)
    
    return render_template('results.html', tables=df.to_html(classes='data', index=False, escape=False), download_link=url_for('download_file', filename='results.csv'))

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
