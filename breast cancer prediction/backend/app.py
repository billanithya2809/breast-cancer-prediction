from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import numpy as np

# Load Model and Selected Features
svm_model = joblib.load("breast_cancer_svm_model.pkl")
selected_features = joblib.load("selected_features.pkl")

# Initialize Flask app
app = Flask(__name__)

# ----------- ROUTES -----------

@app.route('/')
def home():
    return render_template('Predict homepage.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/survival')
def survival():
    return render_template('survival.html')

@app.route('/predicthome')
def predicthome():
    return render_template('predicthome.html')


def process_patient_data(form):
    """Processes user input from form into a DataFrame for model prediction."""
    try:
        # Extract user input dynamically and convert to float
        patient_data = {}
        for key in selected_features:
            value = form.get(key, 0)  # Get input value, default to 0 if missing
            
            # Convert to float if possible, otherwise default to 0.0
            try:
                patient_data[key] = float(value)
            except ValueError:
                patient_data[key] = 0.0  # Handle non-numeric values safely
            
        # Convert categorical fields to numeric (ER_Status & HER2_Status)
        patient_data["ER_Status"] = 1.0 if form.get("ER_Status") == "Positive" else 0.0
        patient_data["HER2_Status"] = 1.0 if form.get("HER2_Status") == "Positive" else 0.0

        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])

        # Ensure correct column order
        patient_df = patient_df[selected_features]

        # Convert all values to float64 explicitly
        patient_df = patient_df.apply(pd.to_numeric, errors='coerce').astype(np.float64)

        # 🔍 Debugging: Check for NaN values before prediction
        print("\n🔍 Checking for NaN values in patient_df:")
        print(patient_df.isna().sum())

        # If there are NaNs, replace them with zero
        patient_df.fillna(0.0, inplace=True)

        return patient_df

    except Exception as e:
        print(f"\n❌ Error in processing patient data: {e}")
        return pd.DataFrame(columns=selected_features)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction request and renders result page."""
    try:
        # Process patient data
        patient_df = process_patient_data(request.form)

        # Ensure correct column order
        patient_df = patient_df[selected_features]

        # Convert to float64 before prediction
        patient_df = patient_df.astype(np.float64)

        # Make prediction
        prediction = svm_model.predict(patient_df)
        probability = svm_model.predict_proba(patient_df)[:, 1][0]

        # Convert prediction to integer
        result = int(prediction[0])

        # Render the result page with prediction data
        return render_template('predict_result.html', result=result, probability=round(probability, 4))

    except Exception as e:
        print(f"\n❌ Error in prediction process: {e}")
        return f"❌ Error in prediction process: {e}"



@app.route('/predict_result')
def predict_result():
    """Displays the prediction result."""
    result = request.args.get('result', "Unknown")
    probability = request.args.get('probability', "0.0")
    return render_template('predict_result.html', result=result, probability=probability)


if __name__ == '__main__':
    app.run(debug=True)
