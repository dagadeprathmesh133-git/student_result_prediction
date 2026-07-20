import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the SVM model using pickle
MODEL_PATH = "svm_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Single-file UI Layout with Modern Glassmorphism & Micro-animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Predictive Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --accent-color: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.5);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            overflow-x: hidden;
        }

        /* Subtle Background Animation Floating Elements */
        body::before, body::after {
            content: '';
            position: absolute;
            width: 300px;
            height: 300px;
            border-radius: 50%;
            background: var(--accent-color);
            filter: blur(120px);
            opacity: 0.15;
            z-index: -1;
            animation: float 12s infinite alternate ease-in-out;
        }
        body::before { top: 10%; left: 15%; }
        body::after { bottom: 10%; right: 15%; animation-delay: 6s; }

        @keyframes float {
            0% { transform: translateY(0) scale(1); }
            100% { transform: translateY(-40px) scale(1.1); }
        }

        .container {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            width: 100%;
            max-width: 750px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #fff, var(--text-muted));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 2.5rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        @media (max-width: 600px) {
            .grid-form { grid-template-columns: 1fr; }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group.full-width {
            grid-column: span 2;
        }
        @media (max-width: 600px) {
            .form-group.full-width { grid-column: span 1; }
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        input, select {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: all 0.3s ease;
            outline: none;
        }

        input:focus, select:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 12px var(--accent-glow);
            background: rgba(255, 255, 255, 0.08);
        }

        select option {
            background: #1e1b4b;
            color: var(--text-main);
        }

        .btn-submit {
            grid-column: span 2;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px var(--accent-glow);
            margin-top: 1rem;
        }

        @media (max-width: 600px) {
            .btn-submit { grid-column: span 1; }
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px var(--accent-glow);
            filter: brightness(1.1);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            animation: pulseIn 0.5s ease-out;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .result-box.success {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border-color: rgba(16, 185, 129, 0.3);
        }

        .result-box.danger {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border-color: rgba(239, 68, 68, 0.3);
        }

        @keyframes pulseIn {
            from { transform: scale(0.95); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>SVM Predictive Model</h1>
    <div class="subtitle">Enter feature configurations to run classification inference</div>

    <form action="/predict" method="POST" class="grid-form">
        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
                <option value="1">Male</option>
                <option value="0">Female</option>
            </select>
        </div>

        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" step="any" placeholder="e.g. 18" required>
        </div>

        <div class="form-group">
            <label for="study_hours">Study Hours / Week</label>
            <input type="number" id="study_hours" name="study_hours" step="any" placeholder="e.g. 12" required>
        </div>

        <div class="form-group">
            <label for="attendance">Attendance Rate</label>
            <input type="number" id="attendance" name="attendance" step="any" placeholder="e.g. 90" required>
        </div>

        <div class="form-group">
            <label for="parent_edu">Parent Education Level</label>
            <input type="number" id="parent_edu" name="parent_edu" step="any" placeholder="Encoded Value" required>
        </div>

        <div class="form-group">
            <label for="internet">Internet Access</label>
            <select id="internet" name="internet" required>
                <option value="1">Yes</option>
                <option value="0">No</option>
            </select>
        </div>

        <div class="form-group">
            <label for="extracurricular">Extracurricular Activities</label>
            <select id="extracurricular" name="extracurricular" required>
                <option value="1">Yes</option>
                <option value="0">No</option>
            </select>
        </div>

        <div class="form-group">
            <label for="prev_score">Previous Score</label>
            <input type="number" id="prev_score" name="prev_score" step="any" placeholder="e.g. 85" required>
        </div>

        <div class="form-group full-width">
            <label for="final_score">Final Score</label>
            <input type="number" id="final_score" name="final_score" step="any" placeholder="e.g. 88" required>
        </div>

        <button type="submit" class="btn-submit">Execute Prediction</button>
    </form>

    {% if prediction %}
    <div class="result-box {% if prediction == 'Yes' %}success{% else %}danger{% endif %}">
        Prediction Status: {{ prediction }}
    </div>
    {% endif %}
    
    {% if error %}
    <div class="result-box danger">
        {{ error }}
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Error: svm_model.pkl file not found or failed to load.")
    
    try:
        # Extract features mapping to the model metadata
        features = [
            float(request.form['gender']),
            float(request.form['age']),
            float(request.form['study_hours']),
            float(request.form['attendance']),
            float(request.form['parent_edu']),
            float(request.form['internet']),
            float(request.form['extracurricular']),
            float(request.form['prev_score']),
            float(request.form['final_score'])
        ]
        
        # Format for input array
        input_data = np.array([features])
        
        # Run inference
        prediction_output = model.predict(input_data)
        prediction_result = str(prediction_output[0])
        
        return render_template_string(HTML_TEMPLATE, prediction=prediction_result)
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f"Inference Failure: {str(e)}")

if __name__ == '__main__':
    # Default to port 5000 for local testing, Render injects its own PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
