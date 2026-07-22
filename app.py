import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained SVM model
MODEL_PATH = "SVM_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML Template with modern Tailwind CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Model Prediction Interface</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex items-center justify-center p-4">
    <div class="max-w-3xl w-full bg-slate-800 rounded-2xl shadow-2xl border border-slate-700 p-8">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-500">
                Student Performance Prediction
            </h1>
            <p class="text-slate-400 mt-2">Enter the required feature values to predict the target outcome.</p>
        </div>

        {% if prediction %}
        <div class="mb-8 p-6 rounded-xl text-center bg-gradient-to-r from-indigo-950 to-blue-950 border border-indigo-500/30">
            <h3 class="text-sm uppercase tracking-wider text-indigo-400 font-semibold mb-1">Prediction Result</h3>
            <p class="text-4xl font-extrabold text-white">{{ prediction }}</p>
        </div>
        {% endif %}

        <form action="/predict" method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase mb-2">Gender</label>
                <select name="gender" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
                    <option value="0">Female (0)</option>
                    <option value="1">Male (1)</option>
                </select>
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase mb-2">Age</label>
                <input type="number" step="any" name="age" placeholder="e.g. 18" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase mb-2">Study Hours Per Week</label>
                <input type="number" step="any" name="study_hours_per_week" placeholder="e.g. 15" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase mb-2">Attendance Rate (%)</label>
                <input type="number" step="any" name="attendance_rate" placeholder="e.g. 85" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase mb-2">Parent Education Level</label>
                <input type="number" step="any" name="parent_education" placeholder="e.g. 1, 2, 3" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase mb-2">Internet Access</label>
                <select name="internet_access" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
                    <option value="1">Yes (1)</option>
                    <option value="0">No (0)</option>
                </select>
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase mb-2">Extracurricular Activities</label>
                <select name="extracurricular" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
                    <option value="1">Yes (1)</option>
                    <option value="0">No (0)</option>
                </select>
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase mb-2">Previous Score</label>
                <input type="number" step="any" name="previous_score" placeholder="e.g. 75" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>

            <div class="md:col-span-2">
                <label class="block text-xs font-semibold text-slate-300 uppercase mb-2">Final Score</label>
                <input type="number" step="any" name="final_score" placeholder="e.g. 80" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>

            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 px-6 rounded-lg shadow-lg hover:shadow-indigo-500/30 transition-all duration-200">
                    Predict Outcome
                </button>
            </div>
        </form>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction="Error: Model file 'SVM_model.pkl' not loaded.")

    try:
        # Extract features matching the model's expected inputs
        feature_dict = {
            "gender": [float(request.form.get("gender"))],
            "age": [float(request.form.get("age"))],
            "study_hours_per_week": [float(request.form.get("study_hours_per_week"))],
            "attendance_rate": [float(request.form.get("attendance_rate"))],
            "parent_education": [float(request.form.get("parent_education"))],
            "internet_access": [float(request.form.get("internet_access"))],
            "extracurricular": [float(request.form.get("extracurricular"))],
            "previous_score": [float(request.form.get("previous_score"))],
            "final_score": [float(request.form.get("final_score"))],
        }

        input_df = pd.DataFrame(feature_dict)
        prediction = model.predict(input_df)[0]

        return render_template_string(HTML_TEMPLATE, prediction=str(prediction))

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
