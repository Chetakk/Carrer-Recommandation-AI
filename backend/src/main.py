from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import random

from ML2 import career_recommendation_system, create_new_student

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/predict" , methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    
    predictions = career_recommendation_system(create_new_student(student_id=data["student_id"],
        academic_scores=data["academic_scores"],
        interest_scores=data["interest_scores"],
        personality_traits=data["personality_traits"]), generate_synthetic=True, num_synthetic=2000)
    print(predictions)
    return {"recommendation":predictions}

if __name__ == '__main__':
    app.run(debug=True)