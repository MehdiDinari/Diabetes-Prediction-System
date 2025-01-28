from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

# Absolute Path to CSV File
CSV_FILE_PATH = r"C:\Users\Fddkk\PycharmProjects\Diabete\diabetes.csv"

# Home Page
def home(request):
    return render(request, 'home.html')

# Prediction Page
def predict(request):
    return render(request, 'predict.html')

# Result Page (Processing User Input)
def result(request):
    try:
        # Ensure the file exists
        if not os.path.exists(CSV_FILE_PATH):
            raise FileNotFoundError(f"CSV file not found at: {CSV_FILE_PATH}")

        # Load dataset
        data = pd.read_csv(CSV_FILE_PATH)

        # Splitting Data
        X = data.drop("Outcome", axis=1)
        Y = data["Outcome"]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        # Train Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)

        # Retrieve User Input Safely
        val1 = float(request.POST.get('n1', 0))
        val2 = float(request.POST.get('n2', 0))
        val3 = float(request.POST.get('n3', 0))
        val4 = float(request.POST.get('n4', 0))
        val5 = float(request.POST.get('n5', 0))
        val6 = float(request.POST.get('n6', 0))
        val7 = float(request.POST.get('n7', 0))
        val8 = float(request.POST.get('n8', 0))

        # Convert input into NumPy array
        user_input = np.array([[val1, val2, val3, val4, val5, val6, val7, val8]])
        pred = model.predict(user_input)

        # Assign Result
        result2 = "Positive" if pred[0] == 1 else "Negative"

    except FileNotFoundError as e:
        result2 = f"Error: {str(e)} - Ensure the file exists at {CSV_FILE_PATH}"
    except Exception as e:
        result2 = f"Error: {str(e)}"

    return render(request, 'predict.html', {'result2': result2})
