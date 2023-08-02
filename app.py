# app.py
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the pickled model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json

        # Convert data to DataFrame
        test_data = pd.DataFrame(data)

        # Perform the same calculations as before to get the predicted_date
        test_data['predicted_date'] = pd.to_timedelta(test_data['delay'], unit="D") + pd.to_datetime(test_data['due_in_date'], format="%Y%m%d")

        # Make predictions using the loaded model
        predictions = model.predict(test_data)

        # Return the predictions as a JSON response
        result = {'predicted_date': predictions.tolist()}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
