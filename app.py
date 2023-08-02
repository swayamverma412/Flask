from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the ML model from 'model.pkl'
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_date', methods=['POST'])
def get_predicted_date():
    try:
        delay = int(request.form['delay'])  
        due_date_str = request.form['due_in_date']

        due_date = pd.to_datetime(due_date_str, format="%Y%m%d")

        predicted_date = model.predict(delay, due_date)

        predicted_date_str = predicted_date.strftime("%Y-%m-%d")

        response = {
            'predicted_date': predicted_date_str
        }

        return render_template('index.html', result=response['predicted_date'])

    except Exception as e:
        print("Error:", e)  
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
