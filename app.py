from flask import Flask, request, jsonify
import numpy as np
import pickle

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the trained model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))

# Create a function for prediction
def diabetes_prediction(input_data):
    # Changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Create a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data as a JSON array
        input_data = request.get_json(force=True)['data']

        # Make predictions
        result = diabetes_prediction(input_data)

        # Return the result as JSON
        return jsonify(result=result)

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
