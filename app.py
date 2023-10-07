import numpy as np
from flask import Flask, render_template, request

# Import necessary functions for loading and using the Keras model
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('my_model.h5')
scaler = StandardScaler()  # Create a scaler for normalization

@app.route('/future-sales-by-family')
def get_future_sales_by_family():
    future_sales_by_family = get_future_sales_by_family()
    return render_template('future_sales_by_family.html', future_sales_by_family=future_sales_by_family)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = []
        features = ['date', 'store_nbr', 'family', 'onpromotion', 'oil_price', 'type']
        for feature in features:
            input_field = request.form.get(f'input_{feature}')
            if input_field is None or input_field.strip() == '':
                return render_template('index.html', error='Input data cannot be empty.')

            input_data.append(float(input_field))

        # Normalize the input data
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)[0][0]

        return render_template('prediction.html', prediction=prediction)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
