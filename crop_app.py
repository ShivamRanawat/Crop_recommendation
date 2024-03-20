import pickle
import numpy as np
from flask import jsonify
from flask import Flask, render_template, request

# Load the model and scalers
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# Define the Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return render_template('Home_1.html')

@app.route('/Predict')
def prediction():
    return render_template('index.html')

    
def recommend_crop(Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall):
    try:
        values = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]
        single_pred = np.array(values).reshape(1, -1)
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
        if prediction[0] in crop_dict:
            return crop_dict[prediction[0]]
        else:
            return "Unknown"
    except ValueError:
        return "Error in recommendation"

# Modify the Flask app route to handle recommendations
@app.route('/form', methods=["POST"])

def get_recommendation():
    try:
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Ph = float(request.form['ph'])
        Rainfall = float(request.form['Rainfall'])
    except ValueError:
        return jsonify({"error": "Please provide valid numeric values for all input fields."}), 400

    recommended_crop = recommend_crop(Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall)
    return render_template('prediction.html', recommended_crop=recommended_crop)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
