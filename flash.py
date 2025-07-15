from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np


app = Flask(__name__)
model = tf.keras.models.load_model('asl_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    img = preprocess(request.files['image'])  # Preprocesar como en el entrenamiento
    prediction = model.predict(img)
    return jsonify({'class': chr(65 + np.argmax(prediction))})  # Devuelve la letra predicha