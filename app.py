from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
from load import init
import os

gender_dict = {0: 'Male', 1: 'Female'}
app = Flask(__name__)
cnn_model = init()

# Create 'static/uploads' directory if not exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocessing function
def preprocess_input_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize image to match expected input shape
    img_array = np.array(img.convert('L'))  # Convert to grayscale
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        input_image = preprocess_input_image(file_path)
        pred = cnn_model.predict(input_image)
        pred_gender = gender_dict[int(np.round(pred[0][0]))]
        pred_age = int(np.round(pred[1][0]))
        
        return render_template('result.html', age=pred_age, gender=pred_gender, filename=filename)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=8000)
