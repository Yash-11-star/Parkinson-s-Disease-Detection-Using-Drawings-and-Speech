import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
# model = tf.keras.models.load_model('Ann_Model.h5') 
model = load_model('Ann_Model', compile=False)# Assuming 'parkinson_model.h5' is the name of your trained model file

# Function to sharpen the image
def sharpen_image(image):
    sharpened = cv2.GaussianBlur(image, (0,0), 3)
    sharpened = cv2.addWeighted(image, 1.5, sharpened, -0.5, 0)
    return sharpened

# Function to preprocess and predict blurred image
def predict_blurred_image(blur_image_path):
    # Load and preprocess the blurred image
    blur_image = cv2.imread(blur_image_path, cv2.IMREAD_GRAYSCALE)
    sharpened_image = sharpen_image(blur_image)
    sharpened_image = cv2.resize(sharpened_image, (100, 100))  # Resize image for consistency
    sharpened_image = sharpened_image / 255.0  # Normalize image

    # Reshape the image for model input
    sharpened_image = np.expand_dims(sharpened_image, axis=0)

    # Predict using the model
    prediction = model.predict(sharpened_image)

    # Print the prediction
    if prediction[0][0] > 0.5:
        print("The model predicts Parkinson's disease.")
    else:
        print("The model predicts a healthy individual.")

# Example usage
if __name__ == "__main__":
    blurred_image_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/NewData/H_wave.png"  # Replace with the path to your blurred image
    predict_blurred_image(blurred_image_path)
