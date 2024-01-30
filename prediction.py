import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Path to the new image
new_Spiral = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/P_Spiral.webp"
new_Wave = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/H_wave.png"

# Load and preprocess the new image
new_image_wave = image.load_img(new_Wave, target_size=(224, 224))
new_image_array_wave = image.img_to_array(new_image_wave)
new_image_array_wave = np.expand_dims(new_image_array_wave, axis=0)
new_image_array_wave /= 255.0  # Normalize pixel values

new_image_spiral = image.load_img(new_Spiral, target_size=(224, 224))
new_image_array_spiral = image.img_to_array(new_image_spiral)
new_image_array_spiral = np.expand_dims(new_image_array_spiral, axis=0)
new_image_array_spiral /= 255.0  # Normalize pixel values

spiral_model = load_model('spiral_model', compile=False)
wave_model = load_model('wave_model', compile=False)

# Make predictions
prediction_spiral = spiral_model.predict(new_image_array_spiral)
prediction_wave = wave_model.predict(new_image_array_wave)

print(f"Chances of Parkinson's disease is {prediction_spiral[0][0]*100}")
print(f"Chances of Parkinson's disease is {prediction_wave[0][0]*100}")

# Interpret the prediction
if prediction_spiral[0][0] > 0.5 and prediction_wave[0][0] > 0.5:
    print("The person likely has Parkinson's disease.")
else:
    print("The person is likely healthy.")
