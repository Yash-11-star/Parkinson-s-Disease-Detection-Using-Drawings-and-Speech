import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Function to extract features from images
def extract_features_from_images(image_folder, label):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))  # Resize image for consistency
        images.append(img)
        labels.append(label)
    return images, labels

# Function to load images and labels from folders
def load_images_from_folders(folders):
    images = []
    labels = []
    for folder in folders:
        healthy_folder = os.path.join(folder, 'healthy')
        parkinson_folder = os.path.join(folder, 'parkinson')
        healthy_images, healthy_labels = extract_features_from_images(healthy_folder, 0)
        parkinson_images, parkinson_labels = extract_features_from_images(parkinson_folder, 1)
        images.extend(healthy_images)
        images.extend(parkinson_images)
        labels.extend(healthy_labels)
        labels.extend(parkinson_labels)
    return images, labels

# Load images and labels
folders = [
    "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/drawings/spiral/training",
    "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/drawings/spiral/testing",
    "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/drawings/wave/training",
    "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/drawings/wave/testing"
]
images, labels = load_images_from_folders(folders)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape input data to include the batch dimension
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Create ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Random rotation up to 20 degrees
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Shear intensity
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    vertical_flip=True,  # Random vertical flip
    fill_mode='nearest'  # Fill mode for newly created pixels
)


# Define the model architecture with added layers and dropout for regularization
model = models.Sequential([
    layers.Flatten(input_shape=(100, 100)),  # Flatten the input images
    layers.Dense(256, activation='relu'),  # Dense layer with 256 neurons
    layers.Dropout(0.5),  # Dropout layer with dropout rate of 0.5
    layers.Dense(128, activation='relu'),  # Dense layer with 128 neurons
    layers.Dropout(0.5),  # Dropout layer with dropout rate of 0.5
    layers.Dense(64, activation='relu'),  # Dense layer with 64 neurons
    layers.Dropout(0.5),  # Dropout layer with dropout rate of 0.5
    layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
batch_size = 32
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
steps_per_epoch = len(X_train) // batch_size
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=1000, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

model.save('Ann_Model', save_format='tf')
