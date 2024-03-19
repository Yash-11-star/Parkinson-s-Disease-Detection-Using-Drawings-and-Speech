import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model



wandb.init(project="Parkinson's Detection for Wave")

# Define data paths
data_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/drawings/"

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load Training Data
train_data_generator = datagen.flow_from_directory(
    data_path + 'wave/training/',
    target_size=(224, 224),  # VGG16 input size
    batch_size=32,
    class_mode='binary'
)

# Load Testing Data
test_data_generator = datagen.flow_from_directory(
    data_path + 'wave/testing/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load pre-trained VGG16 model without top (fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained base model
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with WandbCallback
model.fit(
    train_data_generator,
    epochs=10,
    validation_data=test_data_generator,
    callbacks=[WandbCallback()]
)

# Evaluate the Model
accuracy = model.evaluate(test_data_generator)[1]
wandb.log({'Test Accuracy': accuracy})
print(f"Accuracy: {accuracy}")

plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
# model.save('wave_model', save_format='tf')

