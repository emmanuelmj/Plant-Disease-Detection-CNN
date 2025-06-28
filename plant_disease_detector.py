
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# Set image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 5  # Reduced for quick runs

# Data loading (dummy directory structure expected)
train_dir = 'data/train'
val_dir = 'data/val'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Save model
model.save('plant_disease_cnn_model.h5')

# Load and predict on an image
def predict_disease(image_path, model_path='plant_disease_cnn_model.h5'):
    model = load_model(model_path)
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction[0])
    class_labels = list(train_generator.class_indices.keys())
    DISP = class_labels[predicted_class]

    print(f"Predicted Disease: {DISP}")
    return DISP

# Convert DISP to image
def text_to_image(text, output_path='prediction_output.png'):
    img = Image.new('RGB', (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 40), f"Predicted Disease: {text}", font=font, fill=(0, 0, 0))
    img.save(output_path)

# Example usage
if __name__ == '__main__':
    test_image = 'sample_leaf.jpg'  # Replace with your image file
    DISP = predict_disease(test_image)
    text_to_image(DISP)
