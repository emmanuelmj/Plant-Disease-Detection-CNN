# Step 1: Clone a GitHub plant disease dataset (small version for demo)
!git clone https://github.com/spMohanty/PlantVillage-Dataset.git
!mkdir -p plant_data
!mv PlantVillage-Dataset/raw/color plant_data/PlantVillage
!pip install pytorch_transformers

# Step 2: Import libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# importing and using Groq for text to image classification
from groq import Groq
import os
os.environ["Emmanuel"] = "GROQ_API_KEY"
#making chat.application using groq

while True:
    data = input("Enter your prompt : ")
    if data == "bye":
        print("Thank you for chatting!!")
        break

    client = Groq(
        api_key=os.environ.get("Emmanuel"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": data, #content == ur prompt
            }
        ],
        model="deepseek-r1-distill-llama-70b",
        stream=False,
    )

    print(chat_completion.choices[0].message.content)

# Step 3: Prepare dataset
data_dir = 'plant_data/PlantVillage'
img_width, img_height = 128, 128
batch_size = 32

# Remove non-image files or irrelevant classes if needed (optional)

# Data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
# Step 4: Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Step 5: Class label mapping
class_indices = train_gen.class_indices
class_labels = {v: k for k, v in class_indices.items()}

# Step 6: Prediction function (DISP stores disease name)

def predict_plant_disease(image_path):
    image = load_img(image_path, target_size=(img_width, img_height))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    DISP = class_labels[predicted_class]
    return DISP

# Step 7: Example prediction
# Replace with a real test image from the dataset:
test_image_path = train_gen.filepaths[0]  # just grab one sample image

DISP = predict_plant_disease(test_image_path)
print("Detected Disease (DISP):", DISP)

#Converting output {DISP} into a image using a pipeline
os.environ["HF_TOKEN"] = "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"
prompt = DISP
image = pipe(prompt).images[0]
