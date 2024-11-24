

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

model_path = r"C:\Users\user\Desktop\pneumonia_detection\model\saved_model.h5"
test_folder_path = r"C:\Users\user\Desktop\pneumonia_detection\dataset\test"  

model = load_model(model_path)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

subfolders = ['NORMAL', 'PNEUMONIA']

for subfolder in subfolders:
    subfolder_path = os.path.join(test_folder_path, subfolder)

    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)

        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename} in {subfolder} folder")
            
            # Image Pre-processing
            image_size = (150, 150)
            image = load_img(file_path, target_size=image_size)
            image = img_to_array(image) / 255.0  # Rescale to [0, 1]
            image = np.expand_dims(image, axis=0)

            # Prediction
            prediction = model.predict(image)[0][0]
            if prediction > 0.5:
                print(f"Prediction for {filename}: Pneumonia detected!")
            else:
                print(f"Prediction for {filename}: Normal chest X-ray.")
