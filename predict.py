import cv2
import keras
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

types = ['Cat', 'Dog']

def load_and_preprocess_image(path):
    # Load the image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error loading image from path: {path}")
        return None
    
    # Resize the image
    new_arr = cv2.resize(img, (224, 224))
    
    # Normalize and reshape the image
    new_arr = new_arr / 255.0
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(1, 224, 224, 1)
    
    return new_arr

image_path = '/Users/satvikverma/Workspace/Dog-Cat-Classification/2.jpg'  # Provide the correct image path here

# Load and preprocess the image
input_image = load_and_preprocess_image(image_path)

if input_image is not None:
    # Make a prediction
    prediction = model.predict(input_image)
    
    # Get the predicted class
    predicted_class = types[prediction.argmax()]
    print(f"The image is a {predicted_class}.")
