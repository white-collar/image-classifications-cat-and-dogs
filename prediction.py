from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('cats_vs_dogs.h5')

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)  # Convert image to array
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
    img_tensor /= 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(img_tensor)

    # Interpret the result
    if prediction[0] > 0.5:
        print("This is a dog.")
    else:
        print("This is a cat.")

# Example usage
predict_image('unknown-dog-1.jpeg')

