import cv2
import numpy as np
from tensorflow.keras.models import load_model #type: ignore

def predict_image(image, model):
    try:
        # Make prediction
        prediction = model.predict(image)  # image is already preprocessed
        
        # Process the prediction
        prediction = np.squeeze(prediction)  # Remove batch dimension
        
        # Convert prediction to heatmap
        prediction = (prediction * 255).astype(np.uint8)  # Scale to 0-255 range
        heatmap = cv2.applyColorMap(prediction, cv2.COLORMAP_JET)
        
        # Get the original image (first image from batch, remove normalization)
        original = (image[0] * 255).astype(np.uint8)
        
        # Overlay heatmap on original image
        output = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        
        return output

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None