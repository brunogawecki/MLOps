"""
Client script to test the Fashion MNIST BentoML service.

This script sends a sample image to the BentoML service and displays the prediction.
"""

import requests
import numpy as np
from PIL import Image
import PIL.ImageOps


# The URL where our BentoML service is running
SERVICE_URL = "http://localhost:3000/predict"

def main():
    image_path = 'tshirt.jpg'
    image = Image.open(image_path)

    # accept only square images
    if image.width != image.height:
        raise ValueError(f"Image must be square! Current dimensions: {image.width}x{image.height}")

    image = image.resize((28, 28))
    print(f'Loaded image: {image_path}')

    # Fashion MNIST is white object on black background (inverse of usual photos)
    image = PIL.ImageOps.invert(image.convert('RGB'))
    gray_image = np.array(image.convert('L'))
    
    pixel_values = gray_image.flatten().tolist()
    
    print("=== BentoML Fashion MNIST Service Test ===")
    print(f"Sending {len(pixel_values)} pixel values to the service...")
    print(f"Image shape: 28x28")
    print(f"Service URL: {SERVICE_URL}")
    print()
    
    # Send POST request to the service
    try:
        response = requests.post(
            SERVICE_URL,
            json={"image": pixel_values}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("=== Prediction Result ===")
            print(f"Predicted class ID: {result['class_id']}")
            print(f"Predicted class name: {result['class_name']}")
            print(f"Predicted class probabilities: {result['probabilities']}")
            print()
            print("✓ Service is working correctly!")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the service.")
        print("Make sure the service is running with: bentoml serve service:FashionMNISTService")


if __name__ == "__main__":
    main()
