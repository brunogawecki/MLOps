import requests
import numpy as np
from PIL import Image
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    image_path = script_dir / "cow.jpg"

    try:
        # Load image and convert to numpy array
        pil_image = Image.open(image_path)
        
        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Convert to numpy array (shape: height, width, channels)
        image_array = np.array(pil_image, dtype=np.uint8)
        
        # Convert to list for JSON serialization
        image_list = image_array.tolist()
        
        # Send as JSON
        response = requests.post(
            "http://localhost:3000/predict",
            json={"image": image_list}
        )
        print(response.json())
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the service.")
        print("Make sure the service is running with: bentoml serve service.py:ResNet50Service")

if __name__ == "__main__":
    main()