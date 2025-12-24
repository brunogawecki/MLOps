import requests
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    image_path = script_dir / "cow.jpg"

    try:
        # Send image path as JSON
        # Use absolute path so the service can find the file
        # Convert Path object to string for JSON serialization
        response = requests.post(
            "http://localhost:3000/predict",
            json={"image_path": str(image_path.absolute())}
        )
        print(response.json())
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the service.")
        print("Make sure the service is running with: bentoml serve service.py:ResNet50Service")

if __name__ == "__main__":
    main()