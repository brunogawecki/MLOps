import requests
import os
from dotenv import load_dotenv

load_dotenv()

CAT_IMAGE_URL = 'https://preview.redd.it/i-need-low-quality-cat-pics-v0-kmz6bpqggd5d1.jpeg?width=1024&format=pjpg&auto=webp&s=0c932d922b093de200d473743b37fcce2a8da418'

TART_IMAGE_URL = 'https://assets.tmecosys.com/image/upload/t_web_rdp_recipe_584x480/img/recipe/ras/Assets/EDF1539C-6C08-49DA-9672-92D6C6CCFE4C/Derivates/bd15f9b5-4bdc-478e-9f0b-567bd923a1d1.jpg'

IMAGE_URL = CAT_IMAGE_URL

def predict_image(server_ip: str, image_url: str) -> dict:
    """Send a POST request to the /predict endpoint with an image URL."""
    try:
        response = requests.post(
            f"http://{server_ip}:3000/predict",
            json={"image_url": image_url}
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the service.")
        print("Make sure the service is running with: bentoml serve service.py:ResNet50Service")
        return None

def main():
    SERVER_IP = os.getenv("SERVER_IP")
    
    result = predict_image(SERVER_IP, IMAGE_URL)

    print(f'Top 5 predictions:')
    for idx, prediction in enumerate(result['top_predictions'], start=1):
        print(f'{idx}. Class ID: {prediction["class_id"]}')
        print(f'   Class Name: {prediction["class_name"]}')
        print(f'   Confidence: {prediction["score"]}')
        print('--------------------------------')

if __name__ == "__main__":
    main()