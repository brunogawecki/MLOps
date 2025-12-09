import bentoml
import torch
import numpy as np
from model import CNNModel

# Fashion MNIST class labels for human-readable output
CLASS_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


@bentoml.service(
    name="fashion_mnist_classifier",
    resources={"cpu": "1"},
    traffic={"timeout": 60}
)
class FashionMNISTService:
    def __init__(self):
        # Get the model from BentoML store
        bento_model = bentoml.models.get("fashion_mnist_cnn:latest")
        
        # Create the model architecture and load the saved state dict
        self.model = CNNModel(dropout_rate=0.25)
        state_dict = torch.load(
            bento_model.path_of("model.pt"),
            map_location="cpu",
            weights_only=True  # Safe loading
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Model loaded successfully!")
    
    @bentoml.api
    def predict(self, image: np.ndarray) -> dict:
        """
        Predict the class of a Fashion MNIST image.
        
        Args:
            image: A flat list of 784 pixel values (0-255) representing a 28x28 image
        
        Returns:
            dict with 'class_id' (0-9) and 'class_name' (human readable)
        """
        # Convert to numpy array and normalize (same as training)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Reshape to (1, 1, 28, 28) - batch, channels, height, width
        image_tensor = torch.from_numpy(image_array).reshape(1, 1, 28, 28)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()
            predicted_class = torch.argmax(output, dim=1).item()
        
        return {
            "class_id": predicted_class,
            "class_name": CLASS_LABELS[predicted_class],
            "probabilities": {label: prob for label, prob in zip(CLASS_LABELS, probabilities)}
        }
