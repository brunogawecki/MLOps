import bentoml
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from PIL import Image

@bentoml.service(
    name='resnet-50-classifier',
    resources={'cpu': '1'},
    traffic={'timeout': 60}
)
class ResNet50Service:
    def __init__(self):
        # Get model from bentoml store
        bento_model = bentoml.models.get('resnet-50:latest')

        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(bento_model.path_of('processor'))
        
        # Load config and create model architecture
        config = AutoConfig.from_pretrained(bento_model.path_of('model_config'))
        self.model = AutoModelForImageClassification.from_config(config)
        
        # Load the saved weights
        state_dict = torch.load(bento_model.path_of('model.pt'), map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print('ResNet-50 model loaded successfully')

    @bentoml.api
    def predict(self, image: np.ndarray) -> dict:
        # Convert input to numpy array with correct dtype
        # JSON deserializes integers as int64, but PIL needs uint8
        if isinstance(image, list):
            image = np.array(image, dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            # Ensure uint8 dtype (handle int64 from JSON)
            if image.dtype != np.uint8:
                # Clip values to 0-255 range and convert to uint8
                image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = np.array(image, dtype=np.uint8)
        
        # Ensure correct shape: (height, width, channels) or (height, width)
        if len(image.shape) == 3 and image.shape[2] == 1:
            # Handle grayscale with channel dimension
            image = image.squeeze(axis=2)
        elif len(image.shape) == 1:
            # If it's a flat array, we can't determine shape - this shouldn't happen
            raise ValueError(f"Unexpected image shape: {image.shape}. Expected 2D or 3D array.")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Ensure RGB format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Preprocess image
        inputs = self.processor(pil_image, return_tensors='pt')

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get top 5 predictions
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        top5_probs, top5_indices = torch.topk(probabilities, 5)

        # Format results
        results = []
        for prob, idx in zip(top5_probs, top5_indices):
            class_id = int(idx.item())
            results.append({
                'class_id': class_id,
                'class_name': self.model.config.id2label[class_id],
                'score': float(round(prob.item(), 4))
            })

        return {
            'top_predictions': results,
            'predicted_class': results[0]['class_id'],
            'predicted_class_name': results[0]['class_name'],
            'confidence': results[0]['score']
        }