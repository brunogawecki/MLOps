import bentoml
import torch
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
    def predict(self, image_path: str) -> dict:
        # Load image from path
        image = Image.open(image_path)
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        inputs = self.processor(image, return_tensors='pt')

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

    @bentoml.api
    def add(self, a: int, b: int) -> int:
        return a + b