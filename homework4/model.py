import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from pathlib import Path
from PIL import Image
import pandas as pd

MODEL_PATH = Path("mobilenetv2")

def load_model():
    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return model, image_processor

def predict(image_path: Path, model, image_processor, top_k=3):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    inputs = image_processor(image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get top k predictions
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
    
    results = []
    for prob, idx in zip(top_k_probs, top_k_indices):
        class_id = int(idx.item())
        results.append({
            'class_id': class_id,
            'class_name': model.config.id2label[class_id],
            'score': float(round(prob.item(), 4))
        })
    
    return results

def save_results_to_csv(results: list, filename: str = "results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    model, image_processor = load_model()
    results = predict("test_images/cow.jpg", model, image_processor)

    save_results_to_csv(results)

    print(f'PREDICTIONS:')
    for pred in results:
        print(f"Class ID: {pred['class_id']}, Class Name: {pred['class_name']}, Score: {pred['score']}")