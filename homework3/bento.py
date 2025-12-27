"""
This script saves the model and processor to BentoML.
"""

import bentoml
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path

def main():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    model.eval()
    
    with bentoml.models.create(name='resnet-50') as bento_model:
        torch.save(model.state_dict(), bento_model.path_of('model.pt'))

        processor.save_pretrained(bento_model.path_of('processor'))

        model.config.save_pretrained(bento_model.path_of('model_config'))

        print(f'Model saved to BentoML: {bento_model.tag}')

if __name__ == '__main__':
    main()