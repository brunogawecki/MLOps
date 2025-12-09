import bentoml
import torch
import pytorch_lightning as pl
from pathlib import Path

from model import CNNModel, FashionMNISTLightningModule

def main():
    model_path = Path("model_checkpoint.ckpt")
    lightning_model = FashionMNISTLightningModule.load_from_checkpoint(model_path)
    lightning_model.eval()
    print(f'Loaded model from: {model_path}')
    
    pytorch_model = lightning_model.model
    
    with bentoml.models.create(name="fashion_mnist_cnn") as bento_model:
        torch.save(pytorch_model.state_dict(), bento_model.path_of("model.pt"))
        print(f"Model saved to BentoML: {bento_model.tag}")

if __name__ == "__main__":
    main()