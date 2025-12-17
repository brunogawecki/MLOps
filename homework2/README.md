# Fashion MNIST BentoML Service

## Setup

1. Save the model to BentoML:
   ```bash
   python src/bento.py
   ```

2. Start the service:
   ```bash
   bentoml serve src.service:FashionMNISTService
   ```

## Testing with Streamlit

In a new terminal, run:
```bash
streamlit run frontend/dashboard.py
```

The dashboard will open in your browser where you can upload images and see predictions.
