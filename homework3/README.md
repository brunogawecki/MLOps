# Deploy ResNet-50 Image Classifier on EC2

Deployed ResNet-50 image classifier service using BentoML on AWS EC2.

## Setup

Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Create a `.env` file with your server IP:
   ```
   SERVER_IP=your_server_ip_here
   ```

## Deployment

Connect to your EC2 instance:
```bash
ssh -i MyTTTKeyPair.pem ubuntu@SERVER_IP
```

Run the Docker container:
```bash
docker run -d --name resnet -p 3000:3000 brunogawecki/resnet-50-classifier
```

## Running the Client

```bash
python client.py
```

The client sends an image URL to the deployed service and prints the top 5 predictions.


## API Endpoints

- `POST /predict` - Predict image class from URL
- `POST /add` - Simple addition endpoint (test)
