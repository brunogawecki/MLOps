Connect to Ubuntu linux64 EC2 instance

`ssh -i MyTTTKeyPair.pem ubuntu@SERVER_IP`

Run the docker container:

`docker run -d --name resnet -p 3000:3000 brunogawecki/resnet-50-classifier`