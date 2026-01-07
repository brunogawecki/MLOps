# Testing Guide
Uses MobileNetV2

### Upload image to S3 bucket:
`aws s3 cp test_images/image-name.jpeg s3://mlops-hw4-bucket/`

### Check if predictions were created:
`aws s3 ls s3://mlops-hw4-bucket/results/`

### Print JSON predictions:
`aws s3 cp s3://mlops-hw4-bucket/results/image-name_predictions.json -`

### Cleanup: delete all files
`aws s3 rm s3://mlops-hw4-bucket/ --recursive`