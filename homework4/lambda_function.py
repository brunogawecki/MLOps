import json
import boto3
from pathlib import Path
from model import load_model, predict
import traceback

s3 = boto3.client('s3')

# Global model cache
model = None
image_processor = None

def get_model():
    """Load model once per container"""
    global model, image_processor
    if model is None or image_processor is None:
        model, image_processor = load_model()
    return model, image_processor

def lambda_handler(event, context):
    try:
        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            object_key = record['s3']['object']['key']

            print(f'Processing image from {bucket_name}/{object_key}')

            # download image as bytes
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            image_bytes = response['Body'].read()

            model, image_processor = get_model()

            results = predict(image_bytes, model, image_processor)

            # upload results as json to s3
            results_key = f"results/{Path(object_key).stem}_predictions.json"
            s3.put_object(Bucket=bucket_name, Key=results_key, Body=json.dumps(results).encode('utf-8'), ContentType='application/json')

            print(f'Results uploaded to {bucket_name}/{results_key}')

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Success',
                    'results' : results_key,
                    'predictions' : results
                })
            }    
    except Exception as e:
        print(f'Error: {str(e)}')
        print(f'Traceback: {traceback.format_exc()}')
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error',
                'error': str(e)
            })
        }