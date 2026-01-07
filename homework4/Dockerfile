# use AWS Lambda base image
FROM public.ecr.aws/lambda/python:3.11

# install system dependencies (gcc-c++ is the correct package name for Amazon Linux)
RUN yum install -y gcc gcc-c++ python3-devel && yum clean all

# install python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# copy model directory to /opt/ml/model (lambda's model directory)
COPY mobilenetv2 /opt/ml/model/mobilenetv2

# copy appllication code
COPY model.py  ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# set the cmd to run the handler function
CMD ["lambda_function.lambda_handler"]