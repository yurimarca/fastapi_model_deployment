FROM public.ecr.aws/lambda/python:3.9

# Create artifacts directory
RUN mkdir -p ${LAMBDA_TASK_ROOT}/artifacts

# Copy function code
COPY ./app ${LAMBDA_TASK_ROOT}

# Copy artifacts (model files)
COPY ./artifacts/* ${LAMBDA_TASK_ROOT}/artifacts/

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip3 install --target "${LAMBDA_TASK_ROOT}" --upgrade --no-cache-dir -r requirements.txt

# AWS Lambda requires a handler function
CMD ["main.handler"]
