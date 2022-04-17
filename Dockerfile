FROM python:3.10
RUN apt-get update
COPY req.txt .
RUN pip install -r req.txt
RUN python -c 'import mlflow; print(mlflow.__version__)'