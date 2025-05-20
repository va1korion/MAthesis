FROM python:3.10-slim

# open ingress ports for application
EXPOSE 8000
EXPOSE 80
RUN apt-get update && apt-get install -y git && apt-get clean
RUN apt-get install -y build-essential libssl-dev libffi-dev python3-dev poppler-utils tesseract-ocr-rus tesseract-ocr

STOPSIGNAL SIGINT
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install python-multipart
COPY * .
COPY .. /app/
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


CMD ["python", "main.py"]