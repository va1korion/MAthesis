FROM python:3.10-slim

# open ingress ports for application
EXPOSE 8000
EXPOSE 80
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*


STOPSIGNAL SIGINT
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
COPY * .
COPY . /app/



CMD ["python", "main.py"]