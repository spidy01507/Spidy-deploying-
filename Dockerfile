FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD gunicorn --bind 0.0.0.0:8000 --timeout 600 hf_wsgi:application
