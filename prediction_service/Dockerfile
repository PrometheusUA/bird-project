FROM mirror.gcr.io/library/python:3.11-slim-bookworm

RUN mkdir -p /app/model
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD exec gunicorn --bind :$AIP_HTTP_PORT --log-level info --workers 1 --timeout 90 app:app