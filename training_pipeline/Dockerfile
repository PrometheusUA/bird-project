FROM mirror.gcr.io/library/python:3.11-slim-bookworm
WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY src /src
ENTRYPOINT [ "bash" ]