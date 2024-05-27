FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

ENV FLASK_ENV=development

ENTRYPOINT [ "python" ] 
CMD ["-u", "main.py"]