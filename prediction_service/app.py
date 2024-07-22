import os
import sys
import logging
import re
import base64
from typing import Tuple
from time import sleep

import onnxruntime as ort
import numpy as np
from flask import Flask, jsonify, request
import librosa
import pandas as pd
from google.cloud import storage


app = Flask(__name__)

logger = logging.getLogger("App")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_STORAGE_URI = os.environ["AIP_STORAGE_URI"]  # Vertex AI sets this env with path to the model artifact
logger.info(f"MODEL PATH: {AIP_STORAGE_URI}")
UPLOAD_FOLDER = './data'

DATASET_BUCKET = os.environ.get("DATASET_BUCKET", 'bird-project-mlops-vertex-data')

MODEL_PATH = "model/model.onnx"


def decode_gcs_url(url: str) -> Tuple[str, str, str]:
    """
        Split a google cloud storage path such as: gs://bucket_name/dir1/filename into
        bucket and path after the bucket: bucket_name, dir1/filename
        :param url: storage url
        :return: bucket_name, blob
        """
    bucket = re.findall(r'gs://([^/]+)', url)[0]
    blob = url.split('/', 3)[-1]
    return bucket, blob

def download_artifacts(artifacts_uri:str = AIP_STORAGE_URI, local_path:str = MODEL_PATH):
    model_uri = os.path.join(artifacts_uri, "model_onnx")
    logger.info(f"Downloading {model_uri} to {local_path}")
    storage_client = storage.Client()
    src_bucket, src_blob = decode_gcs_url(model_uri)
    source_bucket = storage_client.bucket(src_bucket)
    source_blob = source_bucket.blob(src_blob)
    source_blob.download_to_filename(local_path)
    logger.info(f"Downloaded.")

# Flask route for Liveness checks
@app.route(HEALTH_ROUTE, methods=['GET'])
def health_check():
    return "I am alive", 200

@app.route(PREDICT_ROUTE, methods=['POST'])
def predict():
    if not request.is_json:
        return "Invalid input, JSON expected", 400

    try:
        instances = request.get_json()['instances']
    except Exception as e:
        return f"Invalid JSON: {str(e)}", 400

    results = []
    for instance in instances:
        try:
            content = instance['content']
            filename = instance['filename']
            label = instance['label']

            audio_bytes = base64.b64decode(content)
            final_path = os.path.join(UPLOAD_FOLDER, label, filename)
            os.makedirs(os.path.join(UPLOAD_FOLDER, label), exist_ok=True)
            with open(final_path, 'wb') as f:
                f.write(audio_bytes)
            
            try:
                audio, _ = librosa.load(final_path, sr=sample_rate, mono=False)
            except:
                return "Corrupted file!", 400
            
            mono_audio = False
            if len(audio.shape) == 1: # Mono channel audio
                mono_audio = True
                audio = audio[None, :]

            result = ort_sess.run(None, {'input': audio})[0]

            if mono_audio:
                result = result[0]
            else:
                result = np.max(result, axis=0)

            if label != '':
                file_upload_path = f"train_data_s3/{label}/{filename}"

                storage_client = storage.Client()
                bucket = storage_client.bucket(DATASET_BUCKET)

                dataset_df_path = os.path.join(UPLOAD_FOLDER, 'dataset.csv')

                df_blob = bucket.blob('dataset.csv')
                df_blob.download_to_filename(dataset_df_path)

                df = pd.read_csv(dataset_df_path)
                if f"gs://{DATASET_BUCKET}/{file_upload_path}" not in df['path']:
                    file_blob = bucket.blob(file_upload_path)
                    file_blob.upload_from_filename(final_path, if_generation_match=0)

                    new_row = {'path': f"gs://{DATASET_BUCKET}/{file_upload_path}", 'label': label, 'trained_on': False}
                    df = df._append(new_row, ignore_index = True)
                    df.to_csv(dataset_df_path, index=False)
                    
                    bucket.delete_blob('dataset.csv')

                    sleep(5)

                    df_blob.upload_from_filename(dataset_df_path, if_generation_match=0)
        except Exception as e:
            results.append({"error": str(e)})

    return jsonify(results), 200


if __name__ == '__main__':
    sample_rate = int(os.environ.get("SAMPLE_RATE", 32000))

    if not os.path.exists(os.path.basename(MODEL_PATH)):
        os.makedirs(os.path.basename(MODEL_PATH))

    download_artifacts()

    ort_sess = ort.InferenceSession(MODEL_PATH)

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=False, host='0.0.0.0')
