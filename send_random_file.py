import base64
from typing import Dict, List, Union
import os

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import pandas as pd
from glob import glob

def convert_audio_to_base64(file_path):
    with open(file_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
    return encoded_string


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))


if __name__ == '__main__':
    endpoint_id = "2840344198777929728"
    project = 'bird-project-mlops-vertex'

    aiplatform.init()

    all_files = glob('./data/train_audio/*/*.ogg')
    files_df = pd.DataFrame({'path': all_files})
    selected_file = files_df.sample(n=1).iloc[0]['path']

    print(f"Chosen file: {selected_file}")

    base64_audio = convert_audio_to_base64(selected_file)
    print("Loaded and converted audio")
    instances = {
        "content": base64_audio,
        "label": os.path.basename(os.path.dirname(selected_file)),
        "filename": os.path.basename(selected_file),
    }

    predict_custom_trained_model_sample(project, endpoint_id, instances)
