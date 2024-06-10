import os

import mlflow
import torch
import numpy as np
import onnx

from flask import Flask
from tqdm import tqdm
from time import time

from code_base.model import BaselineBirdClassifier
from code_base.datasets import obtain_dataloaders
from code_base.download_data_s3 import download_bucket
from code_base.utils import obtain_metrics


app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

@app.route('/train', methods=['POST'])
def train():
    download_bucket(os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY"),
                    '', os.environ.get("TRAIN_DATA_PATH"), os.environ.get("DATA_BUCKET_NAME"))
    
    val_frac = float(os.environ.get("VAL_FRAC", 0.1))
    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    sample_rate = int(os.environ.get("SAMPLE_RATE", 32000))
    learning_rate = float(os.environ.get("LEARNING_RATE", 1e-3))
    epochs_count = int(os.environ.get("EPOCHS_COUNT", 4))
    eval_every_steps = int(os.environ.get("EVAL_EVERY_STEPS", 50))
    model_save_path = os.environ.get("MODEL_SAVE_PATH")
    sample_len_sec = int(os.environ.get("SAMPLE_LEN_SEC"))

    train_loader, val_loader, (CLASS2ID, ID2CLASS) = obtain_dataloaders(os.environ.get("TRAIN_DATA_PATH"), val_frac, batch_size,
                                                                        sample_rate=sample_rate, sample_len_sec=sample_len_sec)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BaselineBirdClassifier(len(CLASS2ID), sr=sample_rate).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

    experiment_name = f"/Users/{os.environ.get('DATABRICKS_USERNAME')}/{os.environ.get('EXPERIMENT_NAME')}"

    try:
        mlflow.create_experiment(experiment_name, artifact_location=os.environ.get("ARTIFACTS_URL"))
    except mlflow.exceptions.RestException:
        pass

    mlflow.set_experiment(experiment_name)

    os.makedirs(model_save_path, exist_ok=True)

    with mlflow.start_run() as run:

        mlflow.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs_count": epochs_count,
            "sample_rate": sample_rate
        })

        batch_num = 0

        min_eval_loss = np.inf
        corresp_train_loss = np.inf
        best_loss_metrics = None

        training_start_time = time()

        for epoch in tqdm(range(epochs_count), desc='Epoch'):
            running_loss = 0.
            last_loss = 0.

            for audios, labels in train_loader:
                audios = audios.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(audios)

                loss = loss_fn(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                if batch_num % eval_every_steps == eval_every_steps - 1:
                    last_loss = running_loss / eval_every_steps
                    print(f'Batch {batch_num + 1}. Loss: {last_loss:.6f}.', end=' ')
                    running_loss = 0.

                    model.eval()
                    eval_running_loss = 0.
                    outputs_list = []
                    labels_list = []
                    with torch.no_grad():
                        for audios, labels in val_loader:
                            audios = audios.to(device)
                            labels = labels.to(device)

                            outputs = model(audios)
                            loss = loss_fn(outputs, labels)

                            eval_running_loss += loss.item()
                            outputs_list.append(outputs.cpu().numpy())
                            labels_list.append(labels.cpu().numpy())
                    
                    eval_running_loss = eval_running_loss/len(val_loader.dataset)

                    print(f'Val loss: {eval_running_loss:.6f}.')                

                    if eval_running_loss < min_eval_loss:
                        min_eval_loss = eval_running_loss
                        corresp_train_loss = last_loss
                        print("Saving the model")

                        outputs = np.concatenate(outputs_list, axis=0)
                        labels = np.concatenate(labels_list, axis=0)

                        best_loss_metrics = obtain_metrics(labels, outputs)

                        torch.save(model.state_dict(), os.path.join(model_save_path, f'baseline-{len(CLASS2ID)}.pt'))

                    model.train()
                batch_num += 1

        mlflow.log_metric("train_time_sec", time() - training_start_time)
        mlflow.log_metric("min_val_loss", min_eval_loss)
        mlflow.log_metric("train_loss", corresp_train_loss)
        mlflow.log_metrics(best_loss_metrics)

        print("Exporting to ONNX")

        model.load_state_dict(torch.load(os.path.join(model_save_path, f'baseline-{len(CLASS2ID)}.pt'), map_location=torch.device('cpu')))
        model.eval()

        torch_input = torch.randn(8, sample_rate*sample_len_sec)
        torch.onnx.export(model.cpu(),
                        torch_input,
                        os.path.join(model_save_path, f'baseline-{len(CLASS2ID)}.onnx'),
                        export_params=True,
                        do_constant_folding=True,
                        input_names = ['input'],
                        output_names = ['output'],
                        dynamic_axes={'input' : {0: 'batch_size', 1: 'sample_length'},
                                    'output' : {0: 'batch_size'}}
        )

        print("ONNX export finished")

        onnx_model = onnx.load(os.path.join(model_save_path, f'baseline-{len(CLASS2ID)}.onnx'))
        onnx.checker.check_model(onnx_model)

        print("ONNX model checked")

        mlflow.log_artifact(os.path.join(model_save_path, f'baseline-{len(CLASS2ID)}.onnx'))

        return "Model created", 201


if __name__ == '__main__':
    mlflow.login("databricks", interactive=False)
    
    app.run(debug=False, host='0.0.0.0')
