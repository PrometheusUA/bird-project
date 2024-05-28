# Bird project

MLOps project based on [BirdCLEF-2024](https://www.kaggle.com/competitions/birdclef-2024/data) competition.

## Folders structure:

Folders structure is as follows:
- `airflow`: this folder contains everything you have to have to run airflow. It is used as a labeling emulator and sends a random bunch of data to the S3 bucket every day. To run it, use `docker-compose up` and then add Amazon Web Services credentials with ID `AWS_MAIN_CONN` via Connections->+->Amazon Web Services.
- `data`: this folder contains:
    - `train_audio` folder and `train_metadata.csv` from [BirdCLEF-2024](https://www.kaggle.com/competitions/birdclef-2024/data).
    - `train_data_s3` folder, which contains data downloaded from S3 to train the model.
    - `models` folder, which contains already train models in .pt and .onnx formats.
    - Several test samples, copied from `train_audio`.
- `ml_base`: this folder contains code base for ML objects (only models for now).
- `notebooks`: several notebooks used for model training and experiments.
- `prediction_service`: Docker file and docker-compose for prediction service run. To run it, use `docker-compose up`.
- `download_data_s3.py`: script to download "labeled" data from AWS S3. Needs `credentials.py` file with access key, based on `credentials-sample.py`.
- `requirements.txt`: Text file with venv packages.
