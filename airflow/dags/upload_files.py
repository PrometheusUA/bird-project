from datetime import datetime, timedelta
from airflow import DAG
# Import below module to work with S3 operator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateObjectOperator
import pandas as pd
import random

import os

data_base_path = os.path.realpath('./data/')


dag = DAG(
    'upload_files_to_s3',
    start_date=datetime(2024, 5, 1),
    schedule="@daily",
    catchup=False,  # Set to True if you want historical DAG runs upon creation
)

def upload_to_s3(**kwargs):
    print(f'{data_base_path=}')

    selected_files = pd.read_csv(os.path.join(data_base_path, 'train_metadata.csv')).sample(random.randint(3, 20))['filename']
 
    for filename in selected_files:
        file_path = os.path.join(data_base_path, 'train_audio', filename)

        print(f'{file_path=}')

        with open(file_path, 'rb') as f:
            upload_to_s3 = S3CreateObjectOperator(
                task_id="upload_to_s3",
                aws_conn_id= 'AWS_MAIN_CONN',
                s3_bucket='bird-project-bucket',
                s3_key=filename,
                data=f.read(), 
            )

        try:        
            upload_to_s3.execute(context=kwargs)
        except ValueError:
            print("Warning! Duplicating entry!")

upload_to_s3 = PythonOperator(
    task_id='upload_to_s3',
    python_callable=upload_to_s3,
    provide_context=True,
    dag=dag, 
)

upload_to_s3