import os

import torch
from flask import Flask, jsonify, request
import librosa

from model import BaselineBirdClassifier


app = Flask(__name__)
model = None


@app.route('/predict', methods=['POST'])
def example():
    if 'file' not in request.files:
        return "File not provided", 400
    
    file = request.files['file']
    if file.filename == '':
        return "File not selected", 400

    if not file.filename.endswith('.ogg'):
        return "Wrong format, should be .ogg", 400

    final_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(final_path)
    except:
        return "Something went wrong", 500
    
    try:
        audio, sr = librosa.load(final_path, sr=model.sr)
    except:
        return "Corrupted file!", 400

    result = model(torch.tensor(audio).unsqueeze(0))[0]
    return jsonify(result), 200


if __name__ == '__main__':
    model_weights_path = os.environ.get('WEIGHTS_PATH', '../data/models/baseline.pt')
    sample_rate = os.environ.get("SAMPLE_RATE", 32000)
    model_classes_count = os.environ.get("CLASSES_COUNT", 100)
    upload_folder = os.environ.get('UPLOAD_FOLDER', './data')

    model = BaselineBirdClassifier(model_classes_count, sample_rate)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    app.config['UPLOAD_FOLDER'] = upload_folder
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    app.run(debug=False, host='0.0.0.0')
