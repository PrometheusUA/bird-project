import os

import onnxruntime as ort
import numpy as np
from flask import Flask, jsonify, request
import librosa


app = Flask(__name__)


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

    return jsonify({"probabilities": result.tolist()}), 200


if __name__ == '__main__':
    model_path = os.environ.get('MODEL_PATH', '../data/models/baseline.onnx')
    sample_rate = int(os.environ.get("SAMPLE_RATE", 32000))
    upload_folder = os.environ.get('UPLOAD_FOLDER', './data')

    ort_sess = ort.InferenceSession(model_path)

    app.config['UPLOAD_FOLDER'] = upload_folder
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    app.run(debug=False, host='0.0.0.0')
