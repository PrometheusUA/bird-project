import librosa
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score

from torch.utils.data import Dataset
from glob import glob


SAMPLE_LEN_SEC = 10
SAMPLE_RATE = 32000

class AudioDataset(Dataset):
    def __init__(self, paths, labels=None, sample_len=SAMPLE_LEN_SEC, sr=SAMPLE_RATE):
        assert labels is None or len(paths) == len(labels), "Data and targets should be of the same samples count"
        self.paths = paths
        self.labels = labels
        self.sample_len = sample_len
        self.sr = sr

    def __getitem__(self, i):
        audio, sr = librosa.load(self.paths[i], sr=self.sr)

        if self.sample_len is not None:
            desired_len = self.sample_len * sr
            if len(audio) >desired_len:
                audio = audio[:desired_len]
            else:
                audio =  np.pad(audio, (0, desired_len - len(audio)))

        if self.labels is not None:
            return audio, self.labels[i]
        else:
            return audio

    def __len__(self):
        return len(self.paths)

def obtain_metrics(y_true, y_pred_prob):
    y_pred = y_pred_prob.argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=1)
    
    best_loss_metrics = {
        "accuracy": accuracy,
        "macro_f1": f1,
        "macro_precision": prec,
        "macro_recall": rec,
    }

    return best_loss_metrics
