import librosa
import numpy as np
import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
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
    
def obtain_dataloaders(path, validation_fraction = 0.1, train_batch_size = 16, sample_rate = SAMPLE_RATE, sample_len_sec = SAMPLE_LEN_SEC):
    all_files = glob(os.path.join(path, '**/*.ogg'))

    all_df = pd.DataFrame({'file_path': all_files})
    all_df['class'] = all_df['file_path'].apply(lambda filepath: os.path.basename(os.path.dirname(filepath)))

    CLASS2ID = {classname: i for i, classname in enumerate(all_df['class'].unique())}
    ID2CLASS = {i: classname for classname, i in CLASS2ID.items()}

    all_df['class_id'] = all_df['class'].apply(CLASS2ID.get)

    val_df = all_df.sample(int(validation_fraction * len(all_df)))
    train_df = all_df.loc[~all_df.index.isin(val_df.index)]

    train_ds = AudioDataset(train_df['file_path'].tolist(), train_df['class_id'].tolist(), sample_len=sample_len_sec, sr=sample_rate)
    val_ds = AudioDataset(val_df['file_path'].tolist(), val_df['class_id'].tolist(), sample_len=None, sr=sample_rate)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader, (CLASS2ID, ID2CLASS)
