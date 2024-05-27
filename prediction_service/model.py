from torch import nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


class BaselineBirdClassifier(nn.Module):
    def __init__(self, n_classes: int, sr=16000, f_min=20, n_fft=1024, n_mels=64, hop_length=256, dropout_prob=0.1):
        super().__init__()
        self.feature_extractor = nn.Sequential(MelSpectrogram(sample_rate=sr,
                                                              f_min=f_min,
                                                              n_fft=n_fft,
                                                              n_mels=n_mels,
                                                              hop_length=hop_length), 
                                                              AmplitudeToDB())
        
        self.backbone = nn.LSTM(n_mels, 32, 3, bidirectional=True, batch_first=True, dropout=dropout_prob/2)
        self.head = nn.Sequential(nn.Dropout(dropout_prob), 
                                  nn.Linear(64, 16), 
                                  nn.ReLU(), 
                                  nn.Dropout(dropout_prob),
                                  nn.Linear(16, n_classes),
                                  nn.Sigmoid())
        self.sr = sr
        
    def forward(self, audio_samples, return_spec=False):
        if len(audio_samples.size()) == 1:
            audio_samples = audio_samples.unsqueeze(0)
        
        audio_features = self.feature_extractor(audio_samples)

        if return_spec:
            return audio_features
        
        audio_features = audio_features.permute(0, 2, 1)
        
        features_extracted = self.backbone(audio_features)[0][:, -1]

        output = self.head(features_extracted)

        return output
