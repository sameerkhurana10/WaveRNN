import numpy as np
import librosa


class DSP(object):
    def __init__(self, hparams):
        self.sample_rate = hparams.sample_rate
        self.n_fft = hparams.n_fft
        self.fft_bins = self.n_fft // 2 + 1
        self.num_mels = hparams.num_mels
        self.hop_length = int(self.sample_rate * hparams.hop_period)
        self.win_length = int(self.sample_rate * hparams.win_period)
        self.fmin = hparams.fmin
        self.min_level_db = hparams.min_level_db
        self.ref_level_db = hparams.ref_level_db
        self.mel_basis = None
        hparams.hop_length = self.hop_length  # TODO: this is bad style, ideally should be refactored

    def load_wav(self, filename, encode=True) :
        x = librosa.load(filename, sr=self.sample_rate)[0]
        if encode:
            x = self.encode_16bits(x)
        return x

    def save_wav(self, y, filename):
        librosa.output.write_wav(filename, y.astype(np.float), self.sample_rate)

    @staticmethod
    def split_signal(x):
        unsigned = x + 2**15
        coarse = unsigned // 256
        fine = unsigned % 256
        return coarse, fine

    @staticmethod
    def combine_signal(coarse, fine):
        return coarse * 256 + fine - 2**15

    @staticmethod
    def encode_16bits(x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    def linear_to_mel(self, spectrogram):
        if self.mel_basis is None:
            self.mel_basis = self.build_mel_basis()
        return np.dot(self.mel_basis, spectrogram)

    def build_mel_basis(self):
        return librosa.filters.mel(self.sample_rate, self.n_fft, n_mels=self.num_mels, fmin=self.fmin)

    def normalize(self, S):
        return np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)

    def denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db

    @staticmethod
    def amp_to_db(x):
        return 20 * np.log10(np.maximum(1e-5, x))

    @staticmethod
    def db_to_amp(x):
        return np.power(10.0, x * 0.05)

    def spectrogram(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.ref_level_db
        return self.normalize(S)

    def melspectrogram(self, y):
        D = self.stft(y)
        S = self.amp_to_db(self.linear_to_mel(np.abs(D)))
        return self.normalize(S)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
