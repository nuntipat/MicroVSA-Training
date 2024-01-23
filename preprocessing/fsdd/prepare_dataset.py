import torch
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa

rng = np.random.default_rng()


def plot_waveform(waveform, sr, title="Waveform"):
    # From: https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#melspectrogram
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show()


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    # From: https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#melspectrogram
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show()


def prepare_FSDD(train_test_split=0.8, feature_type='mel', num_windows=20, n_features_bin=40, normalize=True,
             trim_start=False, num_sample_include_pad=16000, pad_left=False):
    x = []
    y = []
    for filename in os.listdir('../../data/fsdd/free-spoken-digit-dataset-1.0.10/recordings/'):
        path = f'../../data/fsdd/free-spoken-digit-dataset-1.0.10/recordings/{filename}'

        waveform, sample_rate = torchaudio.load(path, normalize=True)
        if normalize:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, [['gain', '-n']], channels_first=True)
        # plot_waveform(waveform, sample_rate)

        num_sample = waveform.shape[1]
        if num_sample_include_pad != -1:
            if num_sample < num_sample_include_pad:
                left_pad_sample = rng.integers(0, num_sample_include_pad - num_sample) if pad_left else 0
                right_pad_sample = num_sample_include_pad - num_sample - left_pad_sample
                waveform = torch.nn.functional.pad(waveform, (left_pad_sample, right_pad_sample), 'constant', 0)
            else:
                waveform = waveform[:, :num_sample_include_pad]
        # plot_waveform(waveform, sample_rate)

        if feature_type == 'mel':
            transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                             n_mels=n_features_bin,
                                                             f_min=100)  # normalized=True, norm='slaney', f_min=125
            feature = transform(waveform)
            feature = librosa.power_to_db(feature[0])
            if trim_start:
                m = np.nonzero(np.max(feature, axis=0) >= 10)[0]
                if m.shape[0] > 0:
                    feature = feature[:, m[0]:]
            feature = feature[:, :num_windows]
            # plot_spectrogram(feature)
        elif feature_type == 'mfcc':    # untested
            transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                log_mels=True,
                melkwargs={"n_fft": 512, "win_length": 240, "hop_length": 160, "n_mels": 40, "f_min": 20, "f_max": 4000},
            )
            feature = transform(waveform)[0][2:14].numpy()

        x.append(feature.T)
        y.append(int(filename.split('_')[0]))

    num_sample = len(x)
    x = np.array(x).reshape((num_sample, -1)).astype(np.float32)
    x = np.clip((x + 64) / 64, 0, 1)
    y = np.array(y).astype(int)

    permute_index = rng.permutation(num_sample)
    x = x[permute_index]
    y = y[permute_index]

    train_count = int(train_test_split * num_sample)

    np.save(f'../../data/fsdd/x_train.npy', x[:train_count])
    np.save(f'../../data/fsdd/y_train.npy', y[:train_count])
    np.save(f'../../data/fsdd/x_test.npy',  x[train_count:])
    np.save(f'../../data/fsdd/y_test.npy', y[train_count:])

if __name__ == '__main__':
    prepare_FSDD()