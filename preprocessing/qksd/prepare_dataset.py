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


def load_random_noise(noise_paths, num_sample=32000):
    num_noise = len(noise_paths)
    path = noise_paths[rng.integers(0, num_noise)]

    while True:
        waveform, sample_rate = torchaudio.load(path, normalize=True)
        noise_num_sample = waveform.shape[1]
        if noise_num_sample < 32000:
            print ('Error: noise too short!!!')
            exit(1)
        noise_start_sample = rng.integers(0, noise_num_sample - num_sample)
        waveform = waveform[:, noise_start_sample:noise_start_sample + num_sample]
        if torch.count_nonzero(waveform) != 0:
            break

    return waveform

def prepare_qualcomm_kws(train_test_split=0.8, selected_word='hey_snapdragon', normalize=True, n_features_bin=10, num_augmented=10):
    # selected_word can be hey_android, hey_snapdragon, hi_galaxy, hi_lumina

    hotword_folder = f'../../data/qksd/qualcomm_keyword_speech_dataset/{selected_word}'
    hotword_paths = []
    for speaker_name in os.listdir(hotword_folder):
        spaeker_path = os.path.join(hotword_folder, speaker_name)
        if os.path.isdir(spaeker_path):
            for filename in os.listdir(spaeker_path):
                path = os.path.join(hotword_folder, speaker_name, filename)
                if not path.endswith('.wav'):
                    continue
                hotword_paths.append(path)

    speech_folder = f'../../data/MS-SNSD-master/clean_train'
    speech_paths = [os.path.join(speech_folder, filename) for filename in os.listdir(speech_folder) if filename.endswith('.wav')]

    noise_folder = f'../../data/MS-SNSD-master/noise_train'
    noise_paths = [os.path.join(noise_folder, filename) for filename in os.listdir(noise_folder) if filename.endswith('.wav')]

    hotword_sample = []
    for i, path in enumerate(hotword_paths):
        print (f'Processing hot word [{i+1}/{len(hotword_paths)}]')
        waveform, sample_rate = torchaudio.load(path, normalize=True)
        if normalize:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate,
                                                                                [['gain', '-n']],
                                                                                channels_first=True)
        # plot_waveform(waveform, sample_rate)

        for _ in range(num_augmented):
            num_sample = waveform.shape[1]
            left_pad_sample = rng.integers(0, 32000 - num_sample)
            right_pad_sample = 32000 - num_sample - left_pad_sample
            waveform_pad = torch.nn.functional.pad(waveform, (left_pad_sample, right_pad_sample), 'constant', 0)
            # plot_waveform(waveform, sample_rate)

            noise = load_random_noise(noise_paths, 32000)
            snr = rng.integers(3, 20, endpoint=True)
            waveform_augmented = torchaudio.functional.add_noise(waveform_pad, noise, torch.full((1, ), snr))

            transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                             n_fft=800,
                                                             n_mels=n_features_bin,
                                                             f_min=100
                                                             )  # normalized=True, norm='slaney', f_min=125
            feature = transform(waveform_augmented)
            feature = librosa.power_to_db(feature[0], ref=np.max)
            # torchaudio.save(f'augment/{filename}_{i}.wav', waveform_augmented, sample_rate)
            # plot_spectrogram(feature)

            hotword_sample.append(feature)

    unknown_sample = []
    for i, path in enumerate(rng.permutation(speech_paths)[:len(hotword_sample)]):
        print(f'Processing unknown word [{i + 1}/{len(hotword_sample)}]')
        waveform, sample_rate = torchaudio.load(path, normalize=True)
        if normalize:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate,
                                                                                [['gain', '-n']],
                                                                                channels_first=True)
        # plot_waveform(waveform, sample_rate)

        if waveform.shape[1] < 32000:
            waveform = torch.nn.functional.pad(waveform, (0, 32000 - waveform.shape[1]), 'constant', 0)
        else:
            start_sample = waveform.shape[1] // 2 - 16000
            waveform = waveform[:, start_sample:start_sample + 32000]
        # plot_waveform(waveform, sample_rate)

        noise = load_random_noise(noise_paths, 32000)
        snr = rng.integers(3, 20, endpoint=True)
        waveform_augmented = torchaudio.functional.add_noise(waveform, noise, torch.full((1,), snr))

        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=800,
                                                         n_mels=n_features_bin,
                                                         f_min=100
                                                         )  # normalized=True, norm='slaney', f_min=125
        feature = transform(waveform_augmented)
        feature = librosa.power_to_db(feature[0], ref=np.max)
        # torchaudio.save(f'augment/{filename}_{i}.wav', waveform_augmented, sample_rate)
        # plot_spectrogram(feature)

        unknown_sample.append(feature)

    num_sample_per_class = len(hotword_sample)
    hotword_sample = np.array(hotword_sample).reshape((num_sample_per_class, -1)).astype(np.float32)
    hotword_sample = np.clip((hotword_sample + 64) / 64, 0, 1)
    unknown_sample = np.array(unknown_sample).reshape((num_sample_per_class, -1)).astype(np.float32)
    unknown_sample = np.clip((unknown_sample + 64) / 64, 0, 1)

    permute_index = rng.permutation(num_sample_per_class)
    hotword_sample = hotword_sample[permute_index]
    unknown_sample = unknown_sample[permute_index]

    train_count = int(train_test_split * num_sample_per_class)
    test_count = num_sample_per_class - train_count
    x_train = np.vstack([hotword_sample[:train_count], unknown_sample[:train_count]])
    y_train = np.concatenate([np.zeros(train_count), np.ones(train_count)])
    x_test = np.vstack([hotword_sample[train_count:], unknown_sample[train_count:]])
    y_test = np.concatenate([np.zeros(test_count), np.ones(test_count)])

    np.save(f'../../data/qksd/{selected_word}/x_train.npy', x_train)
    np.save(f'../../data/qksd/{selected_word}/y_train.npy', y_train)
    np.save(f'../../data/qksd/{selected_word}/x_test.npy', x_test)
    np.save(f'../../data/qksd/{selected_word}/y_test.npy', y_test)

if __name__ == '__main__':
    prepare_qualcomm_kws()