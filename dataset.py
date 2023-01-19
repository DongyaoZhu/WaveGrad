import numpy as np
from multiprocessing import cpu_count
import mindspore as ms

from ljspeech import LJSpeechTTS
FEATURE_POSTFIX = '_feature.npy'
WAV_POSTFIX = '_wav.npy'


class DistributedSampler:
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(dataset)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xFFFFFFFF
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len)
        else:
            indices = np.arange(self.dataset_len)
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.dataset_len


def create_base_dataset(
    ds,
    rank: int = 0,
    group_size: int = 1,
):
    input_columns = ["audio", "text"]
    sampler = DistributedSampler(ds, rank, group_size, shuffle=True)
    ds = ms.dataset.GeneratorDataset(
        ds,
        column_names=input_columns,
        sampler=sampler
    )
    return ds


from hparams import hps
beta = hps.noise_schedule
noise_level = np.cumprod(1 - beta) ** 0.5
noise_level = np.concatenate([[1.0], noise_level], axis=0).astype(np.float32)


def diffuse(x):
    s = np.random.randint(1, hps.noise_schedule_S + 1)
    l_a, l_b = noise_level[s - 1], noise_level[s]
    r = np.random.rand()
    noise_scale = (l_a + r * (l_b - l_a)).astype(np.float32)
    noise = np.random.randn(*(x.shape)).astype(np.float32)
    noisy_audio = noise_scale * x + (1.0 - noise_scale ** 2) ** 0.5 * noise

    return noisy_audio.astype(np.float32), noise_scale, noise


def create_dataset(data_path, manifest_path, batch_size, is_train=True, rank=0, group_size=1):
    ds = LJSpeechTTS(
        data_path=data_path,
        manifest_path=manifest_path,
        is_train=is_train,
    )
    ds = create_base_dataset(ds, rank=rank, group_size=group_size)

    def read_feat(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        x = np.load(filename.replace('.wav', WAV_POSTFIX))
        c = np.load(filename.replace('.wav', FEATURE_POSTFIX))
        return x, c

    input_columns = ['audio', 'spectrogram']
    ds = ds.map(
        input_columns=['audio'],
        output_columns=input_columns,
        column_order=input_columns,
        operations=[read_feat],
        num_parallel_workers=cpu_count(),
    )

    output_columns = ['noisy_audio', 'noise_scale', 'noise', 'spectrogram']
    def batch_collate(audio, spectrogram, unused_batch_info=None):
        batch_noisy_audio, batch_noise_scale, batch_noise, batch_spectrogram = [], [],  [], []
        samples_per_frame = hps.hop_samples
        for x, c in zip(audio, spectrogram):
            start = np.random.randint(0, c.shape[1] - hps.crop_mel_frames)
            end = start + hps.crop_mel_frames
            c = c[:, start:end]
            batch_spectrogram.append(c)

            start *= samples_per_frame
            end *= samples_per_frame
            x = x[start: end]
            x = np.pad(x, (0, (end-start) - len(x)), mode='constant')

            noisy_audio, noise_scale, noise = diffuse(x)

            batch_noisy_audio.append(noisy_audio)
            batch_noise_scale.append(noise_scale)
            batch_noise.append(noise)

        batch_noisy_audio = np.stack(batch_noisy_audio)
        batch_noise_scale = np.stack(batch_noise_scale)
        batch_noise = np.stack(batch_noise)
        batch_spectrogram = np.stack(batch_spectrogram)

        return batch_noisy_audio, batch_noise_scale, batch_noise, batch_spectrogram

    ds = ds.batch(
        batch_size, 
        per_batch_map=batch_collate,
        input_columns=input_columns,
        output_columns=output_columns,
        column_order=output_columns,
        drop_remainder=True,
        python_multiprocessing=False,
        num_parallel_workers=8
    )

    return ds


if __name__ == '__main__':
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='CPU')#, device_id=6)
    ds = create_dataset(hps.data_path, hps.manifest_path, hps.batch_size)
    it = ds.create_dict_iterator()
    for nb, d in enumerate(it):
        print('nb:', nb)
        for k, v in d.items():
            print('k:', k, 'v:', v.shape)
        break