import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from math import log as ln


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        kwargs['weight_init'] = 'Orthogonal'
        kwargs['has_bias'] = True
        super().__init__(*args, **kwargs)


class PositionalEncoding(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def construct(self, x, noise_level):
        count = self.dim // 2
        step = ms.numpy.arange(count, dtype=noise_level.dtype) / count
        encoding = noise_level.expand_dims(1) * ms.ops.exp(-ln(1e4) * step.expand_dims(0))
        encoding = ms.ops.concat([ms.ops.sin(encoding), ms.ops.cos(encoding)], -1)
        return x + encoding[:, :, None]


class FiLM(nn.Cell):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoding = PositionalEncoding(input_size)
        self.input_conv = nn.Conv1d(input_size, input_size, 3,
            weight_init='XavierUniform',
            has_bias=True,
            pad_mode='pad',
            padding=1
        )
        self.output_conv = nn.Conv1d(input_size, output_size * 2, 3,
            weight_init='XavierUniform',
            has_bias=True,
            pad_mode='pad',
            padding=1
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

    def construct(self, x, noise_scale):
        x = self.input_conv(x)
        x = self.leaky_relu(x)
        x = self.encoding(x, noise_scale)
        shift, scale = ops.split(self.output_conv(x), 1, output_num=2)
        return shift, scale


class UBlock(nn.Cell):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        self.factor = factor
        self.block1 = Conv1d(input_size, hidden_size, 1)
        self.block2_a = Conv1d(input_size, hidden_size, 3, dilation=dilation[0], pad_mode='pad', padding=dilation[0])
        self.block2_b = Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], pad_mode='pad', padding=dilation[1])
        self.block3_a = Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], pad_mode='pad', padding=dilation[2])
        self.block3_b = Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], pad_mode='pad', padding=dilation[3])
        self.leaky_relu = nn.LeakyReLU(0.2)
        # self.upscale1 = nn.SequentialCell([
        #     # nn.Conv1dTranspose(input_size, input_size, 5, stride=self.factor, pad_mode='same', weight_init='XavierUniform'),
        #     # nn.LeakyReLU(0.2),
        #     nn.Conv1d(input_size, input_size, 3, pad_mode='same', weight_init='XavierUniform'),
        # ])
        # self.upscale2 = nn.SequentialCell([
        #     # nn.Conv1dTranspose(input_size, input_size, 5, stride=self.factor, pad_mode='same', weight_init='XavierUniform'),
        #     # nn.LeakyReLU(0.2),
        #     nn.Conv1d(input_size, input_size, 3, pad_mode='same', weight_init='XavierUniform'),
        # ])
        self.const = ms.Tensor(2 ** 0.5, dtype=ms.float32)

    def construct(self, x, film_shift, film_scale):
    
        block1 = self.block1(x)
        block1 = block1.repeat(self.factor, 2) / self.factor

        block2 = self.leaky_relu(x)
        block2 = block2.repeat(self.factor, 2) / self.factor
        block2 = self.block2_a(block2)

        block2 = (film_shift + film_scale * block2) / self.const
        block2 = self.leaky_relu(block2)
        block2 = self.block2_b(block2)

        x = (block1 + block2) / self.const

        block3 = (film_shift + film_scale * x) / self.const
        block3 = self.leaky_relu(block3)
        block3 = self.block3_a(block3)

        block3 = (film_shift + film_scale * block3) / self.const
        block3 = self.leaky_relu(block3)
        block3 = self.block3_b(block3)

        x = (x + block3) / self.const
        return x


class DBlock(nn.Cell):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.conv = nn.SequentialCell([
            nn.LeakyReLU(0.2),
            Conv1d(input_size, hidden_size, 3, dilation=1, pad_mode='pad', padding=1),
            nn.LeakyReLU(0.2),
            Conv1d(hidden_size, hidden_size, 3, dilation=2, pad_mode='pad', padding=2),
            nn.LeakyReLU(0.2),
            Conv1d(hidden_size, hidden_size, 3, dilation=4, pad_mode='pad', padding=4),
        ])
        self.downscale1 = nn.Conv1d(hidden_size, hidden_size,
            kernel_size=self.factor,
            stride=self.factor,
            pad_mode='valid',
            has_bias=True,
            weight_init='XavierUniform',
        )
        self.downscale2 = nn.Conv1d(input_size, input_size,
            kernel_size=self.factor,
            stride=self.factor,
            pad_mode='valid',
            has_bias=True,
            weight_init='XavierUniform',
        )

    def construct(self, x):
        residual = self.residual_dense(x)
        residual = self.downscale1(residual)
        x = self.downscale2(x)
        x = self.conv(x)
        return x + residual


class WaveGrad(nn.Cell):
    def __init__(self, hps):
        super().__init__()
        self.d1 = Conv1d(1, 32, 5, pad_mode='pad', padding=2)
        self.d2 = DBlock(32, 128, 2)
        self.d3 = DBlock(128, 128, 2)
        self.d4 = DBlock(128, 256, 3)
        self.d5 = DBlock(256, 512, 5)
        self.downsample = [self.d1, self.d2, self.d3, self.d4, self.d5]

        self.f1 = FiLM(32, 128)
        self.f2 = FiLM(128, 128)
        self.f3 = FiLM(128, 256)
        self.f4 = FiLM(256, 512)
        self.f5 = FiLM(512, 512)
        self.film = [self.f1, self.f2, self.f3, self.f4, self.f5]

        self.u1 = UBlock(768, 512, 5, [1, 2, 1, 2])
        self.u2 = UBlock(512, 512, 5, [1, 2, 1, 2])
        self.u3 = UBlock(512, 256, 3, [1, 2, 4, 8])
        self.u4 = UBlock(256, 128, 2, [1, 2, 4, 8])
        self.u5 = UBlock(128, 128, 2, [1, 2, 4, 8])
        self.upsample = [self.u1, self.u2, self.u3, self.u4, self.u5]

        self.first_conv = Conv1d(128, 768, 3, pad_mode='pad', padding=1)
        self.last_conv = Conv1d(128, 1, 3, pad_mode='pad', padding=1)

    def forward(self, noisy_audio, noise_scale, spectrogram):
        x = noisy_audio.expand_dims(1)
        downsampled = []
        num_blocks = len(self.film)
        for i in range(num_blocks):
            film = self.film[i]
            layer = self.downsample[i]
            x = layer(x)
            downsampled.append(film(x, noise_scale))

        x = self.first_conv(spectrogram)
        for i in range(num_blocks):
            layer = self.upsample[i]
            film_shift, film_scale = downsampled[num_blocks - 1 - i]
            x = layer(x, film_shift, film_scale)
        x = self.last_conv(x).squeeze(1)
        return x

    def construct(self, noisy_audio, noise_scale, spectrogram):
        return self.forward(noisy_audio, noise_scale, spectrogram)


class WaveGradWithLoss(WaveGrad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.L1Loss()
    
    def construct(self, noisy_audio, noise_scale, noise, spectrogram):
        yh = self.forward(noisy_audio, noise_scale, spectrogram)
        return self.loss_fn(yh, noise)


if __name__ == '__main__':
    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    # ms.context.set_context(mode=ms.context.PYNATIVE_MODE)

    b, c, t = 2, 128, 7200
    x = np.random.random([b, t]).astype(np.float32)
    s = np.random.random([b, ]).astype(np.float32)
    n = np.random.random([b, t]).astype(np.float32)
    c = np.random.random([b, c, t // 300]).astype(np.float32)

    net = WaveGradWithLoss(1)
    y = net(ms.Tensor(x), ms.Tensor(s), ms.Tensor(n), ms.Tensor(c))
    print('y:', y.shape)