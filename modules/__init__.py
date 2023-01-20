import mindspore as ms

if ms.__version__ in {'1.8.1', '1.7.0'}:
    from modules.wavegrad_v181 import WaveGrad, WaveGradWithLoss
elif ms.__version__ in {'1.9.0'}:
    from modules.wavegrad_v190 import WaveGrad, WaveGradWithLoss
else:
    raise NotImplementedError('unknown mindspore version: %s' % ms.__version__)
