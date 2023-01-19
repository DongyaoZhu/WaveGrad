# WaveGrad

## MindSpore implementation of [WaveGrad](https://arxiv.org/abs/2009.00713), a diffusion based vocoder model for text-to-speech systems. 

## Demo

Generated from LJ010-0142.wav, test set, LJSpeech-1.1 ("Be this as it may, the weapon used was only an ordinary axe, which rather indicates that force, not skill, was employed."):

<audio src="results/1000000_predicted_LJ010-0142_1000.wav" preload="none" controls loop></audio>
![compare_lj](results/1000000_LJ010-0142.gif?raw=true "lj")

Generated from [FastSpeech2](https://github.com/ming024/FastSpeech2) ("This is a MindSpore implementation of the WaveGrad model, a diffusion based vocoder model for text to speech systems. Many thanks to Open I for computational resources!"):

<audio src="results/1000000_predicted_fs_1000.wav" preload="none" controls loop></audio>
![compare_fs2](results/1000000_fs.gif?raw=true "fs2")

## **Dependencies**

1. `pip install -r requirements.txt`
2. Install [mindspore](https://www.mindspore.cn/install) based on your platform.
3. (Optional) Install `mpirun` for distributed training.

### Generate from your data

From wav files:

`python reverse.py --restore model_1000000.ckpt --wav LJ010-0142.wav --save results --device_target Ascend --device_id 0 --plot`

From melspectrograms:

`python reverse.py --restore model_1000000.ckpt --mel fs.npy --save results --device_target Ascend --device_id 0 --plot`

## Pretrained Models

| Model | Dataset | Checkpoint | Total Batch Size | Num Frames | Hardware |
| -----| ----- | -----| -----| -----| -----|
| WaveGrad (base) | LJ-Speech1.1 | [1M steps](https://download.mindspore.cn/toolkits/mindaudio/wavegrad/model_1000000.ckpt) | 256 | 30 | 8 \times Ascend |
| WaveGrad (base) | AiShell | [TODO]() | 256 | 30 | 8 \times Ascend |
| FastSpeech2 | LJ-Speech1.1 | [160K steps]() | 64 | - | 8 \times GPU

For FastSpeech2 model, we skipped the audio preprocess part and directly used this repo's preprocessed melspectrograms. We used the alignments following readme in the link.

## Train your own model

### Step 0 (Data)

#### 0.0

Download [LJSpeech-1.1](http://keithito.com/LJ-Speech-Dataset/) to `./data/`.

#### 0.1

Preprocess data to get a "_wav.npy" and "_feature.npy" for each ".wav" file in your dataset folder. Set your `data_path` and 
`manifest_path` in `base.yaml`. You can now run the following command:

`python preprocess.py --device_target CPU --device_id 0`

### Step 1 (Train)

#### 1.1 Train on local server

Set up device information:
```
export MY_DEVICE=Ascend # options: [Ascend, GPU]
export MY_DEVICE_NUM=8
export MY_DEVICE_ID=0
```

Other training and model parameters can be set in `base.yaml`. 

Train on multiple cards: (each card will have a batch size of hparams.batch_size // MY_DEVICE_NUM)
```
nohup mpirun --allow-run-as-root -n $MY_DEVICE_NUM python train.py --device_target $MY_DEVICE --is_distributed True --context_mode graph > train_distributed.log &
```

Train on 1 card:
```
nohup python train.py --device_target $MY_DEVICE --device_id $MY_DEVICE_ID --context_mode graph > train_single.log &
```

#### 1.2 Train on 8 Ascend cards on [openi](https://openi.pcl.ac.cn/)

A quick guide on how to use openi:
1. git clone a repo
2. create a train task
3. locally preprocess the data, zip it, and upload to your job's dataset
4. set task options as follows:

start file: 

`train.py`

Run Parameter:	

`is_openi`, `'True`

`is_distributed`, `'True`

`device_target`, `'Ascend`

`context_mode`, `'graph`

### Implementation details

The interpolation operator in both downsample and upsample blocks are replaced by a simple repeat operator, then divided by repeat factor.

Some additions in UBlock are divided by a constant $\sqrt{2}$ to avoid potential numerical overflow.

### Acknowlegements

Some materials helpful for understanding diffusion models:
- [youtube](https://www.youtube.com/watch?v=nv-WTeKRLl0)
- [youtube](https://www.youtube.com/watch?v=HoKDTa5jHvg)
- [youtube](https://www.youtube.com/watch?v=XCUlnHP1TNM)
- [math](lilianweng.github.io/posts/2021-07-11-diffusion-models)

Some repositories that inspired this implementation:
- [lmnt](https://github.com/lmnt-com/wavegrad)
- [MindAudio](https://github.com/mindspore-lab/mindaudio)
- [FastSpeech2](https://github.com/ming024/FastSpeech2)

Computational Resources:
- [openi](https://openi.pcl.ac.cn/)

### License

GNU General Public License v2.0
