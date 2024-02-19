# Efficient Region-Aware Neural Radiance Fields for High-Fidelity Talking Portrait Synthesis
This is the official repository for our ICCV 2023 paper **Efficient Region-Aware Neural Radiance Fields for High-Fidelity Talking Portrait Synthesis**.
### [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Efficient_Region-Aware_Neural_Radiance_Fields_for_High-Fidelity_Talking_Portrait_Synthesis_ICCV_2023_paper.html) | [Project](https://fictionarry.github.io/ER-NeRF/) | [ArXiv](https://arxiv.org/abs/2307.09323) | [Video](https://youtu.be/Gc2d3Z8MMuI)
## Installation

Tested on Ubuntu 18.04, Pytorch 1.12 and CUDA 11.3.

### Install dependency

```bash
conda create -n ernerf python=3.10
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
conda install PyAudio
pip install -r requirements.txt
source /etc/network_turbo
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
```

### Preparation

- Prepare face-parsing model.

  ```bash
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
  ```

- Prepare the 3DMM model for head pose estimation.

先在data_utils/face_tracking下创建3DMM文件夹

下载01_MorphableModel.mat到3DMM文件夹
  ```bash
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/exp_info.npy?raw=true -O data_utils/face_tracking/3DMM/exp_info.npy
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/keys_info.npy?raw=true -O data_utils/face_tracking/3DMM/keys_info.npy
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/sub_mesh.obj?raw=true -O data_utils/face_tracking/3DMM/sub_mesh.obj
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/topology_info.npy?raw=true -O data_utils/face_tracking/3DMM/topology_info.npy
  ```

- Download 3DMM model from [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details):

  ```
  # 1. copy 01_MorphableModel.mat to data_util/face_tracking/3DMM/
  # 2.
    cd data_utils/face_tracking
    python convert_BFM.py
  ```

## Datasets and pretrained models

We get the experiment videos mainly from [AD-NeRF](https://github.com/YudongGuo/AD-NeRF), [DFRF](https://github.com/sstzal/DFRF), [GeneFace](https://github.com/yerfor/GeneFace) and YouTube. Due to copyright restrictions, we can't distribute all of them. You may have to download and crop these videos by youself. Here is an example training video (Obama) from AD-NeRF with the resolution of 450x450. 

```
mkdir -p data/obama
wget https://github.com/YudongGuo/AD-NeRF/blob/master/dataset/vids/Obama.mp4?raw=true -O data/obama/obama.mp4
```

We also provide pretrained checkpoints on the Obama video clip. After completing the data pre-processing step, you can [download](https://github.com/Fictionarry/ER-NeRF/releases/tag/ckpt) and test them by:

```bash
python main.py data/obama/ --workspace trial_obama/ -O --test --ckpt trial_obama/checkpoints/ngp.pth   # head
python main.py data/obama/ --workspace trial_obama_torso/ -O --test --torso --ckpt trial_obama_torso/checkpoints/ngp.pth   # head+torso
```

The test results should be about:

| setting    | PSNR   | LPIPS  | LMD   |
| ---------- | ------ | ------ | ----- |
| head       | 35.607 | 0.0178 | 2.525 |
| head+torso | 26.594 | 0.0446 | 2.550 |

## Usage

### Pre-processing Custom Training Video

* Put training video under `data/<ID>/<ID>.mp4`.

  The video **must be 25FPS, with all frames containing the talking person**. 
  The resolution should be about 512x512, and duration about 1-5 min.

首先`pip install 'protobuf~=3.20.0'`一下，否则会报错。然后开一下代理，可能在下载一个zip文件时会停留不少时间。


* Run script to process the video. (may take several hours)

  ```bash
  python data_utils/process.py data/<ID>/<ID>.mp4
  ```

* Obtain AU45 for eyes blinking
  在openface界面，点击wiki链接进入使用方法，可以点击windows，然后下载64位的，然后直接在powershell中输入` .\download_models.ps1`就可以运行，但是可能报错，需要先将powershell设置一下： `set-executionpolicy remotesigned` 然后需要在download_models.ps1的属性中点击一下解除锁定，即可开始运行下载模型，然后在图形化界面中操作即可。下载完成后最好重启下OpenFaceOffline。在图形化界面中打开视频后会自动分析，分析完成后会自动创建processed文件夹，文件夹内有处理结果
  
  Run `FeatureExtraction` in [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), rename and move the output CSV file to `data/<ID>/au.csv`.

### Audio Pre-process

In our paper, we use DeepSpeech features for evaluation. 

You should specify the type of audio feature by `--asr_model <deepspeech, esperanto, hubert>` when **training and testing**.

* DeepSpeech

  ```bash
  python data_utils/deepspeech_features/extract_ds_features.py --input data/<ID>/aud.wav # save to data/<name>.npy
  ```
  完成后将npy文件后面加个_ds，即aud_ds.npy

* Wav2Vec

  You can also try to extract audio features via Wav2Vec like [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF) by:

  ```bash
  python data_utils/wav2vec.py --wav data/<name>.wav --save_feats # save to data/<name>_eo.npy
  ```

* HuBERT

  In our test, HuBERT extractor performs better for more languages, which has already been used in [GeneFace](https://github.com/yerfor/GeneFace).

  ```bash
  # Borrowed from GeneFace. English pre-trained.
  python data_utils/hubert.py --wav data/<name>.wav # save to data/<name>_hu.npy
  ```

### Train

First time running will take some time to compile the CUDA extensions.

```bash
# train (head and lpips finetune, run in sequence)
# 工作空间即存储一些中间表示的文件夹，会直接创建在主目录下，会自动进行tensorboard记录
python main.py data/obama/ --workspace trial_obama/ -O --iters 100000
python main.py data/obama/ --workspace trial_obama/ -O --iters 125000 --finetune_lips --patch_size 32

# train (torso)
# <head>.pth should be the latest checkpoint in trial_obama
# 这里得保证pth文件在主目录下，而不是放在trial obama文件夹下
python main.py data/obama/ --workspace trial_obama_torso/ -O --torso --head_ckpt <head>.pth --iters 200000
```

### Test

```bash
# test on the test split
python main.py data/obama/ --workspace trial_obama/ -O --test # only render the head and use GT image for torso
python main.py data/obama/ --workspace trial_obama_torso/ -O --torso --test # render both head and torso
```

### Inference with target audio

```bash
# Adding "--smooth_path" may help decrease the jitter of the head, while being less accurate to the original pose.
# npy文件也需要放在主目录下
python main.py data/obama/ --workspace trial_obama_torso/ -O --torso --test --test_train --aud <audio>.npy
```

## Acknowledgement

This code is developed heavily relying on [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF), and also [DFRF](https://github.com/sstzal/DFRF), [GeneFace](https://github.com/yerfor/GeneFace), and [AD-NeRF](https://github.com/YudongGuo/AD-NeRF).  Thanks for these great projects.
