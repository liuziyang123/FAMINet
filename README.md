# FAMINet
Implementation of our paper entitiled [FAMINet: Learning Real-time Semi-supervised Video Object Segmentation with Steepest Optimized Optical Flow](https://ieeexplore.ieee.org/abstract/document/9638507) published in TIM. This code is mainly based on [frtm-vos](https://github.com/andr345/frtm-vos). Thanks for their provided codes.
## 1. Get Started
### a. Test environment:
```shell script
Ubuntu 16.04
Python 3.7
Pytorch 1.7
```
### b. Install:
```shell script
sudo apt install ninja-build
pip install scipy scikit-image tqdm opencv-python easydict
```
## 2. Datasets
### a. DAVIS dataset
DAVIS dataset is from the DAVIS benchmark: <https://davischallenge.org/davis2017/code.html>. Users can directly download DAVIS2017 from <https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip>.
### b. Youtube-VOS dataset
Youtube-VOS dataset is from the Youtube-VOS benchmark: <https://youtube-vos.org/dataset/>.
### c. Set path
After downloading the overall datasets, set the corresponding dataset path in `evaluate.py` and `train.py`, e.g.:
```shell script
davis="./DAVIS",  # DAVIS dataset root
yt="/data3/YouTubeVOS",  # YouTubeVOS root
output="./results", # output results
```
## 3. Test
```shell script
python evaluate.py --model model_path --fast --dset ytval # YouTubeVos
python evaluate.py --model model_path --fast --dset dv2016val   # DAVIS 2016
python evaluate.py --model model_path --fast --dset dv2017val   # DAVIS 2017
```
We provided our model FAMINet-2F and FAMINet-3F for reference:
| Name            | Backbone  |  Weights  |
|-----------------|:---------:|:---------:|
| FAMINet-2F, FAMINet-3F  | ResNet18  |[Download](https://drive.google.com/drive/folders/1WhIYaXHx8zhZQ4Nat1_HbM8mQd8z5UZS?usp=sharing) | #0bPx
## 4. Train
```shell script
python train.py name --ftext resnet18 --dset all --dev gpu_id
```
`name` experiment name.

`dset` dataset used for training, e.g., DAVIS, Youtube2018.
