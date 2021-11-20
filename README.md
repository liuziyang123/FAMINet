# FAMINet
Implementation of our paper entitiled ``FAMINet: Learning Real-time Semi-supervised Video Object Segmentation with Steepest Optimized Optical Flow'' submitted to TIM, under review. This code is mainly based on [frtm-vos](https://github.com/andr345/frtm-vos). Thanks for their provided codes.
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
After downloading the overall datasets, set the corresponding dataset path in `evaluate.py`, e.g.:
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
| FAMINet-2F  | ResNet18  |[download, 0bPx](https://pan.baidu.com/s/1v-rXfuwTNJOl7NiMye8pXA) | #0bPx
| FAMINet-3F  | ResNet18  |[download, 0bPx](https://pan.baidu.com/s/1v-rXfuwTNJOl7NiMye8pXA) |
