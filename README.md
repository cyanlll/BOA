# Bidirectional Cross-Modal Collaborative Alignment via Semantic-Guided Visual Embeddings for Partially Relevant Video Retrieval

# Requiments
Please install the necessary dependencies listed in requirements.txt.

# Data Preparation
Please download the data from [GMMFormer](https://github.com/huangmozhi9527/GMMFormer) or [DL-DKD](https://github.com/HuiGuanLab/DL-DKD). 

#Training and Inference
## Training and Inference on ActivityNet Captions

```bash
cd src
python main.py -d act --gpu 0

```
## Training and Inference on Charades-STA

```bash
cd src
python main.py -d cha --gpu 0
```

## Training and Inference on TVR

```bash
cd src
python main.py -d tvr --gpu 0
```
You can download the trained model checkpoint from [Baidu Netdisk](https://pan.baidu.com/s/1CAv1dCHn1Pv9LCelChv_hg?pwd=g5w4)
