# Bidirectional Cross-Modal Collaborative Alignment via Semantic-Guided Visual Embeddings for Partially Relevant Video Retrieval

## 1.Introduction
[IEEE TIP 2026]

Official implementation of paper:

Bidirectional Cross-Modal Collaborative Alignment via Semantic-Guided Visual Embeddings for Partially Relevant Video Retrieval

### Core idea

## 1.Requiments
Please install the necessary dependencies listed in requirements.txt.

## 2.Data Preparation
Please download the data from [GMMFormer](https://github.com/huangmozhi9527/GMMFormer) or [DL-DKD](https://github.com/HuiGuanLab/DL-DKD). 

## 3.Training and Inference
### Training and Inference on ActivityNet Captions

```bash
cd src
python main.py -d act --gpu 0

```
### Training and Inference on Charades-STA

```bash
cd src
python main.py -d cha --gpu 0
```

### Training and Inference on TVR

```bash
cd src
python main.py -d tvr --gpu 0
```
You can download the trained model checkpoint from [Baidu Netdisk](https://pan.baidu.com/s/1CAv1dCHn1Pv9LCelChv_hg?pwd=g5w4)

## 4.Results

For this repository, the expected performance is:

| *Dataset* | *R@1* | *R@5* | *R@10* | *R@100* | *SumR* |
|:---|---:|---:|---:|---:|---:|
| TVR | 16.3 | 38.2 | 50.0 | 87.6 | 192.2 |
| ActivityNet Captions | 9.5 | 28.3 | 41.1 | 79.4 | 158.3 |
| Charades-STA | 2.7 | 9.2 | 14.9 | 52.8 | 79.7 |

## 5.Citation

If you find this repository useful, please consider citing our work:

```bibtex
@ARTICLE{11370453,
  author={Li, Huafeng and Zhao, Jialong and Zhang, Yafei and Wen, Jie},
  journal={IEEE Transactions on Image Processing}, 
  title={Bidirectional Cross-Modal Collaborative Alignment via Semantic-Guided Visual Embeddings for Partially Relevant Video Retrieval}, 
  year={2026},
  volume={35},
  pages={1423-1435}}
