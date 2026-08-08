# Bidirectional Cross-Modal Collaborative Alignment via Semantic-Guided Visual Embeddings for Partially Relevant Video Retrieval

## 1.Introduction
[IEEE TIP 2026]

Official implementation of paper:

Bidirectional Cross-Modal Collaborative Alignment via Semantic-Guided Visual Embeddings for Partially Relevant Video Retrieval

### Core idea
!(./assets/Idea.png)

Core idea for addressing the challenges in PRVR. Semantic–visual association library serves not only as a bridge to mitigate modality gaps but also
provides the foundation for generating dynamic visual anchors to address partial mismatches between textual and video content.

### Overview
!(./assets/Framework.png)

Overview of the proposed architecture. We cluster keywords from the textual query set to construct a SVAL. Relevant visual features are selected
based on their similarity to query features and used to update the library. The SGFP encodes visual features with the help of dynamically retrieved semantic
anchors. In the VITL, semantic-level visual features are concatenated with textual features for joint encoding, reducing the vision-language modality gap.

## 2.Requiments
Please install the necessary dependencies listed in requirements.txt.

## 3.Data Preparation
Please download the data from [GMMFormer](https://github.com/huangmozhi9527/GMMFormer) or [DL-DKD](https://github.com/HuiGuanLab/DL-DKD). 

## 4.Training and Inference
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

## 5.Results

For this repository, the expected performance is:

| *Dataset* | *R@1* | *R@5* | *R@10* | *R@100* | *SumR* |
|:---|---:|---:|---:|---:|---:|
| TVR | 16.3 | 38.2 | 50.0 | 87.6 | 192.2 |
| ActivityNet Captions | 9.5 | 28.3 | 41.1 | 79.4 | 158.3 |
| Charades-STA | 2.7 | 9.2 | 14.9 | 52.8 | 79.7 |

## 6.Citation

If you find this repository useful, please consider citing our work:

```bibtex
@ARTICLE{11370453,
  author={Li, Huafeng and Zhao, Jialong and Zhang, Yafei and Wen, Jie},
  journal={IEEE Transactions on Image Processing}, 
  title={Bidirectional Cross-Modal Collaborative Alignment via Semantic-Guided Visual Embeddings for Partially Relevant Video Retrieval}, 
  year={2026},
  volume={35},
  pages={1423-1435}}
