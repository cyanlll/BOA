# Bidirectional Cross-Modal Collaborative Alignment via Semantic-Guided Visual Embeddings for Partially Relevant Video Retrieval

# Requiments
Please install the necessary dependencies listed in requirements.txt.

# Data Preparation
Please download the data from [GMMFormer]([https://www.openai.com](https://github.com/huangmozhi9527/GMMFormer)) or [DL-DKD]([https://www.openai.com](https://github.com/HuiGuanLab/DL-DKD)). 

## Training on ActivityNet Captions

```bash
cd src
python main.py -d act --gpu 0

```
## Training on Charades-STA

```bash
cd src
python main.py -d cha --gpu 0
```
