# PLGMNet: Parallel Local-Global Mamba Network for Real-Time Steel Surface Defect Detection

This repository provides the official implementation of the PRCV accepted paper **"PLGMNet: Parallel Local-Global Mamba Network for Real-Time Steel Surface Defect Detection"**.

PLGMNet is a lightweight real-time segmentation model for steel surface defect detection. The model introduces a Parallel Local-Global Information Mamba (PLGIM) layer to capture fine-grained local defect patterns and global contextual information in parallel.


## Quick Start

Clone this repository:

```bash
git clone https://github.com/ic1fy/PLGMNet.git
cd PLGMNet
```

Train and evaluate the model:

```bash
python train.py
python test.py
```

Run the web demo:

```bash
python app.py
```

Open `http://localhost:5000` in your browser.


## Dataset Organization

Prepare the training and testing image-mask pairs under `data/`:

```bash
data/
|-- Train/
|   |-- Image/
|   `-- Mask/
|-- Test/
|   |-- Image/
|   `-- Mask/
|-- train.txt
`-- test.txt
```


## Citation

    @inproceedings{zhang_plgmnet_2026,
          title = {PLGMNet: Parallel Local-Global Mamba Network for Real-Time Steel Surface Defect Detection},
          booktitle = {Pattern Recognition and Computer Vision},
          publisher = {Springer Nature Singapore},
          author = {Zhang, Ju and Li, Chenlei and Zhou, Xiaofei and Wu, Yong and Liu, Deyang and Zhang, Jiyong and Liu, Zhi},
          year = {2026},
          pages = {355--369},
    }
