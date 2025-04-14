# The Art of Camouflage: Few-shot Learning for Animal Detection and Segmentation

This repository is the official implementation of the paper entitled: **The Art of Camouflage: Few-shot Learning for Animal Detection and Segmentation**, IEEE Access, 2024.

**Authors**: Thanh-Danh Nguyen, Anh-Khoa Nguyen Vu, Nhat-Duy Nguyen, Vinh-Tiep Nguyen, Thanh Duc Ngo, Thanh-Toan Do, Minh-Triet Tran, Tam V. Nguyen*.

[[Paper]](https://ieeexplore.ieee.org/document/10608133) [[ArXiv]](https://arxiv.org/abs/2304.07444) [[Code]](https://github.com/danhntd/FS-CDIS) [[Project Page]](https://danhntd.github.io/projects.html#CAMO-FS)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/few-shot-camouflaged-animal-detection-and/few-shot-instance-segmentation-on-camo-fs)](https://paperswithcode.com/sota/few-shot-instance-segmentation-on-camo-fs?p=few-shot-camouflaged-animal-detection-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/few-shot-camouflaged-animal-detection-and/few-shot-object-detection-on-camo-fs)](https://paperswithcode.com/sota/few-shot-object-detection-on-camo-fs?p=few-shot-camouflaged-animal-detection-and)

---
## Updates
[2024/7] We have released the checkpoints, visualization, and initial instructions for FS-CDIS⚡!

## 1. Environment Setup
Download and install Anaconda with the recommended version from [Anaconda Homepage](https://www.anaconda.com/download): [Anaconda3-2019.03-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh) 
 
```
git clone https://github.com/danhntd/FS-CDIS.git
cd FS-CDIS
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

After completing the installation, please create and initiate the workspace with the specific versions below. The experiments were conducted on a Linux server with a single `GeForce RTX 2080Ti GPU`, CUDA 10.1/10.2, Torch 1.7.

```
conda create --name FSCDIS python=3
conda activate FSCDIS
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
```

This source code is based on [Detectron2](https://github.com/facebookresearch/detectron2). Please refer to INSTALL.md for the pre-built or building Detectron2 from source.

After setting up the dependencies, use the command `pip install -e .` in this root to finish.

## 2. Data Preparation


### Download the datasets

The proposed CAMO-FS is available at this [link](https://www.kaggle.com/datasets/danhnt/camo-fs-dataset).

### Register datasets
Detectron2 requires a step of data registration for those who want to use the external datasets ([Detectron2 Docs](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)), which is already prepared in this repository.





## 3. Training Pipeline
<!-- Our proposed FS-CDIS framework:
<img align="center" src="/visualization/framework.png"> -->

Our detailed proposals of instance triplet loss and instance memory storage:
<img align="center" src="/visualization/framework_fs-cdis-memo-redesign-ieee-access.png">




### Configurations

All configs can be found in the `./configs/` directory.

<!-- Initial parameters:
```

```

### Training

```

```
-->
<!-- ### Pre-defined variables
```
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1

cfg_MODEL='
MODEL.ROI_HEADS.NUM_CLASSES 16
SOLVER.MAX_ITER 2000
'

MODEL_NAME='novel1_1shot'
OUTPUT_DIR=checkpoints/camo_mtfa_default/camo_model_${MODEL_NAME}_mask_rcnn_R_101_FPN_mtfa
config=configs/CAMO-shot_mtfa_default/mask_rcnn_R_101_FPN_ft_fsdet_cos_${MODEL_NAME}.yaml
WEIGHT=weights/mrcnn_r101_fpn_80cls.pkl
```

### Testing

```
python tools/run_train.py --num-gpus ${NGPUS} \
			   --dist-url auto \
			   --resume \
			   --config-file ${config} \
			   --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL} SOLVER.STEPS "(40000, 54000)"

```  -->

The whole script commands can be found in `./scripts/*`.


### Released checkpoints and results:

We provide the checkpoints of our final model:

<!-- | Model R-101 |   FS-CDIS-ITL   |   FS-CDIS-IMS    |
| ----------- |:---------------:|:----------------:|
|    1-shot   |[link](https://) | [link](https://) |
|    2-shot   |[link](https://) | [link](https://) |
|    3-shot   |[link](https://) | [link](https://) |
|    5-shot   |[link](https://) | [link](https://) | -->


| Model R-101 |   FS-CDIS-ITL    | mask AP  | box AP  |   FS-CDIS-IMS    | mask AP  | box AP  |
|:-----------:|:----------------:|:---:|:---:|:----------------:|:---:|:---:|
|   1-shot    | [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/Eef2Z-cEJkBOj-iOn3Cj_1IBYES8HEznelUznSkBR0qJNw?e=M9KC0a) |4.46 |4.04 | [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/EUTsy7lkCU5PlqO8jIhLNsUBbAg-lstj6LygKPGff82CmA?e=H55kwj) |5.46 |4.50 |
|   2-shot    | [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/EUUPqoMv3CROhw3_X0171sQBos1ro9nGmsnWReMQEavoTA?e=EngAXx) |5.57 |7.28 | [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/EdvySeWhDzZHss_AB9OiDGwBQAwu5576wIHb7fyBcCW2aA?e=D8w4ml) |6.95 |6.95 |
|   3-shot    | [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/EeMmWhLwAyZMn0mA0QUPC2UB2ELHzRrAwmToiCKC0bIdmw?e=jqDjTJ) |6.41 |7.49 | [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/ETNHaMzUjTdAkHPgOYTe8uEBZ9qsmKeXGcmXG6789R-oBA?e=NgUcEA) |7.36 |7.55 |
|   5-shot    | [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/Ecs66hnQn5dBkgQVdjXUkMgBTM-MppZjLnhcScmO1uF4Pw?e=4T5uws) |8.48 |9.76 | [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/EXfk8mPDjSREp4Q5LRb2Aw4B8lSlAfuHS3ym1cB-uRmbrw?e=U0Sb6x) |9.61 |10.36 |



## 4. Visualization

<p align="center">
  <img width="800" src="/visualization/visualization.png">
</p>

## Citation
Please use the following bibtex to cite this repository:
```
@article{nguyen2024art,
  title={The Art of Camouflage: Few-shot Learning for Animal Detection and Segmentation},
  author={Nguyen, Thanh-Danh and Vu, Anh-Khoa Nguyen and Nguyen, Nhat-Duy and Nguyen, Vinh-Tiep and Ngo, Thanh Duc and Do, Thanh-Toan and Tran, Minh-Triet and Nguyen, Tam V},
  journal={IEEE Access},
  volume={-},
  pages={-},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgements

[iMTFA](https://github.com/danganea/iMTFA) [Detectron2](https://github.com/facebookresearch/detectron2.git) 