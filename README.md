# Interbank
Interbank Risk and Credit Rating: Datasets and Methods

## Overview
Accurately assessing and forecasting bank credit ratings at an early stage is vitally important for a healthy financial environment and sustainable economic development. 
In this repository, we contribute awesome datasets and state-of-the-art methods for the academic research of interbank risk and credit rating. 
We will introduce the dataset, the credit rating methods, and the credit rating defense methods.

Source codes implementation of papers:

- Credit Rating Method:
  - `TGAR`: Transformed Graph Attention for Credit Rating, published in [ICIEA 2023](https://ieeexplore.ieee.org/document/10241546).
  - `SA-GNN`: Preventing Attacks in Interbank Credit Rating with Selective-aware Graph Neural Network, published in [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0675.pdf). 
  - `PSAGNN`: Preferential Selective-aware Graph Neural Network for Preventing Attacks in Interbank Credit Rating, published in [TNNLS 2024a](https://ieeexplore.ieee.org/document/10815626). 
  - `HFTCRNet`: HFTCRNet: Hierarchical Fusion Transformer for Interbank Credit Rating and Risk Assessment, published in [TNNLS 2024b](https://ieeexplore.ieee.org/abstract/document/10729282).
  - `DPTAP`: Dual Pairwise Pre-training and Prompt-tuning with Aligned Prototypes for Interbank Credit Rating, accepted by [WWW 2025]().

## Contents

- [Repo Structure](#repo-structure)
- [Dataset](#dataset)
- [Methods](#Methods)
  - [Contagion Chain Generation](#contagion-chain-generation)
  - [Data Attack](#data-attack)
  - [Train](#train)
- [Results](#results)
- [Requirements](#requirements) 
- [License](#license) 
- [Citation](#citation) 

## Repo Structure

`datasets/:` dataset files;

`methods/:` the implementation of interbank credit rating methods;

`models/:` the trained offline models of interbank credit rating;

`results/:` detailed quarterly results and figures in the paper.

## Dataset

We collected data spanning 32 quarters, covering a diverse array of banks worldwide, from the first quarter of 2016 to the fourth quarter of 2023. The included bank categories encompass commercial banks, savings banks, cooperative banks, real estate and mortgage banks, investment banks, Islamic banks, and central banks. In particular, the data set includes Silicon Valley Bank, Signature Bank, First Republic Bank and Credit Suisse Group during the 2023 financial crisis, as well as banks speculated to be closely related to them, so that they can be used to analyze this crisis event in the future.

For each quarter, there is an Edge table, a Feature table. 

+ For the Edge table, we used our improved minimum density method to generate Interbank networks. 
+ For the Feature table, we collected over 300 features related to various bank finance. After undergoing data cleaning, feature selection, dimensionality reduction, and normalization, we finally identified 70 features. 

  + For data ratings, we collect existing ratings given by organizations such as Moody's, use machine learning methods to fit existing data to fill in the gaps, and rank the original dozen or so ratings according to relative levels in each quarter's data. The last three columns are credit ratings, SRISK ratio, and SRISK value, respectively. Credit ratings represent the predicted rating label for the next quarter, and it is divided into four categories. For HFTCRNet, the original ratings of banks are transfered to relative ranks with ``A``,``B``,``C``,and ``D`` to facilitate further analysis on the systemic risk. 
  + For the SRISK value and ratio, we collect them from the [vLab](https://vlab.stern.nyu.edu).

## Methods

### Contagion Chain Generation

To generate the risk contagion chain from the features for the HFTCRNet, run

```
cd methods/credit_rating/HFTCRNet
python generate_contagionlist.py
```

### Data attack

To **attack** the data, run 

```
python methods/credit_rating_under_attack/attack_methods/attack.py --year [attack_year] --Q [attack_quarter] --attack_rate [attack_rate]
```

The attack_quarter is the first two quarters of the predict_quarter.

### Train

To train the **TGAR** model, run

```
python methods/credit_rating/HFTCRNet/train.py --method [TGAR] --year [predict_year] --quarter [predict_quarter] --epochs [epoch]
```

To train the **HFTCRNet** model, please first configure the ``methods/credit_rating/HFTCRNet/run.sh`` and then run the ``methods/credit_rating/HFTCNet/run.sh`` with ``bash``.

* Configurate the ``methods/credit_rating/HFTCRNet/run.sh``

  ```
  start_year=2018
  start_quarter=1
  end_year=2023
  end_quarter=1
  time_steps=7
  
  CUDA_VISIBLE_DEVICES=0 python train.py ...
  ```

* Run the ``methods/credit_rating/HFTCRNet/run.sh`` with ``bash``

  ```
  cd methods/credit_rating/HFTCRNet
  bash run.sh
  ```

To train the **DPTAP** model, you can run the quarterly experiment by using the following instruction:

``````
python methods/credit_rating/DPTAP/main.py --device=0 --gnn_type='GT' --input_dim=70 --hid_dim=256 --exp_obj=None --update_epochs=300 --tune_epochs=500 --shot_num=100 --lr=0.001 --decay=5e-4 --num_layer=2
``````

* This hyper-parameter setting can reproduce the experimental results in the paper, and the quarterly experimental results will be saved at `methods/credit_rating/DPTAP/results/C1/main/`. For each quarter, the GNN model weights pre-trained from the last quarter are saved at `methods/credit_rating/DPTAP/pretrained_gnn/`.

* To run experiments for the studies of shot number, rating category imbalance, and node connectivity imbalance, you can change the `--exp_obj` setting, and configure the corresponding ratio, respectively. For example: 

  ```
  python methods/credit_rating/DPTAP/main.py --shot_num=200 --exp_obj='shot_num'
  ```

  ```
  python methods/credit_rating/DPTAP/main.py --class_ratios="40 30 20 10" --exp_obj='class_ratios'
  ```

  ```
  python methods/credit_rating/DPTAP/main.py --dense_sparse_ratios="10 90" --exp_obj='dense_sparse_ratios'
  ```



To train the **SAGNN** or **PSAGNN** model, run

```
python methods/credit_rating_under_attack/methods/train.py --method [SAGNN]/[PSAGNN] --year [predict_year] --Q [predict_quarter] --attack_rate [attack_rate] --epochs [epoch]
```

The attack_quarter is the first two quarters of the predict_quarter.


### Results

The results of different credit rating methods with regards to different years are listed as follows:

**Accuracy**

| Method   | 2023   | 2022   | 2021   | 2020   | 2019   | 2018   |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ |
| GCN      | 0.4721 | 0.6384 | 0.6408 | 0.6211 | 0.5709 | 0.4405 |
| GAT      | 0.4556 | 0.6887 | 0.6891 | 0.6601 | 0.6082 | 0.4987 |
| TGAR     | 0.6197 | 0.7536 | 0.7539 | 0.7112 | 0.6511 | 0.5810 |
| HFTCRNet | 0.5090 | 0.7130 | 0.6935 | 0.6600 | 0.6405 | 0.6563 |
| DPTAP    | 0.7401 | 0.7234 | 0.6745 | 0.6189 | 0.5539 | 0.5320 |
| SAGNN    | 0.3830 | 0.4385 | 0.4734 | 0.4531 | 0.4295 | 0.3570 |
| PSAGNN   | 0.4048 | 0.4868 | 0.5147 | 0.5603 | 0.5907 | 0.5483 |

**Macro** $F_1$

| Method   | 2023   | 2022   | 2021   | 2020   | 2019   | 2018   |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ |
| GCN      | 0.3857 | 0.5299 | 0.4910 | 0.4871 | 0.4497 | 0.3175 |
| GAT      | 0.4736 | 0.5896 | 0.5371 | 0.5729 | 0.5111 | 0.3444 |
| TGAR     | 0.5865 | 0.6553 | 0.5958 | 0.6206 | 0.5974 | 0.4947 |
| HFTCRNet | 0.5150 | 0.6688 | 0.6193 | 0.6373 | 0.6145 | 0.5930 |
| DPTAP    | -      | -      | -      | -      | -      | -      |
| SAGNN    | 0.2043 | 0.2056 | 0.2570 | 0.2563 | 0.3396 | 0.2949 |
| PSAGNN   | 0.3148 | 0.3679 | 0.3986 | 0.4564 | 0.4893 | 0.4797 |

Note: For the quarterly credit rating results and the credit rating results under attack, please refer to ```results/further_results.md```


## Requirements

```
python                       3.7.16
torch                        1.13.1
torch-cluster                1.6.1
pyg                          2.3.0
torch-scatter                2.1.1
torch-sparse                 0.6.17
tqdm                         4.42.1
scikit-learn                 1.0.2
pandas                       1.2.3
numpy                        1.21.5
powerlaw                     1.3.4
```

To install all these packages, please first install miniconda and then try this command:

```
conda install tqdm scikit-learn pandas numpy pyg pytorch-scatter pytorch-sparse pytorch-cluster pytorch-cluster pytorch-spline-conv powerlaw=1.3.4 pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -c pyg
pip install powerlaw==1.3.4
```

## Contributors :
<a href="https://github.com/AI4Risk/interbank/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Risk/interbank" />
</a>



## License

The use of the source code of interbank complies with the GNU GENERAL PUBLIC LICENSE.
Please contact us if you find any potential violations. 

## Citation

If you find *Interbank* is useful for your research, please consider citing the following papers:

```
@inproceedings{tang2025Dual,
author = {Tang, Jiehao and Wang, Wenjun and Cheng, Dawei and Zhao, Hui and Jiang, Changjun},
title = {Dual Pairwise Pre-training and Prompt-tuning with Aligned Prototypes for Interbank Credit Rating},
year = {2025},
booktitle = {Proceedings of the ACM Web Conference 2025},
}

@article{liu2024preferential,
  title={Preferential Selective-Aware Graph Neural Network for Preventing Attacks in Interbank Credit Rating},
  author={Liu, Junyi and Cheng, Dawei and Jiang, Changjun},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}

@article{li2024hftcrnet,
  title={HFTCRNet: Hierarchical Fusion Transformer for Interbank Credit Rating and Risk Assessment},
  author={Li, Jiangtong and Zhou, Ziyuan and Zhang, Jingkai and Cheng, Dawei and Jiang, Changjun},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}

@inproceedings{liu2023preventing,
    title={Preventing attacks in interbank credit rating with selective-aware graph neural network},
    author={Liu, Junyi and Cheng, Dawei and Jiang, Changjun},
    booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
    pages={6085--6093},
    year={2023}
}

@inproceedings{liu2023transformed,
    title={Transformed Graph Attention for Credit Rating},
    author={Liu, Charles Z. and Xiang, Sheng and Cheng, Dawei and Liu, Junyi and Zhang, Ying and Qin, Lu},
    booktitle={2023 IEEE 18th Conference on Industrial Electronics and Applications (ICIEA)},
    pages={1011-1016},
    year={2023}
}
```

## 
