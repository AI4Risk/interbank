# Interbank
Interbank Risk and Credit Rating: Datasets and Methods

## Overview
Accurately assessing and forecasting bank credit ratings at an early stage is vitally important for a healthy financial environment and sustainable economic development. 
In this repository, we contribute awesome datasets and state-of-the-art methods for the academic research of interbank risk and credit rating. 
We will introduce the dataset, the credit rating method, and the credit rating defense method.

Source codes implementation of papers:

- Credit Rating Method:
  - `TGAR`: Transformed Graph Attention for Credit Rating, published in [ICIEA 2023](https://ieeexplore.ieee.org/document/10241546). 
- Credit Rating Defense Method
  - `SA-GNN`: Preventing Attacks in Interbank Credit Rating with Selective-aware Graph Neural Network, published in [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0675.pdf). 
  - `PSAGNN`: Preferential Selective-aware Graph Neural Network for Preventing Attacks in Interbank Credit Rating, Under Review. 

## Contents

- [Dataset](#dataset)
- [Methods](#Methods)
  - [Credit Rating Method](#credit-rating-method)
  - [Credit Rating Defense Method](#credit-rating-defense-method)

- [Repo Structure](#repo-structure)
- [Requirements](#requirements) 
- [License](#license) 
- [Citation](#citation) 

## Repo Structure

`methods/:` the implementation of interbank credit rating methods;

`models/:` the trained offline models of interbank credit rating;

`datasets/:` dataset files;

`images/:` the image resource used in this repository;

## Dataset

We collected data spanning 29 quarters, covering a diverse array of banks worldwide, from the first quarter of 2016 to the first quarter of 2023. The included bank categories encompass commercial banks, savings banks, cooperative banks, real estate and mortgage banks, investment banks, Islamic banks, and central banks. In particular, the data set includes Silicon Valley Bank, Signature Bank, First Republic Bank and Credit Suisse Group during the 2023 financial crisis, as well as banks speculated to be closely related to them, so that they can be used to analyze this crisis event in the future.

For each quarter, there is an Edge table and a Feature table. 

+ For the Edge table, we used our improved minimum density method to generate Interbank networks. 

+ For the Feature table, we collected over 300 features related to various bank finance. After undergoing data cleaning, feature selection, dimensionality reduction, and normalization, we finally identified 70 features. 

  + For data ratings, we collect existing ratings given by organizations such as Moody's, use machine learning methods to fit existing data to fill in the gaps, and rank the original dozen or so ratings according to relative levels in each quarter's data. 
  
  + It is divided into four categories of ratings, where the last column of each feature table represents the predicted credit rating for the next quarter.

## Methods

### Credit Rating Method

#### Usage

To **train** the TGAR model, run
```
python train.py --method [TGRA] --year [predict_year] --quarter [predict_quarter] --epochs [epoch]
```

#### Results

The results of different credit rating method with regards to different quarters are listed as follows:

##### Accuracy

| Method | 2023Q1 | 2022Q4 | 2022Q3 | 2022Q2 | 2022Q1 | 2021Q4 | 2021Q3 | 2021Q2 | 2021Q1 | 2020Q4 | 2020Q3 | 2020Q2 | 2020Q1 | 2019Q4 | 2019Q3 | 2019Q2 | 2019Q1 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| GCN    | 0.4478 | 0.4265 | 0.5576 | 0.5356 | 0.5549 | 0.5624 | 0.5485 | 0.4938 | 0.5395 | 0.5507 | 0.3836 | 0.5747 | 0.4967 | 0.5301 | 0.5626 | 0.6347 | 0.6273 |
| GAT    | 0.5061 | 0.4487 | 0.5828 | 0.5639 | 0.5861 | 0.5789 | 0.5820 | 0.5444 | 0.5785 | 0.5952 | 0.4804 | 0.5266 | 0.4997 | 0.5466 | 0.5850 | 0.6646 | 0.6471 |
| TGAR   | 0.5294 | 0.4788 | 0.6257 | 0.6257 | 0.6218 | 0.6198 | 0.5987 | 0.5701 | 0.6290 | 0.6198 | 0.6079 | 0.6117 | 0.5136 | 0.5686 | 0.6053 | 0.6952 | 0.6699 |

##### Macro $F_1$

| Method | 2023Q1 | 2022Q4 | 2022Q3 | 2022Q2 | 2022Q1 | 2021Q4 | 2021Q3 | 2021Q2 | 2021Q1 | 2020Q4 | 2020Q3 | 2020Q2 | 2020Q1 | 2019Q4 | 2019Q3 | 2019Q2 | 2019Q1 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| GCN    | 0.4478 | 0.3623 | 0.4121 | 0.3708 | 0.3739 | 0.3954 | 0.4046 | 0.3786 | 0.3876 | 0.4095 | 0.5340 | 0.5089 | 0.4169 | 0.4278 | 0.4124 | 0.44282 | 0.4374 |
| GAT    | 0.5092 | 0.3845 | 0.4567 | 0.4225 | 0.4509 | 0.4348 | 0.4473 | 0.4349 | 0.4410 | 0.4742 | 0.5732 | 0.5864 | 0.4219 | 0.4554 | 0.4759 | 0.4969 | 0.5236 |
| TGAR   | 0.5384 | 0.4034 | 0.5224 | 0.5126 | 0.5069 | 0.4868 | 0.4834 | 0.4700 | 0.4857 | 0.5082 | 0.5077 | 0.5651 | 0.4623 | 0.4778 | 0.5287 | 0.5861 | 0.4898 |

### Credit Rating Under Attack

#### Usage

To **attack** the data, run
```
python code_defense/attack_methods/attack.py --year [attack_year] --Q [attack_quarter] --attack_rate [attack_rate]
```

To **train** the PSAGNN model, run
```
python code_defense/methods/train.py --method [PSAGNN] --year [predict_year] --Q [predict_quarter] --attack_rate [attack_rate] --epochs [epoch]
```

The attack_quarter is the first two quarters of the predict_quarter.

#### Result

The results of the performance of credit rating predicting in a test of different quarters on different attack rates are listed as follows:

![performance](/images/performance.png "performance")

The decline in accuracy for credit rating prediction was evident in testing scenarios spanning different years (averaging across four quarters) and
various attack rates are listed as follows:

![decline](/images/decline.png "decline")




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