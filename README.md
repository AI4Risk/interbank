# Interbank
Interbank Risk Rating: Awesome Datasets and Methods

## Abstract
Accurately assessing and forecasting bank credit ratings at an early stage is vitally important for a healthy financial environment and sustainable economic development. In this repository, we contribute awesome datasets and state-of-the-art methods for the academic research of interbank risk and credit rating. 

Source codes implementation of papers:

- `SA-GNN`: Preventing attacks in interbank credit rating with selective-aware graph neural network, published in [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0675.pdf). 
- `PSAGNN`: Preferential Selective-aware Graph Neural Network for Preventing Attacks in Interbank Credit Rating, Under Review. 

## Usage

### Attacking
To attack the data, run
```
python attack.py --year attack_year --Q attack_quarter --attack_rate attack_rate
```

### Training & Testing
To train the model, run
```
python train.py --year predict_year --Q predict_quarter --attack_rate attack_rate --epochs epoch
```

The attack_quarter is the first two quarters of the predict_quarter.


### Data Description
We collected data spanning 29 quarters, covering a diverse array of banks worldwide, from the first quarter of 2016 to the first quarter of 2023. The included bank categories encompass commercial banks, savings banks, cooperative banks, real estate and mortgage banks, investment banks, Islamic banks, and central banks. In particular, the data set includes Silicon Valley Bank, Signature Bank, First Republic Bank and Credit Suisse Group during the 2023 financial crisis, as well as banks speculated to be closely related to them, so that they can be used to analyze this crisis event in the future.

For each quarter, there is an Edge table and a Feature table. For the Edge table, we used our improved minimum density method to generate Interbank networks (see 3.1 for details). For the Feature table, we collected over 300 features related to various bank finance. After undergoing data cleaning, feature selection, and dimensionality reduction, we finally identified 70 features. For data ratings, we collect existing ratings given by organizations such as Moody's, use machine learning methods to fit existing data to fill in the gaps, and rank the original dozen or so ratings according to relative levels in each quarter's data. It is divided into four categories of ratings.



## Test Result

The results of the performance of credit rating predicting in a test of different quarters on different attack rates are listed as follows:

![performance](/images/performance.png "performance")

The decline in accuracy for credit rating prediction was evident in testing scenarios spanning different years (averaging across four quarters) and
various attack rates are listed as follows:

![decline](/images/decline.png "decline")


## Repo Structure

`models/:` the defense models for SAGNN and PSAGNN;

`data/:` dataset files;

`config/:` configuration files for different models;

`attack_method/:` data attacking;

`methods/:` implementations of models;

`main.py:` organize all models;

`requirements.txt:` package dependencies;


## Requirements  

```
python                       3.7
torch                        1.8.1+cu111
torch-cluster                1.5.9
torch-geometric              2.1.0.post1
torch-scatter                2.0.8
torch-sparse                 0.6.12
powerlaw                     1.3.3
```

## Citing

If you find *Interbank* is useful for your research, please consider citing the following papers:

    @inproceedings{liu2023preventing,
      title={Preventing attacks in interbank credit rating with selective-aware graph neural network},
      author={Liu, Junyi and Cheng, Dawei and Jiang, Changjun},
      booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
      pages={6085--6093},
      year={2023}
    }
    
