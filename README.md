# PSAGNN
Implementation for the paper "Preferential Selective-aware Graph Neural Network for Preventing Attacks in Interbank Credit Rating"

## Abstract
Accurately assessing and forecasting bank credit ratings at an early stage is vitally important for a healthy financial environment and sustainable economic development. However, the evaluation process faces challenges due to individual attacks on the rating model. Some participants may provide manipulated information in an attempt to undermine the rating model and secure higher scores, further complicating the evaluation process. Therefore, we propose a novel approach called the Preferential Selective-aware Graph Neural Network model (PSAGNN) to simultaneously defend against feature and structural non-target poisoning attacks on Interbank credit ratings. In particular, the model establishes a phased optimization approach combined with biased perturbation, and explores the Interbank preferences and scale-free nature of networks, to adaptively prioritize the poisoning training data and simulate a clean graph. Finally, we apply a weighted penalty on the opposition function to optimize the model so that the model can distinguish between attackers. Extensive experiments on our newly collected Interbank quarter dataset and case studies demonstrate the superior performance of our proposed approach in preventing credit rating attacks compared to state-of-the-art baselines.

## Dataset
We collected data spanning 29 quarters, covering a diverse array of banks worldwide, from the first quarter of 2016 to the first quarter of 2023. The included bank categories encompass commercial banks, savings banks, cooperative banks, real estate and mortgage banks, investment banks, Islamic banks, and central banks. In particular, the data set includes Silicon Valley Bank, Signature Bank, First Republic Bank and Credit Suisse Group during the 2023 financial crisis, as well as banks speculated to be closely related to them, so that they can be used to analyze this crisis event in the future.

For each quarter, there is an Edge table and a Feature table. For the Edge table, we used our improved minimum density method to generate Interbank networks (see 3.1 for details). For the Feature table, we collected over 300 features related to various bank finance. After undergoing data cleaning, feature selection, and dimensionality reduction, we finally identified 70 features. For data ratings, we collect existing ratings given by organizations such as Moody's, use machine learning methods to fit existing data to fill in the gaps, and rank the original dozen or so ratings according to relative levels in each quarter's data. It is divided into four categories of ratings.


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

## Usage
To attack the data, run
```
python attack.py --year attack_year --Q attack_quarter --attack_rate attack_rate
```

To train the model, run
```
python train.py --year predict_year --Q predict_quarter --attack_rate attack_rate --epochs epoch
```

The attack_quarter is the first two quarters of the predict_quarter.
