import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split 
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2019,
        choices=[2016,2017,2018,2019,2020,2021,2022], help='dataset')
parser.add_argument('--Q', type=int, default=1,
        choices=[1,2,3,4], help='dataset')
parser.add_argument('--attack_rate', type=float, default=0.25, help="noise ptb_rate")
args = parser.parse_args()

def edge_attack(year,Q,attack_rate):
    df = pd.read_csv("datasets/rank_pre_Q/"+str(year)+"Q"+str(Q)+".csv")
    ed = pd.read_csv("datasets/edge_Q/edge_"+str(year)+"Q"+str(Q)+".csv")
    df1=df[df['rank_pre']==1]
    df2=df[df['rank_pre']==2]
    df3=df[df['rank_pre']==3]
    df4=df[df['rank_pre']==4]

    rate=2*attack_rate
    df1_train, df1_test = train_test_split(df1, test_size= rate, random_state=1234)
    df2_train, df2_test = train_test_split(df2, test_size= rate, random_state=1234)
    df2_train1, df2_test1 = train_test_split(df2, test_size= rate, random_state=1234)
    df3_train, df3_test = train_test_split(df3, test_size= rate, random_state=1234)
    df3_train1, df3_test1 = train_test_split(df3, test_size= rate, random_state=1234)
    df4_train, df4_test = train_test_split(df4, test_size= rate, random_state=1234)

    a=df1_test['index']
    c=df3_test['index']
    if len(a)>len(c):
        a=df1_test['index'][:len(c)]
        c=df3_test['index']
    else:
        a=df1_test['index']
        c=df3_test['index'][:len(a)]

    d=df4_test['index']
    b=df2_test['index']
    if len(d)>len(b):
        d=df4_test['index'][:len(b)]
        b=df2_test['index']
    else:
        d=df4_test['index']
        b=df2_test['index'][:len(d)]

    a=a.to_list()
    b=b.to_list()
    c=c.to_list()
    d=d.to_list()

    s=ed['Sourceid']
    t=ed['Targetid']
    s=s.to_list()
    t=t.to_list()

    s1=s+a+b
    t1=t+c+d

    dict1 = {"Sourceid":s1,"Targetid":t1}
    edge = pd.DataFrame(dict1)
    save_path_dir="datasets/edge_Q"+str(attack_rate)+"/"
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    edge.to_csv("datasets/edge_Q"+str(attack_rate)+"/edge_"+str(year)+"Q"+str(Q)+".csv",index=False)


def feature_attack(year,Q,attack_rate):
    
    all=[]
    for k in range(1,70):
        allQ=[]
        for i in range(2016,2023):
            for j in range(1,5):
                df = pd.read_csv("datasets/rank_pre_Q/"+str(i)+"Q"+str(j)+".csv")
                df1=df[df['rank_pre']==1]
                df2=df[df['rank_pre']==2]
                df3=df[df['rank_pre']==3]
                df4=df[df['rank_pre']==4]
                allQ.append([sum(df1.iloc[:,k])/len(df1),sum(df2.iloc[:,k])/len(df2),sum(df3.iloc[:,k])/len(df3),sum(df4.iloc[:,k])/len(df4)])
        all.append(allQ)     

    c=['1','2','3','4']
    y=['2016Q1','2016Q2','2016Q3','2016Q4','2017Q1','2017Q2','2017Q3','2017Q4','2018Q1','2018Q2','2018Q3','2018Q4','2019Q1','2019Q2','2019Q3','2019Q4',
    '2020Q1','2020Q2','2020Q3','2020Q4','2021Q1','2021Q2','2021Q3','2021Q4','2022Q1','2022Q2','2022Q3','2022Q4']
    for k in range(1,70):
        all[k-1]=pd.DataFrame(all[k-1],columns=c,index=y)      

    df = pd.read_csv("datasets/rank_pre_Q/"+str(year)+"Q"+str(Q)+".csv")
    df1=df[df['rank_pre']==1]
    df2=df[df['rank_pre']==2]
    df3=df[df['rank_pre']==3]
    df4=df[df['rank_pre']==4]

    rate=2*attack_rate
    df2_train, df2_test = train_test_split(df2, test_size= rate, random_state=1234)
    df3_train, df3_test = train_test_split(df3, test_size= rate, random_state=1234)
    df4_train, df4_test = train_test_split(df4, test_size= rate, random_state=1234)
    for k in range(1,70):
        df3_test.iloc[:,k]=df3_test.iloc[:,k]*sum(all[k-1]['1'])/sum(all[k-1]['3'])
        df4_test.iloc[:,k]=df4_test.iloc[:,k]*sum(all[k-1]['2'])/sum(all[k-1]['4'])

    content = pd.concat([df1,df2_train, df2_test, df3_train, df3_test, df4_train, df4_test], axis=0, join='inner')
    content=content.sort_values(by=['index'])
    content['rank_pre']=content['rank_pre'].astype(int)
    save_path_dir="datasets/rank_pre_Q"+str(attack_rate)+"/"
    if not os.path.exists(save_path_dir):
         os.makedirs(save_path_dir)
    content.to_csv("datasets/rank_pre_Q"+str(attack_rate)+"/"+str(year)+"Q"+str(Q)+".csv",index=False)


# for i in range(2019,2024):
#     for j in range(1,4):
#         for kk in [0.05,0.1,0.15,0.2,0.25]:
#             feature_attack(i,j,kk)
#             edge_attack(i,j,kk)

edge_attack(args.year,args.Q,args.attack_rate)
feature_attack(args.year,args.Q,args.attack_rate)
