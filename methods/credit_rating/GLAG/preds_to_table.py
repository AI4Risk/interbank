
import os
import numpy as np
import pandas as pd
from methods.credit_rating.GLAG.utils.tools import init_time_list
from methods.credit_rating.GLAG.utils.metrics import FAcc,RiskCaptureRecall_k
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


time_list = init_time_list()
folder = "preds"
excel_path = f"results/result.xlsx"

with pd.ExcelWriter(excel_path) as writer:
    for seed in range(1):
        acc_list=[]
        f1_list=[]
        facc_list=[]
        recall_list =[]
        qwk_list=[]
        index_list=[]
        for t in range(len(time_list)):
            y,q=time_list[t].split('-')
            y=int(y)
            q=int(q)      
            if (y, q) in [(2016, 1), (2016, 2)]:
                continue
      
            filename = os.path.join(folder, f"{y}Q{q}.npz")
            quarter_name= f'{y}Q{q}'
            index_list.append(quarter_name)
            results = np.load(filename)
            logits = results['logits']
            preds=results['preds']
            labels = results['labels']
           
            acc = accuracy_score(labels,preds)
            f1 = f1_score(labels,preds,average='macro')
            facc = FAcc(labels,preds)
            recall=RiskCaptureRecall_k(logits,labels)
            qwk = cohen_kappa_score(labels,preds,weights='quadratic')

            acc_list.append(acc)
            f1_list.append(f1)
            facc_list.append(facc)
            recall_list.append(recall)
            qwk_list.append(qwk)

        result_table = {
            'acc':acc_list,
            'f1':f1_list,
            'facc':facc_list,
            'recall':recall_list,
            'qwk':qwk_list
        }
        result = pd.DataFrame(result_table,index=index_list)
        result.to_excel(writer, sheet_name=f'{seed+1}')
        print(f"seed {seed+1} done")
