import csv
import pandas as pd
import json
from collections import OrderedDict, Counter
import os
import argparse


def read_file(base_dir, task_id):
    f = csv.reader(open(os.path.join(base_dir, 'test_'+task_id+'.gold'),'r'))
    ref = {}
    count = 0
    for i in f:
        ref[count] = [i[0].split('\t')[1].strip()]
        count+=1
    f = csv.reader(open(os.path.join(base_dir, 'test_'+task_id+'.output'),'r'))
    hypo = {}
    count = 0
    for i in f:
        hypo[count] = [i[0].split('\t')[1].strip()]
        if len(hypo[count][0].split()) == 0:
            hypo[count] = ['Get']
        count+=1
    return hypo,ref

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default=None, type=str, required=True, help="mode")
parser.add_argument("--lang", default=None, type=str, required=True, help="lang")
args = parser.parse_args()

acc_sum=0
p_sum=0
r_sum=0
f1_sum=0
acc_list=[[] for _ in range(5)]
f1_list=[[] for _ in range(5)]
for i in range(5):
    for j in range(i+1):
        hypo,ref = read_file('model/'+args.mode+'/task_'+str(i)+'/', str(j))
        tp,tn,fp,fn=0,0,0,0
        for k in hypo:
            prediction=hypo[k][0]
            refrence=ref[k][0]
            if prediction=='true' and refrence =='true':
                tp+=1
            elif prediction=='true' and refrence =='false':
                fn+=1
            elif prediction=='false' and refrence =='false':
                tn+=1
            elif prediction=='false' and refrence =='true':
                fp+=1
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = 2*p*r/(p+r)
        acc = (tp+tn)/(tp+tn+fp+fn)
        #print('('+str(round(100*f1,2))+','+str(round(100*acc,2))+')', end='  ')
        print(round(f1*100,2), end='  ')
        f1_sum+=f1/(i+1)
        acc_sum+=acc/(i+1)
        p_sum+=p/(i+1)
        r_sum+=r/(i+1)
    print('')
        
print('Avg_F1:',round(100*f1_sum/5,2))
print('Avg_P:',round(100*p_sum/5,2))
print('Avg_R:',round(100*r_sum/5,2))
print('Avg_ACC:',round(100*acc_sum/5,2))