import csv
import pandas as pd
import json
from collections import OrderedDict, Counter
import os
import argparse

from evall.bleu import corpus_bleu
from evall.rouge import Rouge
from evall.meteor import Meteor

def eval_accuracies(hypotheses, references, sources=None,
                    filename=None, print_copy_info=False, mode='dev'):
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(references, hypotheses)

    return bleu * 100, rouge_l * 100, meteor * 100

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

bleu_sum=0
meteor_sum=0
rouge_sum=0
first_bleu=0
first_meteor=0
first_rouge=0
for i in range(5):
    for j in range(i+1):
        hypo,ref = read_file('model/'+args.lang+'/'+args.mode+'/task_'+str(i)+'/', str(j))
        bleu,rouge,meteor = eval_accuracies(hypo,ref)
        #print('Model:',i,'  Task:',j)
        #print('bleu:',bleu,'  meteor:',meteor,'  rouge:',rouge)
        print(round(bleu,2), end='  ')
        if j==0:
            first_bleu+=bleu
            first_meteor+=meteor
            first_rouge+=rouge
        bleu_sum+=bleu/(i+1)
        meteor_sum+=meteor/(i+1)
        rouge_sum+=rouge/(i+1)
    print('')
        
#print('first_bleu:',first_bleu/5,'  first_meteor:',first_meteor/5,'  first_rouge:',first_rouge/5)
#print('avg_bleu:',bleu_sum/5,'  avg_meteor:',meteor_sum/5,'  avg_rouge:',rouge_sum/5)
print('Avg_BLEU:',round(bleu_sum/5,2))
print('Avg_METEOR:',round(meteor_sum/5,2))
print('Avg_ROUGE:',round(rouge_sum/5,2))