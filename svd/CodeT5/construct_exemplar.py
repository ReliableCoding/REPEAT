import sys
import math
import numpy as np
import torch
import argparse
import random
import jsonlines
import heapq
import pandas as pd
from tqdm import tqdm, trange
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset



def calculate_coefficient(new_data, replay_data):

    vectorizer = CountVectorizer(min_df=15, ngram_range=(1,1))
    transformer = TfidfTransformer()
    train_corpus = ['label_'+str(i.origin_target)+' '+i.origin_source for i in new_data+replay_data]
    train_tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus)).toarray().tolist()
    train_tfidf = np.array(train_tfidf)
    
    replay_feature = train_tfidf[:len(new_data)].mean(axis=0)
    new_feature = train_tfidf[len(new_data):].mean(axis=0)

    a = replay_feature.dot(new_feature)
    b = np.linalg.norm(replay_feature)
    c = np.linalg.norm(new_feature)

    return a/(b*c)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       



def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features

def construct_exemplars(model, train_replay_path, eval_replay_path, train_exemplar, eval_exemplar, train_replay_size, eval_replay_size, mode, task_id):
    if task_id>0:
        old_train_replay_size = train_replay_size//(task_id+1)
        old_eval_replay_size = eval_replay_size//(task_id+1)
        new_train_replay_size = train_replay_size-task_id*old_train_replay_size
        new_eval_replay_size = eval_replay_size-task_id*old_eval_replay_size
    else:
        old_train_replay_size = 0
        old_eval_replay_size = 0
        new_train_replay_size = train_replay_size
        new_eval_replay_size = eval_replay_size

    old_replay_train_data = {}
    old_replay_valid_data = {}
    train_exemplars = []
    eval_exemplars = []
    if task_id>0:
        with jsonlines.open(train_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_train_data:
                    old_replay_train_data[obj['task_id']]=[obj]
                else:
                    old_replay_train_data[obj['task_id']].append(obj)
        with jsonlines.open(eval_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_valid_data:
                    old_replay_valid_data[obj['task_id']]=[obj]
                else:
                    old_replay_valid_data[obj['task_id']].append(obj)
                    
                    
    if mode == 'random':
        random.shuffle(train_exemplar)
        new_train_exemplars = []
        for idx in range(new_train_replay_size):
            new_train_exemplars.append({'func_before':train_exemplar[idx].origin_source, 'vul':train_exemplar[idx].origin_target ,'task_id':str(task_id), 'score':0})
        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)

        random.shuffle(eval_exemplar)
        new_eval_exemplars = []
        for idx in range(new_eval_replay_size):
            new_eval_exemplars.append({'func_before':eval_exemplar[idx].origin_source, 'vul':eval_exemplar[idx].origin_target, 'task_id':str(task_id), 'score':0})
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)

    with jsonlines.open(train_replay_path, mode='w') as f:
        f.write_all(train_exemplars)
    with jsonlines.open(eval_replay_path, mode='w') as f:
        f.write_all(eval_exemplars)


def construct_exemplars_ours(model, args, train_exemplar, eval_exemplar, tokenizer, device, mode):
    task_id = args.task_id
    train_replay_size = args.train_replay_size
    train_replay_path = args.train_examplar_path
    eval_replay_size = args.eval_replay_size
    eval_replay_path = args.eval_examplar_path

    if task_id>0:
        old_train_replay_size = train_replay_size//(task_id+1)
        old_eval_replay_size = eval_replay_size//(task_id+1)
        new_train_replay_size = train_replay_size-task_id*old_train_replay_size
        new_eval_replay_size = eval_replay_size-task_id*old_eval_replay_size
    else:
        old_train_replay_size = 0
        old_eval_replay_size = 0
        new_train_replay_size = train_replay_size
        new_eval_replay_size = eval_replay_size

    old_replay_train_data = {}
    old_replay_valid_data = {}
    train_exemplars = []
    eval_exemplars = []
    if task_id>0:
        with jsonlines.open(train_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_train_data:
                    old_replay_train_data[obj['task_id']]=[obj]
                else:
                    old_replay_train_data[obj['task_id']].append(obj)
        for key in old_replay_train_data:
            old_replay_train_data[key].sort(key = lambda x:x["score"], reverse=True)
        with jsonlines.open(eval_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_valid_data:
                    old_replay_valid_data[obj['task_id']]=[obj]
                else:
                    old_replay_valid_data[obj['task_id']].append(obj)
        for key in old_replay_valid_data:
            old_replay_valid_data[key].sort(key = lambda x:x["score"], reverse=True)
    
    if mode == 'random':
        # train
        random.shuffle(train_exemplar)
        new_train_exemplars = []
        for idx in range(new_train_replay_size):
            new_train_exemplars.append({'func_before':train_exemplar[idx].origin_source, 'vul':train_exemplar[idx].origin_target, \
                                        'task_id':str(task_id), 'score':0})
        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)


        # eval
        random.shuffle(eval_exemplar)
        new_eval_exemplars = []
        for idx in range(new_eval_replay_size):
            new_eval_exemplars.append({'func_before':eval_exemplar[idx].origin_source, 'vul':eval_exemplar[idx].origin_target, \
                                       'task_id':str(task_id), 'score':0})
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)
    elif mode == 'ours':
        #train
        train_exemplar_class = {}
        eval_exemplar_class = {}
        train_sum=len(train_exemplar)
        eval_sum=len(eval_exemplar)
        new_train_replay_size_copy = new_train_replay_size
        new_eval_replay_size_copy = new_eval_replay_size
        new_eval_exemplars = []
        new_train_exemplars = []
        for i in train_exemplar:
            if i.origin_target not in train_exemplar_class:
                train_exemplar_class[i.origin_target] = [i]
            else:
                train_exemplar_class[i.origin_target].append(i)
        for i in eval_exemplar:
            if i.origin_target not in eval_exemplar_class:
                eval_exemplar_class[i.origin_target] = [i]
            else:
                eval_exemplar_class[i.origin_target].append(i)
        #print(train_exemplar_class.keys())
        #print(eval_exemplar_class.keys())

        for clas in train_exemplar_class:
            #train
            train_exemplar = train_exemplar_class[clas]
            new_train_replay_size = math.ceil(new_train_replay_size_copy*len(train_exemplar)//train_sum)

            vectorizer = CountVectorizer(min_df=15, ngram_range=(1,1))
            transformer = TfidfTransformer()
            train_corpus = [i.origin_source for i in train_exemplar]
            train_tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus)).toarray().tolist()
            cluster_number = args.k
            clf = KMeans(n_clusters=cluster_number, init='k-means++') 
            train_label = clf.fit_predict(train_tfidf)
    
            
            train_features = convert_examples_to_features(train_exemplar, tokenizer, args, stage='dev')
            all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)  
            all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)  
            train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)

            model.eval()
            score = []
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
    
                with torch.no_grad():
                    labels = [
                        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for
                        labels_example in target_ids
                    ]
                    labels = torch.tensor(labels).to(device)
                    loss = model(input_ids=source_ids, attention_mask=source_mask, labels=labels).loss
                score.append(loss.cpu().item())
            score_max = max(score)
            score = [(score_max-i) for i in score]
            score_sum = sum(score)
            score = [i/score_sum for i in score]
            cluster_replay_number = []
            for i in range(cluster_number):
                cluster_replay_number.append(math.ceil(train_label.tolist().count(i)*new_train_replay_size//len(train_label)))
            class_score = {}
            class_pos = {}
            for idx in range(len(train_label)):
                i = train_label[idx]
                j = score[idx]
                if i not in class_score:
                    class_score[i] = [j]
                    class_pos[i] = [idx]
                else:
                    class_score[i].append(j)
                    class_pos[i].append(idx)
            train_topk = []
            for i in range(cluster_number):
                class_topk = heapq.nlargest(min(args.mu*cluster_replay_number[i],len(class_score[i])), range(len(class_score[i])), class_score[i].__getitem__)
                class_topk = np.random.choice(class_topk, cluster_replay_number[i], replace=False)
                train_topk.extend([class_pos[i][j] for j in class_topk])
            final_score = score
    
            for idx in train_topk:
                new_train_exemplars.append({'func_before':train_exemplar[idx].origin_source, 'vul':train_exemplar[idx].origin_target, \
                                            'task_id':str(task_id), 'score':final_score[idx]})
            
            #eval
            eval_exemplar = eval_exemplar_class[clas]
            new_eval_replay_size = math.ceil(new_eval_replay_size_copy*len(eval_exemplar)//eval_sum)

            vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
            transformer = TfidfTransformer()
            eval_corpus = [i.origin_source for i in eval_exemplar]
            eval_tfidf = transformer.fit_transform(vectorizer.fit_transform(eval_corpus)).toarray().tolist()
            clf = KMeans(n_clusters=cluster_number, init='k-means++') 
            eval_label = clf.fit_predict(eval_tfidf)
    
            eval_features = convert_examples_to_features(eval_exemplar, tokenizer, args, stage='dev')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
    
            model.eval()
            score = []
            bar = tqdm(eval_dataloader,total=len(eval_dataloader))
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
    
                with torch.no_grad():
                    labels = [
                        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for
                        labels_example in target_ids
                    ]
                    labels = torch.tensor(labels).to(device)
                    loss = model(input_ids=source_ids, attention_mask=source_mask, labels=labels).loss
                score.append(loss.cpu().item())
            score_max = max(score)
            score = [(score_max-i) for i in score]
            score_sum = sum(score)
            score = [i/score_sum for i in score]
            cluster_replay_number = []
            for i in range(cluster_number):
                cluster_replay_number.append(math.ceil(eval_label.tolist().count(i)*new_eval_replay_size//len(eval_label)))
            class_score = {}
            class_pos = {}
            for idx in range(len(eval_label)):
                i = eval_label[idx]
                j = score[idx]
                if i not in class_score:
                    class_score[i] = [j]
                    class_pos[i] = [idx]
                else:
                    class_score[i].append(j)
                    class_pos[i].append(idx)
            eval_topk = []
            for i in range(cluster_number):
                class_topk = heapq.nlargest(min(args.mu*cluster_replay_number[i],len(class_score[i])), range(len(class_score[i])), class_score[i].__getitem__)
                class_topk = np.random.choice(class_topk, cluster_replay_number[i], replace=False)
                eval_topk.extend([class_pos[i][j] for j in class_topk])
            final_score = score
    
            for idx in eval_topk:
                new_eval_exemplars.append({'func_before':eval_exemplar[idx].origin_source, 'vul':eval_exemplar[idx].origin_target, \
                                           'task_id':str(task_id), 'score':final_score[idx]})
                                    
        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)

    with jsonlines.open(train_replay_path, mode='w') as f:
        f.write_all(train_exemplars)
    with jsonlines.open(eval_replay_path, mode='w') as f:
        f.write_all(eval_exemplars)
    