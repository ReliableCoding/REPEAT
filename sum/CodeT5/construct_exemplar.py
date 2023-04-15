import sys
import math
import numpy as np
import torch
import argparse
import random
import jsonlines
import heapq
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset



def calculate_coefficient(new_data, replay_data):

    vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
    transformer = TfidfTransformer()
    train_corpus = [' '.join(i.origin_target+i.origin_source)  for i in new_data+replay_data]
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



def convert_examples_to_features(examples, tokenizer, args, stage=None):
    # collect texts
    codes = []
    target_nl = []
    for example_id, example in enumerate(examples):
        codes.append(example.source)

        if stage == "test":
            target_nl.append("None")
        else:
            target_nl.append(example.target)

    # begin tokenizing
    encoded_codes = tokenizer(
        codes, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    encoded_targets = tokenizer(
        target_nl, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    return {'source_ids':encoded_codes['input_ids'], 'target_ids':encoded_targets['input_ids'],
            'source_mask':encoded_codes['attention_mask'], 'target_mask':encoded_targets['attention_mask']}

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
            new_train_exemplars.append({'code_tokens':train_exemplar[idx].origin_source, 'docstring_tokens':train_exemplar[idx].origin_target, \
                                        'code':train_exemplar[idx].code, 'task_id':str(task_id), 'score':0})
        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)

        # eval
        random.shuffle(eval_exemplar)
        new_eval_exemplars = []
        for idx in range(new_eval_replay_size):
            new_eval_exemplars.append({'code_tokens':eval_exemplar[idx].origin_source, 'docstring_tokens':eval_exemplar[idx].origin_target, \
                                       'code':eval_exemplar[idx].code, 'task_id':str(task_id), 'score':0})
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)
    elif mode == 'ours':
        #train       
        vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
        transformer = TfidfTransformer()
        train_corpus = [' '.join(i.origin_target+i.origin_source) for i in train_exemplar]
        train_tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus)).toarray().tolist()
        cluster_number = args.k
        clf = KMeans(n_clusters=cluster_number, init='k-means++') 
        train_label = clf.fit_predict(train_tfidf)

        train_features = convert_examples_to_features(train_exemplar, tokenizer, args, stage='dev')
        all_source_ids = train_features['source_ids']
        all_source_mask = train_features['source_mask']
        all_target_ids = train_features['target_ids']
        all_target_mask = train_features['target_mask']
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)

        model.eval()
        score = []
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            with torch.no_grad():
                labels = [
                    [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for
                    labels_example in target_ids
                ]
                labels = torch.tensor(labels).to(device)
                tokens_num = torch.tensor([(labels_example != -100).sum().item() for labels_example in labels]).sum().item()
                loss = model(input_ids=source_ids, attention_mask=source_mask, labels=labels).loss
            score.append(loss.div(tokens_num).cpu().item())
        print(len(score),len(train_exemplar))

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

        new_train_exemplars = []
        for idx in train_topk:
            new_train_exemplars.append({'code_tokens':train_exemplar[idx].origin_source, 'docstring_tokens':train_exemplar[idx].origin_target, \
                                        'code':'', 'task_id':str(task_id), 'score':final_score[idx]})
        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)


        #eval
        vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
        transformer = TfidfTransformer()
        eval_corpus = [' '.join(i.origin_target+i.origin_source) for i in eval_exemplar]
        eval_tfidf = transformer.fit_transform(vectorizer.fit_transform(eval_corpus)).toarray().tolist()
        clf = KMeans(n_clusters=cluster_number, init='k-means++') 
        eval_label = clf.fit_predict(eval_tfidf)

        eval_features = convert_examples_to_features(eval_exemplar, tokenizer, args, stage='dev')
        all_source_ids = eval_features['source_ids']
        all_source_mask = eval_features['source_mask']
        all_target_ids = eval_features['target_ids']
        all_target_mask = eval_features['target_mask']
        eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        model.eval()
        score = []
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            with torch.no_grad():
                labels = [
                    [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for
                    labels_example in target_ids
                ]
                labels = torch.tensor(labels).to(device)
                tokens_num = torch.tensor([(labels_example != -100).sum().item() for labels_example in labels]).sum().item()
                loss = model(input_ids=source_ids, attention_mask=source_mask, labels=labels).loss
            score.append(loss.div(tokens_num).cpu().item())
        print(len(score),len(eval_exemplar))

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

        new_eval_exemplars = []
        for idx in eval_topk:
            new_eval_exemplars.append({'code_tokens':eval_exemplar[idx].origin_source, 'docstring_tokens':eval_exemplar[idx].origin_target, \
                                        'code':'', 'task_id':str(task_id), 'score':final_score[idx]})
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)

    with jsonlines.open(train_replay_path, mode='w') as f:
        f.write_all(train_exemplars)
    with jsonlines.open(eval_replay_path, mode='w') as f:
        f.write_all(eval_exemplars)
    