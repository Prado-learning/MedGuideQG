from rouge import Rouge
import json
import jieba
import sys
from text2vec import Similarity
import numpy as np
sys.setrecursionlimit(100000)


dataset_path = './testdata/MyData.json'
model_name = 'OpenBioLLM-70B-20240918'
sim_model = Similarity()
print(model_name)
Results = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        adata = json.loads(line)
        Results.append(adata)
decoded_preds = []
decoded_labels = []
simi = 0
for adata in Results:
    decoded_preds.append(adata[model_name])
    decoded_labels.append(adata['tgt'])
    simi += sim_model.get_score(adata['tgt'], adata[model_name])

# decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
# decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]
decoded_preds = [" ".join(pred.replace(" ", "")) for pred in decoded_preds]
decoded_labels = [" ".join(label.replace(" ", "")) for label in decoded_labels]
rouge = Rouge()
scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
# 打印结果
print('Rouge-1: ', scores['rouge-1']['f'])
print('Rouge-2: ', scores['rouge-2']['f'])
print('Rouge-L: ', scores['rouge-l']['f'])
print('Similarity: ', simi / len(Results))
