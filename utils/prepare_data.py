############################################ IMPORT ##########################################################
import pickle
import sys, os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

############################################ LOAD W2V EMBEDDING ###############################################
def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r',encoding='utf-8')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend(emotion.lower().split() + clause.lower().split())

    words = set(words) # Collection of all unique words
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) # Each word and its position
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) # Each word and its position

    w2v = {}
    inputFile2 = open(embedding_path, 'r',encoding='utf-8')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) 
            # Randomly take from the uniform distribution [-0.1, 0.1]
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))
    # add a noisy embedding in the end for out of vocabulary words
    embedding.extend([list(np.random.rand(embedding_dim) / 5. - 0.1)])

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) \
        for i in range(200)])
    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos

############################################ LOAD DATA PAIR STEP ##############################################
def load_data_pair(input_file, word_idx, max_doc_len = 75, max_sen_len = 45,max_path_num=1000,max_path_len=5,max_rel_len=4):
    adj = []
    doc_id_list = []
    adj_path = []

    with open('../dataset.matrix','rb') as f:
        adj_raw = pickle.load(f)
    print('load data_file: {}'.format(input_file))
    with open('.../path.matrix','rb') as f:
        path_softmax_matrix = pickle.load(f)
    print('load data_file: {}'.format('path'))

    pf_json_data=[]
    pf_json_file='.../concept.pickle'
    print("loading paths from %s" % pf_json_file)
    with open(pf_json_file, 'rb') as handle:
        pf_json_data = pickle.load(handle)
    print('\t Done! ')



    pair_id_all, y_position, y_cause, y_pair, x, sen_len, doc_len, distance = [], [], [], [], [], [], [], []
    graph,paths,rels=[],[],[]
    n_cut = 0
    inputFile = open(input_file, 'r',encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        ######################################## doc_len_condition ########################################
        if d_len >= max_doc_len :
          for i in range(d_len+1) :
            line = inputFile.readline().strip().split(',')
          continue
        ######################################## doc_len_condition ########################################

        pairs = eval('[' + inputFile.readline().strip() + ']')
        pos_list, cause_list = zip(*pairs)
        pairs = [[pos_list[i], cause_list[i]] for i in range(len(pos_list))]
        pair_id_all.extend([doc_id*10000+p[0]*100+p[1] for p in pairs])
        y_position_tmp, y_cause_tmp, y_pair_tmp, sen_len_tmp, x_tmp, distance_tmp = \
        np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros((max_doc_len * max_doc_len, 2)), \
        np.zeros((max_doc_len, )), np.zeros((max_doc_len, max_sen_len)), np.zeros((max_doc_len * max_doc_len, ))


        paths_tmp=np.zeros((max_doc_len * max_doc_len,max_path_num,max_path_len ))
        rels_tmp=np.zeros((max_doc_len * max_doc_len,max_path_num,max_rel_len))
        for qas in pf_json_data[doc_id-1]['clauses']:
            source_clasue=qas['source_clasue']
            target_clause=qas['target_clause']
            pf_res = qas["pf_res"]
            flag=0
            pair_id=0
            if pf_res is not None:
                for item in pf_res:
                    p = item["path"]
                    p = [n + 1 for n in p]
                    p.extend([0] * (max_path_len - len(p)))  # padding
                    r = item["rel"]
                    for i_ in range(len(r)):
                        for j_ in range(len(r[i_])):
                            if r[i_][j_] - 17 in r[i_]:
                                r[i_][j_] -= 17  # to delete realtedto* and antonym*
                    r = [n[0] + 1 for n in r]  # only pick the top relation when multiple ones are okay
                    r.extend([0] * (max_rel_len - len(r)))  # padding
                    

                    if flag==(doc_id*100000+source_clasue*100+target_clause): 
                        paths_tmp[(source_clasue-1)*max_doc_len+(target_clause-1)][pair_id]=p
                        rels_tmp[(source_clasue-1)*max_doc_len+(target_clause-1)][pair_id]=r
                        pair_id=pair_id+1

                    else:
                        flag=(doc_id*100000+source_clasue*100+target_clause)
                        paths_tmp[(source_clasue-1)*max_doc_len+(target_clause-1)][pair_id]=p
                        rels_tmp[(source_clasue-1)*max_doc_len+(target_clause-1)][pair_id]=r
                        pair_id=0
        for i in range(d_len):
            line = inputFile.readline().strip().split(',')
            words = line[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                word = word.lower()
                if j >= max_sen_len:
                    n_cut += 1
                    break
                elif word not in word_idx : x_tmp[i][j] = 24166
                else : x_tmp[i][j] = int(word_idx[word])

        for i in range(d_len):
            for j in range(d_len):
                # Check whether i is an emotion clause
                if i+1 in pos_list :
                    y_position_tmp[i][0] = 0; y_position_tmp[i][1] = 1
                else :
                    y_position_tmp[i][0] = 1; y_position_tmp[i][1] = 0
                # Check whether j is a cause clause
                if j+1 in cause_list :
                    y_cause_tmp[j][0] = 0; y_cause_tmp[j][1] = 1
                else :
                    y_cause_tmp[j][0] = 1; y_cause_tmp[j][1] = 0
                # Check whether i, j clauses are emotion cause pairs
                pair_id_curr = doc_id*10000+(i+1)*100+(j+1)
                if pair_id_curr in pair_id_all :
                    y_pair_tmp[i*max_doc_len+j][0] = 0; y_pair_tmp[i*max_doc_len+j][1] = 1
                else :
                    y_pair_tmp[i*max_doc_len+j][0] = 1; y_pair_tmp[i*max_doc_len+j][1] = 0
                # Find the distance between the clauses, and use the same embedding beyond 10 clauses
                distance_tmp[i*max_doc_len+j] = min(max(j-i+100, 90), 110)
        adj_original = np.array(adj_raw[doc_id - 1],dtype='float32')
        adj_temp = np.pad(adj_original,[0,max_doc_len - adj_original.shape[0]],mode='constant',constant_values=0)
        adj_original_path =np.array(path_softmax_matrix[doc_id - 1],dtype='float32')
        adj_temp_path = np.pad(adj_original_path,[0,max_doc_len - adj_original_path.shape[0]],mode='constant',constant_values=0)
        y_position.append(y_position_tmp)
        y_cause.append(y_cause_tmp)
        y_pair.append(y_pair_tmp)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
        doc_len.append(d_len)
        distance.append(distance_tmp)
        adj.append(adj_temp)
        adj_path.append(adj_temp_path)
        doc_id_list.append(doc_id)
        paths.append(paths_tmp)
        rels.append(rels_tmp)

    y_position, y_cause, y_pair, x, sen_len, doc_len, distance, adj,paths,rels,adj_path = map(torch.tensor, \
    [y_position, y_cause, y_pair, x, sen_len, doc_len, distance, adj,paths,rels,adj_path])

    for var in ['y_position', 'y_cause', 'y_pair', 'x', 'sen_len', 'doc_len', 'distance','paths','rels']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return y_position, y_cause, y_pair, x, sen_len, doc_len, distance, adj, doc_id_list,paths,rels,adj_path

