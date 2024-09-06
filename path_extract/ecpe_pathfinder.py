import configparser
import networkx as nx
import itertools
import math
import random
import json
from tqdm import tqdm
import sys
import time
import timeit
import numpy as np


config = configparser.ConfigParser()
config.read(".../pathfinder/paths.cfg")


cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None




def load_resources():
    global concept2id, relation2id, id2relation, id2concept
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")
def load_cpnet():
    global cpnet,concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def get_edge(src_concept, tgt_concept):
    global cpnet, concept2id, relation2id, id2relation, id2concept
    rel_list = cpnet[src_concept][tgt_concept]
    return list(set([rel_list[item]["rel"] for item in rel_list]))

# source and target is text
def find_paths(source, target, ifprint = False):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    s = concept2id[source]
    t = concept2id[target]

    if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
        return
    # paths =
    all_path = []

    all_path_set = set()

    for max_len in range(1, 5):
        
        for p in nx.all_simple_paths(cpnet_simple, source=s, target=t, cutoff=max_len):
            path_str = "-".join([str(c) for c in p])
            if path_str not in all_path_set:
                all_path_set.add(path_str)
                all_path.append(p)
            if len(all_path) >= 100:  # top shortest 300 paths
                break
        if len(all_path) >= 100:  # top shortest 300 paths
            break


    all_path.sort(key=len, reverse=False)
    pf_res = []
    for p in all_path:
        # print([id2concept[i] for i in p])
        rl = []
        for src in range(len(p) - 1):
            src_concept = p[src]
            tgt_concept = p[src + 1]


            rel_list = get_edge(src_concept, tgt_concept)
            rl.append(rel_list)
            if ifprint:
                rel_list_str = []
                for rel in rel_list:
                    if rel < len(id2relation):
                        rel_list_str.append(id2relation[rel])
                    else:
                        rel_list_str.append(id2relation[rel - len(id2relation)]+"*")
                print(id2concept[src_concept], "----[%s]---> " %("/".join(rel_list_str)), end="")
                if src + 1 == len(p) - 1:
                    print(id2concept[tgt_concept], end="")
        if ifprint:
            print()

        pf_res.append({"path": p, "rel": rl})
    return pf_res


def process(filename):
    output_path = filename  + ".pf"
    import os
    if os.path.exists(output_path):
        print(output_path + " exists. Skip!")
        return

    load_resources()
    load_cpnet()

    with open(filename, 'r') as fp:
        lines = json.load(fp)

    document=[]
    for line in tqdm(lines, desc="loading file"):
        doc_one={}
        doc_one['doc_id']=line['doc_id'] 
        doc_one['doc_len']=line['doc_len'] 
        doc_one['clauses'] =[]
        for clause_i in line['clauses']:

            qc=clause_i['clause_concept_recognition']
            for c in qc:
                for check_clause_i in line['clauses']:
                      for j in check_clause_i['clause_concept_recognition']:
                            if c!=j and c!=''and j!='':
                                pf_res = find_paths(c, j)
                                print(c,'->',j)
                                doc_one['clauses'].append({'source_clasue':clause_i['clause_id'],'target_clause':check_clause_i['clause_id'],"source":c, "target":j, "pf_res":pf_res})
        document.append(doc_one)

    with open(output_path, 'w') as fi:
        json.dump(document, fi)
 
if __name__ == "__main__":
    #process(sys.argv[1], int(sys.argv[2]))
    process('.../all_data_pair_emotion-Lexicon-and-down_add_concept.json.mcp')

    

