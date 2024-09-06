############################################ IMPORT ##########################################################
import sys, os
import numpy as np
import torch
#import dgl.function as fn
from torch import nn
from torch import optim
from torch.nn import functional as F
from utils.funcs import *
from utils.prepare_data import *
from models.gcn import GraphConvolution
from models.gnn_layer import GraphAttentionLayer
from models.bi_gat import BipartiteGraphAttentionLayer
#from models.bi_gat_distance2 import BipartiteGraphAttentionLayer_dis2
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################ FLAGS ############################################################
train_file_path = '.../data_combine_eng/clause_keywords.csv'          # clause keyword file
w2v_file = '.../data_combine_eng/w2v_200.txt'                         # embedding file
embedding_dim = 200                                                 # dimension of word embedding
embedding_dim_pos = 50                                              # dimension of position embedding
max_sen_len = 30                                                    # max number of tokens per sentence
max_doc_len = 41                                                    # max number of tokens per document
n_hidden = 100                                                      # number of hidden unit
n_class = 2                                                         # number of distinct class                                            # number of train epochs
training_epochs = 25    
batch_size = 32                                                         # number of example per batch
learning_rate = 0.0050                                              # learning rate
keep_prob1 = 0.8                                                    # word embedding training dropout keep prob
keep_prob2 = 1.0                                                    # softmax layer dropout keep prob
keep_prob3 = 1.0                                                    # softmax layer dropout keep prob
l2_reg = 0.00010                                                    # l2 regularization
cause = 1.0                                                         # lambda1
pos = 1.0                                                           # lambda2
pair = 2.5                                                          # lambda3
diminish_factor = 0.400                                             # give less weight to -ve examples


max_path_num=10
max_path_len = 5
max_rel_len = 4
concept_dim=100
graph_hidden_dim = 50
graph_output_dim=25

############################################ MODEL ############################################################


class RSTIAG(nn.Module):
    def __init__(self, embedding_dim, embedding_dim_pos, sen_len, doc_len, keep_prob1, keep_prob2, \
                keep_prob3, n_hidden, n_class,concept_dim,graph_hidden_dim,graph_output_dim,max_path_num,max_path_len,max_rel_len):
        super(RSTIAG, self).__init__()
        self.embedding_dim = embedding_dim; self.embedding_dim_pos = embedding_dim_pos 
        self.sen_len = sen_len; self.doc_len = doc_len
        self.keep_prob1 = keep_prob1; self.keep_prob2 = keep_prob2
        self.n_hidden = n_hidden; self.n_class = n_class
        self.concept_dim=concept_dim
        self.graph_hidden_dim=graph_hidden_dim
        self.graph_output_dim = graph_output_dim
        self.max_path_num=max_path_num
        self.max_path_len=max_path_len
        self.max_rel_len=max_rel_len

        self.dropout1 = nn.Dropout(p = 1 - keep_prob1)
        self.dropout2 = nn.Dropout(p = 1 - keep_prob2)
        self.dropout3 = nn.Dropout(p = 1 - keep_prob3)
        self.relu = nn.ReLU()
        self.pos_linear = nn.Linear(4*n_hidden, n_class)
        self.cause_linear = nn.Linear(4*n_hidden, n_class)
        self.pair_linear1 = nn.Linear(8*n_hidden + embedding_dim_pos+self.concept_dim+self.concept_dim, n_hidden//2)
        self.pair_linear2 = nn.Linear(n_hidden//2, n_class)
        self.word_bilstm = nn.LSTM(embedding_dim, n_hidden, batch_first = True, bidirectional = True)
        self.pos_bilstm = nn.LSTM(2*n_hidden, n_hidden, batch_first = True, bidirectional = True)
        self.cause_bilstm = nn.LSTM(2*n_hidden + n_class, n_hidden, batch_first = True, bidirectional = True)
        self.attention = Attention(n_hidden, sen_len)

        self.attention_path_len = Attention_path_len(concept_dim//2,max_path_len,max_path_num)
        self.attention_rel_len = Attention_path_len(concept_dim//2, max_rel_len,max_path_num)
        self.attention_path_num = Attention(concept_dim//2, max_path_num)
        self.attention_rel_num  = Attention(concept_dim//2, max_path_num)

        self.gat1 = GraphAttentionLayer(4,2 * n_hidden,2 * n_hidden//4,0.1)
        self.gat2 = GraphAttentionLayer(4,2 * n_hidden,2 * n_hidden//4,0.1)
        self.gat3 = GraphAttentionLayer(4,2 * n_hidden,2 * n_hidden//4,0.1)
        self.gat4 = GraphAttentionLayer(4,2 * n_hidden,2 * n_hidden//4,0.1)
        self.Bi_gat1 = BipartiteGraphAttentionLayer(4,2 * n_hidden,2 * n_hidden//4,0.1)
        self.Bi_gat2 = BipartiteGraphAttentionLayer(4,2 * n_hidden,2 * n_hidden//4,0.1)
    def get_clause_embedding(self, x):
        '''
        input shape: [batch_size, doc_len, sen_len, embedding_dim]
        output shape: [batch_size, doc_len, 2 * n_hidden]
        '''
        x = x.reshape(-1, self.sen_len, self.embedding_dim)
        x = self.dropout1(x)
        # x is of shape (batch_size * max_doc_len, max_sen_len, embedding_dim)
        x, hidden_states = self.word_bilstm(x.float())
        # x is of shape (batch_size * max_doc_len, max_sen_len, 2 * n_hidden)
        s = self.attention(x).reshape(-1, self.doc_len, 2 * self.n_hidden)
        # s is of shape (batch_size, max_doc_len, 2 * n_hidden)
        return s
    

    def get_path_rel_embedding(self, paths,rels):
        '''
        input shape: [batch_size, doc_len, sen_len, embedding_dim]
        output shape: [batch_size, doc_len, 2 * n_hidden]
        '''
        paths = paths.reshape(-1, self.max_path_num,self.max_path_len, self.concept_dim)
        rels = rels.reshape(-1, self.max_path_num,self.max_rel_len, self.concept_dim)
        paths =self.attention_path_len(paths).reshape(-1, self.max_path_num, self.concept_dim)
        rels = self.attention_rel_len(rels).reshape(-1, self.max_path_num, self.concept_dim)
      
        paths =self.attention_path_num(paths).reshape(-1, self.doc_len*self.doc_len, self.concept_dim)
        rels = self.attention_rel_num(rels).reshape(-1, self.doc_len*self.doc_len, self.concept_dim)
        return paths,rels

    def get_cause_prediction(self, x, adj,adj_path):
        '''
        input shape: [batch_size, doc_len, 2 * n_hidden + n_class]
        output(s) shape: [batch_size, doc_len, 2 * n_hidden], [batch_size, doc_len, n_class]
        '''
        x_context, hidden_states = self.cause_bilstm(x.float())
        x_context_gc = self.gat1(x_context,adj)
        x_context_gc_path = self.gat3(x_context,adj_path)
        x_context_co=self.Bi_gat2(x_context_gc,x_context_gc_path)

        x_context_gc_cat = torch.cat([x_context,x_context_co], dim=2)
        x = x_context_gc_cat.reshape(-1, 4 * self.n_hidden)
        x = self.dropout2(x)
        # x is of shape (batch_size * max_doc_len, 2 * n_hidden)
        pred_cause = F.softmax(self.cause_linear(x), dim = -1)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        pred_cause = pred_cause.reshape(-1, self.doc_len, self.n_class)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        return x_context_gc_cat, pred_cause

    def get_emotion_prediction(self, x, adj,adj_path):
        '''
        input shape: [batch_size, doc_len, 2 * n_hidden]
        output(s) shape: [batch_size, doc_len, 2 * n_hidden], [batch_size, doc_len, n_class]
        '''
        x_context, hidden_states = self.pos_bilstm(x.float())
        x_context_gc = self.gat2(x_context,adj)
        x_context_gc_path = self.gat4(x_context,adj_path)
        x_context_co=self.Bi_gat1(x_context_gc,x_context_gc_path)
        x_context_gc_cat = torch.cat([x_context,x_context_co], dim=2)
        x = x_context_gc_cat.reshape(-1, 4* self.n_hidden)
        x = self.dropout2(x)
        # x is of shape (batch_size * max_doc_len, 2 * n_hidden)
        pred_pos = F.softmax(self.pos_linear(x), dim = -1)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        pred_pos = pred_pos.reshape(-1, self.doc_len, self.n_class)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        return x_context_gc_cat, pred_pos

    def get_pair_prediction(self, x1, x2, distance,paths_a,rels_a, adj):
        '''
        input(s) shape: [batch_size * doc_len, 2 * n_hidden], [batch_size * doc_len, 2 * n_hidden], 
                        [batch_size, doc_len * doc_len, embedding_dim_pos] 
        output shape: [batch_size, doc_len * doc_len, n_class]
        '''        
        x = create_pairs(x1, x2)
        x_distance = torch.cat([x, distance.float(),paths_a.float(),rels_a.float()], -1)
        # x_distance is of shape (batch_size, max_doc_len * max_doc_len, 4 * n_hidden + embedding_dim_pos)
        x_distance = x_distance.reshape(-1, 8 * self.n_hidden + self.embedding_dim_pos+self.concept_dim+self.concept_dim)
        x_distance = self.dropout3(x_distance)
        # x is of shape (batch_size * max_doc_len * max_doc_len, 4 * n_hidden + embedding_dim_pos)
        pred_pair = F.softmax(self.pair_linear2(self.relu(self.pair_linear1(x_distance))), dim = -1)
        # pred_pair is of shape (batch_size * max_doc_len * max_doc_len, n_class)
        pred_pair = pred_pair.reshape(-1, self.doc_len * self.doc_len, self.n_class)
        # pred_pair is of shape (batch_size, max_doc_len * max_doc_len, n_class)
        return pred_pair

    def forward(self, x, distance,adj,paths,rels,adj_path):
        '''
        input(s) shape: [batch_size, doc_len, sen_len, embedding_dim], 
                        [batch_size, doc_len * doc_len, embedding_dim_pos]
        output(s) shape: [batch_size, doc_len, n_class], [batch_size, doc_len, n_class], 
                         [batch_size, doc_len * doc_len, n_class]
        '''
        #output_graphs = self.graph_encoder(graph)
        s = self.get_clause_embedding(x)
        paths_a,rels_a=self.get_path_rel_embedding(paths,rels)
        x_pos, pred_pos = self.get_emotion_prediction(s, adj,adj_path)
        s_pred_pos = torch.cat([s, pred_pos], 2)
        x_cause, pred_cause = self.get_cause_prediction(s_pred_pos, adj,adj_path)
        pred_pair = self.get_pair_prediction(x_pos, x_cause, distance,paths_a,rels_a, adj)
        return pred_pos, pred_cause, pred_pair

def load_embeddings(pretrain_embed_path):
    print("Loading glove concept embeddings with pooling:", pretrain_embed_path)
    concept_vec = np.load(pretrain_embed_path)
    print("done!")
    return concept_vec


############################################ TRAIN #####################################################
def train_and_eval(Model, pos_cause_criterion, pair_criterion, optimizer):
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(
        embedding_dim, embedding_dim_pos, train_file_path, w2v_file)
    word_embedding = torch.from_numpy(word_embedding)
    # Train distance embeddings
    pos_embedding = torch.autograd.Variable(torch.from_numpy(pos_embedding))
    pos_embedding.requires_grad_(True)
    pretrain_cpt_emd_path = ".../concept_glove.max.npy"
    pretrain_rel_emd_path = ".../relation_glove.max.npy"
    pretrained_concept_emd = load_embeddings(pretrain_cpt_emd_path)
    pretrained_relation_emd = load_embeddings(pretrain_rel_emd_path)
    print("pretrained_concept_emd.shape:", pretrained_concept_emd.shape)
    print("pretrained_relation_emd.shape:", pretrained_relation_emd.shape)
     # add one concept vec for dummy concept
    concept_dim = pretrained_concept_emd.shape[1]
    concept_num = pretrained_concept_emd.shape[0] + 1  # for dummy concept
    pretrained_concept_emd = np.insert(pretrained_concept_emd, 0, np.zeros((1, concept_dim)), 0)

    relation_num = pretrained_relation_emd.shape[0] * 2 + 1  # for inverse and dummy relations
    relation_dim = pretrained_relation_emd.shape[1]
    pretrained_relation_emd = np.concatenate((pretrained_relation_emd, pretrained_relation_emd))
    pretrained_relation_emd = np.insert(pretrained_relation_emd, 0, np.zeros((1, relation_dim)), 0)

    pretrained_concept_emd = torch.FloatTensor(pretrained_concept_emd)
    pretrained_relation_emd = torch.FloatTensor(pretrained_relation_emd)  # torch.FloatTensor(pretrained_relation_emd)
    acc_cause_list, p_cause_list, r_cause_list, f1_cause_list = [], [], [], []
    acc_pos_list, p_pos_list, r_pos_list, f1_pos_list = [], [], [], []
    acc_pair_list, p_pair_list, r_pair_list, f1_pair_list = [], [], [], []
    Model.zero_grad()
    #best_model_wts = copy.deepcopy(Model.state_dict())
    #################################### LOOP OVER FOLDS ####################################
    for fold in range(1, 11):
        print('############# fold {} begin ###############'.format(fold))
        ############################# RE-INITIALIZE MODEL PARAMETERS #############################
        for layer in Model.parameters():
            nn.init.uniform_(layer.data, -0.10, 0.10)
        #################################### TRAIN/TEST DATA ####################################
        train_file_name = 'fold{}_train.txt'.format(fold)
        test_file_name = 'fold{}_test.txt'.format(fold)

        tr_y_position, tr_y_cause, tr_y_pair, tr_x, tr_sen_len, tr_doc_len, tr_distance, tr_adj,tr_doc_id_list,tr_paths,tr_rels,tr_adj_path = load_data_pair(
                        '../data_combine_eng/'+train_file_name, word_id_mapping, max_doc_len, max_sen_len,max_path_num,max_path_len,max_rel_len)
        test_y_position, test_y_cause, test_y_pair, test_x, test_sen_len, test_doc_len, test_distance, test_adj,test_doc_id_list,test_paths,test_rels,test_adj_path = load_data_pair(
            '../data_combine_eng/'+test_file_name, word_id_mapping, max_doc_len, max_sen_len,max_path_num,max_path_len,max_rel_len)
        max_f1_cause, max_f1_pos, max_f1_pair, max_f1_avg = [-1.] * 4
        #################################### LOOP OVER EPOCHS ####################################
        for epoch in range(1, training_epochs + 1):
            step = 1
            #################################### GET BATCH DATA ####################################
            for train, _ in get_batch_data_pair(
                tr_x, tr_sen_len, tr_doc_len, tr_y_position, tr_y_cause, tr_y_pair, tr_distance, tr_adj, tr_doc_id_list,tr_paths,tr_rels,tr_adj_path,batch_size):
                tr_x_batch, tr_sen_len_batch, tr_doc_len_batch, tr_true_y_pos, tr_true_y_cause, \
                tr_true_y_pair,tr_adj_batch, tr_distance_batch,tr_doc_id_list_batch,tr_paths_batch,tr_rels_batch,tr_adj_path_batch = train
                Model.train()
                tr_pred_y_pos, tr_pred_y_cause, tr_pred_y_pair = Model(embedding_lookup(word_embedding, \
                tr_x_batch), embedding_lookup(pos_embedding, tr_distance_batch),tr_adj_batch.to(device),\
                embedding_lookup(pretrained_concept_emd,tr_paths_batch),embedding_lookup(pretrained_relation_emd,tr_rels_batch),tr_adj_path_batch.to(device))
                ############################## LOSS FUNCTION AND OPTIMIZATION ##############################
                loss = pos_cause_criterion(tr_true_y_pos, tr_pred_y_pos, tr_doc_len_batch)*pos + \
                pos_cause_criterion(tr_true_y_cause, tr_pred_y_cause, tr_doc_len_batch)*cause + \
                pair_criterion(tr_true_y_pair, tr_pred_y_pair, tr_doc_len_batch)*pair
                #loss =  pos_cause_criterion(tr_true_y_pos, tr_pred_y_cause, tr_doc_len_batch)*pos +\
                #pos_cause_criterion(tr_true_y_cause, tr_pred_y_pos, tr_doc_len_batch)*cause+ \
                #pair_criterion(tr_true_y_pair, tr_pred_y_pair, tr_doc_len_batch)*pair
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #################################### PRINT AFTER EPOCHS ####################################
                if step % 25 == 0:
                    # print(Model.pair_linear.weight.shape); print(Model.pair_linear.weight.grad)
                    print('Fold {}, Epoch {}, step {}: train loss {:.4f} '.format(fold, epoch, step, loss))
                    acc, p, r, f1 = acc_prf_aux(tr_pred_y_pos, tr_true_y_pos, tr_doc_len_batch)
                    #acc, p, r, f1 = acc_prf_aux(tr_pred_y_cause, tr_true_y_pos, tr_doc_len_batch)
                    print('emotion_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 score {:.4f}'.format(
                            acc, p, r, f1))
                    acc, p, r, f1 = acc_prf_aux(tr_pred_y_cause, tr_true_y_cause, tr_doc_len_batch)
                    #acc, p, r, f1 = acc_prf_aux(tr_pred_y_pos, tr_true_y_cause, tr_doc_len_batch)
                    print('cause_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 score {:.4f}'.format(
                            acc, p, r, f1))
                    acc, p, r, f1 = acc_prf_pair(tr_pred_y_pair, tr_true_y_pair, tr_doc_len_batch)
                    print('pair_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 score {:.4f}'.format(
                            acc, p, r, f1)) 
                step += 1
            #################################### TEST ON 1 FOLD ####################################
            with torch.no_grad():
                Model.eval()
                test_pred_y_pos, test_pred_y_cause, test_pred_y_pair = [],[],[]
                test_y_position_all, test_y_cause_all, test_y_pair_all = [],[],[]
                test_doc_len_all=[]
                for test, _ in get_batch_data_pair(test_x, test_sen_len, test_doc_len, test_y_position, test_y_cause, test_y_pair, test_distance, test_adj, test_doc_id_list,test_paths,test_rels,test_adj_path,batch_size):
                    test_x_batch, test_sen_len_batch, test_doc_len_batch, test_true_y_pos, test_true_y_cause, \
                    test_true_y_pair,test_adj_batch, test_distance_batch,test_doc_id_list_batch,test_paths_batch,test_rels_batch,test_adj_path_batch = test

                    test_pred_y_pos_batch, test_pred_y_cause_batch, test_pred_y_pair_batch = Model(embedding_lookup(word_embedding, \
                    test_x_batch), embedding_lookup(pos_embedding, test_distance_batch), test_adj_batch.to(device),\
                    embedding_lookup(pretrained_concept_emd,test_paths_batch),embedding_lookup(pretrained_relation_emd,test_rels_batch),test_adj_path_batch.to(device))
                    ############################## LOSS FUNCTION AND OPTIMIZATION ##############################
    
                    #################################### PRINT AFTER EPOCHS ####################################
                    test_pred_y_pos.extend(test_pred_y_pos_batch)
                    test_pred_y_cause.extend(test_pred_y_cause_batch)
                    test_pred_y_pair.extend(test_pred_y_pair_batch)
                    test_y_position_all.extend(test_true_y_pos)
                    test_y_cause_all.extend(test_true_y_cause)
                    test_y_pair_all.extend(test_true_y_pair)
                    test_doc_len_all.extend(test_doc_len_batch)

                acc, p, r, f1 = acc_prf_aux(torch.stack(test_pred_y_pos), torch.stack(test_y_position_all), torch.stack(test_doc_len_all))
                result_avg_pos = [acc, p, r, f1]
                if f1 > max_f1_pos:
                    max_acc_pos, max_p_pos, max_r_pos, max_f1_pos = acc, p, r, f1
                print('emotion_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_acc_pos, max_p_pos, max_r_pos, max_f1_pos))

                acc, p, r, f1 = acc_prf_aux(torch.stack(test_pred_y_cause), torch.stack(test_y_cause_all), torch.stack(test_doc_len_all))
                #acc, p, r, f1 = acc_prf_aux(torch.stack(test_pred_y_pos), torch.stack(test_y_cause_all), torch.stack(test_doc_len_all))
                result_avg_cause = [acc, p, r, f1]
                if f1 > max_f1_cause:
                    max_acc_cause, max_p_cause, max_r_cause, max_f1_cause = acc, p, r, f1
                print('cause_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_acc_cause, max_p_cause, max_r_cause, max_f1_cause))

                acc, p, r, f1 = acc_prf_pair(torch.stack(test_pred_y_pair), torch.stack(test_y_pair_all), torch.stack(test_doc_len_all))
                result_avg_pair = [acc, p, r, f1]
                if f1 > max_f1_pair:
                    max_acc_pair, max_p_pair, max_r_pair, max_f1_pair = acc, p, r, f1
                print('pair_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_acc_pair, max_p_pair, max_r_pair, max_f1_pair))

            #################################### STORE BETTER PAIR F1 ####################################
            if result_avg_pair[-1] > max_f1_avg:
                max_f1_avg = result_avg_pair[-1]
                result_avg_cause_max = result_avg_cause
                result_avg_pos_max = result_avg_pos
                result_avg_pair_max = result_avg_pair

            print('avg max cause: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(
                result_avg_cause_max[0], result_avg_cause_max[1], result_avg_cause_max[2], result_avg_cause_max[3]))
            print('avg max pos: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(
                result_avg_pos_max[0], result_avg_pos_max[1], result_avg_pos_max[2], result_avg_pos_max[3]))
            print('avg max pair: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                result_avg_pair_max[0], result_avg_pair_max[1], result_avg_pair_max[2], result_avg_pair_max[3]))

        print('############# fold {} end ###############'.format(fold))
        acc_cause_list.append(result_avg_cause_max[0])
        p_cause_list.append(result_avg_cause_max[1])
        r_cause_list.append(result_avg_cause_max[2])
        f1_cause_list.append(result_avg_cause_max[3])
        acc_pos_list.append(result_avg_pos_max[0])
        p_pos_list.append(result_avg_pos_max[1])
        r_pos_list.append(result_avg_pos_max[2])
        f1_pos_list.append(result_avg_pos_max[3])
        acc_pair_list.append(result_avg_pair_max[0])
        p_pair_list.append(result_avg_pair_max[1])
        r_pair_list.append(result_avg_pair_max[2])
        f1_pair_list.append(result_avg_pair_max[3])
       
    #################################### FINAL TEST RESULTS ON 10 FOLDS ####################################
    all_results = [acc_cause_list, p_cause_list, r_cause_list, f1_cause_list, \
    acc_pos_list, p_pos_list, r_pos_list, f1_pos_list, acc_pair_list, p_pair_list, r_pair_list, f1_pair_list,]
    acc_cause, p_cause, r_cause, f1_cause, acc_pos, p_pos, r_pos, f1_pos, acc_pair, p_pair, r_pair, f1_pair = \
        map(lambda x: np.array(x).mean(), all_results)
    print('\ncause_predict_f1: test f1 in 10 fold: {}'.format(np.array(f1_cause_list).reshape(-1,1)))
    print('\ncause_predict_p: test p in 10 fold: {}'.format(np.array(p_cause_list).reshape(-1,1)))
    print('\ncause_predict_r: test r in 10 fold: {}'.format(np.array(r_cause_list).reshape(-1,1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_cause, p_cause, r_cause, f1_cause))
    print('emotion_predict_f1: test f1 in 10 fold: {}'.format(np.array(f1_pos_list).reshape(-1,1)))
    print('emotion_predict_p: test p in 10 fold: {}'.format(np.array(p_pos_list).reshape(-1,1)))
    print('emotion_predict_r: test r in 10 fold: {}'.format(np.array(r_pos_list).reshape(-1,1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_pos, p_pos, r_pos, f1_pos))
    print('pair_predict_f1: test f1 in 10 fold: {}'.format(np.array(f1_pair_list).reshape(-1,1)))
    print('pair_predict_p: test p in 10 fold: {}'.format(np.array(p_pair_list).reshape(-1,1)))
    print('pair_predict_r: test r in 10 fold: {}'.format(np.array(r_pair_list).reshape(-1,1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_pair, p_pair, r_pair, f1_pair))
def set_random_seed(seed, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

############################################### MAIN ########################################################
def main():
    set_random_seed(129,True)
    Model = RSTIAG(embedding_dim, embedding_dim_pos, max_sen_len, max_doc_len, \
    keep_prob1, keep_prob2, keep_prob3, n_hidden, n_class,concept_dim,graph_hidden_dim,graph_output_dim,max_path_num,max_path_len,max_rel_len)
    Model.to(device)
    print(Model)
    pos_cause_criterion = ce_loss_aux(); pair_criterion = ce_loss_pair(diminish_factor)
    optimizer = optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    total_params = sum(p.numel() for p in Model.parameters())
    print('total_params=',total_params)
    train_and_eval(Model, pos_cause_criterion, pair_criterion, optimizer)

def get_right(pred_y, true_y, doc_len, average='weighted'):
    _, true_indices = torch.max(true_y, 2)
    _, pred_indices = torch.max(pred_y, 2)
    true_indices_masked = [];
    pred_indices_masked = []
    for i in range(len(doc_len)):
        true_indices_masked.extend(true_indices[i, :doc_len[i]])
        pred_indices_masked.extend(pred_indices[i, :doc_len[i]])
    acc, p, r, f1 = metrics(true_indices_masked, pred_indices_masked)
    if acc > 0.9:
        return True
    else:
        return False


def infer():
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(
        embedding_dim, embedding_dim_pos, train_file_path, w2v_file)
    word_embedding = torch.from_numpy(word_embedding)
    # Train distance embeddings
    pos_embedding = torch.autograd.Variable(torch.from_numpy(pos_embedding))
    pos_embedding.requires_grad_(True)
    pretrain_cpt_emd_path = ".../concept_glove.max.npy"
    pretrain_rel_emd_path = ".../relation_glove.max.npy"
    pretrained_concept_emd = load_embeddings(pretrain_cpt_emd_path)
    pretrained_relation_emd = load_embeddings(pretrain_rel_emd_path)
    print("pretrained_concept_emd.shape:", pretrained_concept_emd.shape)
    print("pretrained_relation_emd.shape:", pretrained_relation_emd.shape)
     # add one concept vec for dummy concept
    concept_dim = pretrained_concept_emd.shape[1]
    concept_num = pretrained_concept_emd.shape[0] + 1  # for dummy concept
    pretrained_concept_emd = np.insert(pretrained_concept_emd, 0, np.zeros((1, concept_dim)), 0)

    relation_num = pretrained_relation_emd.shape[0] * 2 + 1  # for inverse and dummy relations
    relation_dim = pretrained_relation_emd.shape[1]
    pretrained_relation_emd = np.concatenate((pretrained_relation_emd, pretrained_relation_emd))
    pretrained_relation_emd = np.insert(pretrained_relation_emd, 0, np.zeros((1, relation_dim)), 0)

    pretrained_concept_emd = torch.FloatTensor(pretrained_concept_emd)
    pretrained_relation_emd = torch.FloatTensor(pretrained_relation_emd) 

    Model = RSTIAG(embedding_dim, embedding_dim_pos, max_sen_len, max_doc_len, \
    keep_prob1, keep_prob2, keep_prob3, n_hidden, n_class,concept_dim,graph_hidden_dim,graph_output_dim,max_path_num,max_path_len,max_rel_len)
    Model.to(device)
    Model.eval()
    test_file_name = '.../data_combine_eng/all_data_pair.txt'
    
    test_y_position, test_y_cause, test_y_pair, test_x, test_sen_len, test_doc_len, test_distance, test_adj,test_doc_id_list,test_paths,test_rels,test_adj_path = load_data_pair(
            test_file_name, word_id_mapping, max_doc_len, max_sen_len,max_path_num,max_path_len,max_rel_len)
    max_f1_cause, max_f1_pos, max_f1_pair, max_f1_avg = [-1.] * 4
    true_list_1 = []
    true_list_2 = []
    true_list_3 = []
    index = 1
    with torch.no_grad():
        Model.eval()
        test_pred_y_pos, test_pred_y_cause, test_pred_y_pair = [],[],[]
        test_y_position_all, test_y_cause_all, test_y_pair_all = [],[],[]
        test_doc_id=[]
        test_rst=[]
        test_iag=[]
        test_doc_len_all=[]
        for test, _ in get_batch_data_pair(test_x, test_sen_len, test_doc_len, test_y_position, test_y_cause, test_y_pair, test_distance, test_adj, test_doc_id_list,test_paths,test_rels,test_adj_path,batch_size):
            test_x_batch, test_sen_len_batch, test_doc_len_batch, test_true_y_pos, test_true_y_cause, \
            test_true_y_pair,test_adj_batch, test_distance_batch,test_doc_id_list_batch,test_paths_batch,test_rels_batch,test_adj_path_batch = test
            test_pred_y_pos_batch, test_pred_y_cause_batch, test_pred_y_pair_batch = Model(embedding_lookup(word_embedding, \
            test_x_batch), embedding_lookup(pos_embedding, test_distance_batch), test_adj_batch.to(device),\
            embedding_lookup(pretrained_concept_emd,test_paths_batch),embedding_lookup(pretrained_relation_emd,test_rels_batch),test_adj_path_batch.to(device))
            ############################## LOSS FUNCTION AND OPTIMIZATION ##############################
            #################################### PRINT AFTER EPOCHS ####################################
            test_pred_y_pos.extend(test_pred_y_pos_batch)
            test_pred_y_cause.extend(test_pred_y_cause_batch)
            test_pred_y_pair.extend(test_pred_y_pair_batch)
            test_y_position_all.extend(test_true_y_pos)
            test_y_cause_all.extend(test_true_y_cause)
            test_y_pair_all.extend(test_true_y_pair)
            test_doc_len_all.extend(test_doc_len_batch)
            test_doc_id.extend(test_doc_id_list_batch)
            test_rst.extend(test_adj_batch)
            test_iag.extend(test_adj_path_batch)

        #情感子句   
        acc, p, r, f1 = acc_prf_aux(torch.stack(test_pred_y_pos), torch.stack(test_y_position_all), torch.stack(test_doc_len_all))
        print('emotion_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
        #原因子句
        acc, p, r, f1 = acc_prf_aux(torch.stack(test_pred_y_cause), torch.stack(test_y_cause_all), torch.stack(test_doc_len_all))
        print('cause_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
        #子句对
        acc, p, r, f1 = acc_prf_pair(torch.stack(test_pred_y_pair), torch.stack(test_y_pair_all), torch.stack(test_doc_len_all))
        print('pair_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))


if __name__ == "__main__":
    main()
    #infer()
