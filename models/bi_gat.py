import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BipartiteGraphAttentionLayer(nn.Module):
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(BipartiteGraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn
        self.att_head = att_head

        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

    #def forward(self, feat_src, feat_dst, adj_src, adj_dst):
    def forward(self, feat_src, feat_dst):
        batch_src, N_src, in_dim_src = feat_src.size()
        batch_dst, N_dst, in_dim_dst = feat_dst.size()

        assert in_dim_src == self.in_dim
        assert in_dim_dst == self.in_dim

        feat_src_ = feat_src.unsqueeze(1)
        h_src = torch.matmul(feat_src_, self.W)

        feat_dst_ = feat_dst.unsqueeze(1)
        h_dst = torch.matmul(feat_dst_, self.W)

        attn_src = torch.matmul(F.tanh(h_src), self.w_src)
        attn_dst = torch.matmul(F.tanh(h_dst), self.w_dst)

        attn = attn_src.expand(-1, -1, -1, N_dst) + attn_dst.expand(-1, -1, -1, N_src).permute(0, 1, 3, 2)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)

        attn = F.softmax(attn, dim=-1)
        feat_out = torch.matmul(attn, h_dst) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch_src, N_src, -1)
        feat_out = F.elu(feat_out)

        gate = F.sigmoid(self.H(feat_src))
        feat_out = gate * feat_out + (1 - gate) * feat_src

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim * self.att_head) + ')'
