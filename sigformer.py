"""
SIGformer: Sign-Aware Graph Transformer for Recommendation
符号感知图Transformer推荐模型

注意：这是一个框架实现，具体细节需要参考原论文
如果找不到官方实现，可以使用 SGCN 或 SiGAT 作为替代基线
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGformer(nn.Module):
    """
    SIGformer: 使用 Transformer 架构处理符号社交网络的推荐模型
    
    核心思想：
    1. 分别对正边和负边进行注意力聚合
    2. 使用 Transformer 的 multi-head attention 捕获复杂交互
    3. 结合平衡理论（结构平衡约束）
    
    如果无法找到原论文，可以考虑：
    - SGCN (Signed Graph Convolutional Network)
    - SiGAT (Signed Graph Attention Network)
    作为符号感知的基线模型
    """
    
    def __init__(self, dataset, args):
        super(SIGformer, self).__init__()
        
        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING = "rating:float"
        
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.embedding_size = args.embedding_size
        self.n_heads = getattr(args, 'n_heads', 4)  # multi-head attention
        self.n_layers = getattr(args, 'gnn_layer_size', 2)
        self.device = args.device
        
        # Embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # 符号网络处理
        self.net_matrix = dataset.net_matrix
        self.interaction_matrix = dataset.interaction_matrix
        
        # 构建正/负邻接矩阵
        self.pos_adj, self.neg_adj = self._split_signed_graph(self.net_matrix)
        
        # Transformer 层
        self.pos_attention = nn.ModuleList([
            nn.MultiheadAttention(self.embedding_size, self.n_heads, batch_first=True)
            for _ in range(self.n_layers)
        ])
        
        self.neg_attention = nn.ModuleList([
            nn.MultiheadAttention(self.embedding_size, self.n_heads, batch_first=True)
            for _ in range(self.n_layers)
        ])
        
        # 融合层
        self.fusion = nn.Linear(self.embedding_size * 3, self.embedding_size)
        
        # U-I 图卷积（用于物品侧）
        self.ui_graph = self._build_ui_graph(self.interaction_matrix)
        
        print(f"[SIGformer] Users: {self.n_users}, Items: {self.n_items}, "
              f"Heads: {self.n_heads}, Layers: {self.n_layers}")
    
    def _split_signed_graph(self, net_matrix):
        """
        将符号网络分解为正边和负边的邻接矩阵
        """
        net_coo = net_matrix.tocoo()
        
        # 正边
        pos_mask = net_coo.data > 0
        pos_rows = net_coo.row[pos_mask]
        pos_cols = net_coo.col[pos_mask]
        pos_data = np.ones_like(pos_rows, dtype=np.float32)
        pos_adj = sp.coo_matrix((pos_data, (pos_rows, pos_cols)),
                               shape=(self.n_users, self.n_users))
        
        # 负边
        neg_mask = net_coo.data < 0
        neg_rows = net_coo.row[neg_mask]
        neg_cols = net_coo.col[neg_mask]
        neg_data = np.ones_like(neg_rows, dtype=np.float32)
        neg_adj = sp.coo_matrix((neg_data, (neg_rows, neg_cols)),
                               shape=(self.n_users, self.n_users))
        
        # 转为 PyTorch sparse tensor
        pos_adj = self._to_torch_sparse(pos_adj).to(self.device)
        neg_adj = self._to_torch_sparse(neg_adj).to(self.device)
        
        return pos_adj, neg_adj
    
    def _to_torch_sparse(self, sp_matrix):
        """将 scipy sparse matrix 转为 PyTorch sparse tensor"""
        sp_coo = sp_matrix.tocoo()
        indices = torch.LongTensor(np.vstack([sp_coo.row, sp_coo.col]))
        values = torch.FloatTensor(sp_coo.data)
        shape = torch.Size(sp_coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def _build_ui_graph(self, interaction_matrix):
        """构建 U-I 二部图"""
        R = interaction_matrix.tocoo()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items,
                                self.n_users + self.n_items), dtype=np.float32)
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.tocoo()
        
        # 归一化
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt
        
        return self._to_torch_sparse(norm_adj.tocoo()).to(self.device)
    
    def _get_neighbors(self, adj_matrix, node_emb):
        """
        从邻接矩阵获取邻居嵌入
        返回: (n_users, max_neighbors, embedding_size)
        
        注意：这里简化处理，实际可能需要更复杂的邻居采样
        """
        # 简化版：直接用稀疏矩阵乘法聚合
        neighbor_emb = torch.sparse.mm(adj_matrix, node_emb)  # (n_users, d)
        return neighbor_emb.unsqueeze(1)  # (n_users, 1, d) 作为单个"邻居"
    
    def forward(self):
        """
        SIGformer 前向传播
        1. 对正边和负边分别做 Transformer attention
        2. 融合正边、负边、自身嵌入
        3. 结合 U-I 图信息
        """
        user_emb = self.user_embedding.weight  # (n_users, d)
        item_emb = self.item_embedding.weight  # (n_items, d)
        
        pos_emb = user_emb
        neg_emb = user_emb
        
        # 多层 Transformer
        for layer in range(self.n_layers):
            # 正边注意力
            pos_neighbors = self._get_neighbors(self.pos_adj, pos_emb)  # (n_users, 1, d)
            query = user_emb.unsqueeze(1)  # (n_users, 1, d)
            
            # Multi-head attention (简化版：query=key=value=邻居)
            pos_out, _ = self.pos_attention[layer](query, pos_neighbors, pos_neighbors)
            pos_emb = pos_out.squeeze(1)  # (n_users, d)
            
            # 负边注意力
            neg_neighbors = self._get_neighbors(self.neg_adj, neg_emb)
            neg_out, _ = self.neg_attention[layer](query, neg_neighbors, neg_neighbors)
            neg_emb = neg_out.squeeze(1)
        
        # 融合正边、负边、原始嵌入
        user_final = self.fusion(torch.cat([user_emb, pos_emb, neg_emb], dim=1))
        
        # 物品侧：简单的图卷积
        all_emb = torch.cat([user_final, item_emb])
        item_final = torch.sparse.mm(self.ui_graph, all_emb)[self.n_users:]
        
        return user_final, item_final
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores
    
    def calculate_loss(self, interaction):
        scores = self.predict(interaction)
        rating_gt = interaction[self.RATING]
        loss = F.mse_loss(scores, rating_gt)
        return loss


# ========== 备选：更成熟的符号图模型 ==========

class SGCN_Alternative(nn.Module):
    """
    Signed Graph Convolutional Network (备选基线)
    
    参考：
    - Signed Graph Convolutional Network (ICDM 2018)
    - 更成熟，有官方实现
    
    如果 SIGformer 找不到官方代码，可以用这个替代
    """
    
    def __init__(self, dataset, args):
        super(SGCN_Alternative, self).__init__()
        
        # TODO: 实现 SGCN
        # 核心：分别对正边和负边做卷积，然后聚合
        pass


class SiGAT_Alternative(nn.Module):
    """
    Signed Graph Attention Network (备选基线)
    
    参考：
    - SiGAT: Signed Graph Attention Network (相关论文)
    - 使用注意力机制处理符号边
    """
    
    def __init__(self, dataset, args):
        super(SiGAT_Alternative, self).__init__()
        
        # TODO: 实现 SiGAT
        # 核心：对正/负边分别计算注意力权重
        pass


# ========== 使用指南 ==========
"""
如何选择符号感知基线：

1. 优先级顺序：
   - SIGformer (如果有最新论文/代码)
   - SGCN (成熟，ICDM 2018)
   - SiGAT (注意力机制)
   - 或者直接使用你项目中已有的 sgcn_SGA.py, sigat_SGA.py

2. 查找资源：
   - GitHub: "SIGformer recommendation"
   - Papers with Code: 搜索 "signed graph recommendation"
   - 相关会议：SIGIR, WWW, RecSys, ICDM

3. 如果找不到 SIGformer：
   - 在论文中说明："我们尝试复现 SIGformer，但官方代码未公开，
     因此使用 SGCN 作为符号感知基线"
   - 或者直接使用你已有的 sgcn_SGA.py
"""
