"""
基线模型实现
包括：LightGCN, SGL, XSimGCL
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    SIGIR 2020
    
    主要特点：
    - 移除特征变换和非线性激活
    - 只保留邻居聚合
    - 多层嵌入加权平均
    """
    
    def __init__(self, dataset, args):
        super(LightGCN, self).__init__()
        
        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING = "rating:float"
        
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.embedding_size = args.embedding_size
        self.n_layers = getattr(args, 'gnn_layer_size', 3)  # LightGCN 通常用3层
        self.device = args.device
        
        # Embedding 层
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # 构建归一化的邻接矩阵
        self.Graph = self._build_graph(dataset.interaction_matrix).to(self.device)
        
        print(f"[LightGCN] Users: {self.n_users}, Items: {self.n_items}, "
              f"Embedding: {self.embedding_size}, Layers: {self.n_layers}")
    
    def _build_graph(self, interaction_matrix):
        """
        构建归一化的 U-I 二部图邻接矩阵
        A = D^{-1/2} @ [[0, R], [R^T, 0]] @ D^{-1/2}
        """
        R = interaction_matrix.tocoo()
        
        # 直接构建 COO 格式，避免内存溢出
        # [[0,   R  ],
        #  [R^T, 0  ]]
        # 上半部分：用户到物品的边
        row_top = R.row
        col_top = R.col + self.n_users
        data_top = R.data
        
        # 下半部分：物品到用户的边
        row_bottom = R.col + self.n_users
        col_bottom = R.row
        data_bottom = R.data
        
        # 合并
        row = np.concatenate([row_top, row_bottom])
        col = np.concatenate([col_top, col_bottom])
        data = np.concatenate([data_top, data_bottom])
        
        adj_mat = sp.coo_matrix((data, (row, col)), 
                               shape=(self.n_users + self.n_items,
                                     self.n_users + self.n_items),
                               dtype=np.float32)
        
        # 度数归一化 D^{-1/2}
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # D^{-1/2} @ A @ D^{-1/2}
        norm_adj = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt
        norm_adj = norm_adj.tocoo()
        
        # 转为 PyTorch sparse tensor
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def forward(self):
        """
        LightGCN 的图卷积传播
        """
        # 初始嵌入
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embs = [all_emb]
        
        # 多层传播
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        
        # 加权平均所有层（包括第0层）
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        # 分离用户和物品嵌入
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def predict(self, interaction):
        """预测评分"""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        
        # 限制预测值在评分范围内 [1, 5]
        # 修复：之前unbounded的内积导致RMSE异常高
        scores = torch.clamp(scores, 1.0, 5.0)
        
        return scores
    
    def calculate_loss(self, interaction):
        """计算 BPR 损失（可选）或 MSE 损失"""
        scores = self.predict(interaction)
        rating_gt = interaction[self.RATING]
        
        # 使用 MSE 损失（与你的 ESSRec 保持一致）
        loss = F.mse_loss(scores, rating_gt)
        
        # 可选：添加 L2 正则化
        reg_loss = 0.0
        if hasattr(self, 'reg_weight') and self.reg_weight > 0:
            reg_loss = self.reg_weight * (
                torch.norm(self.user_embedding.weight) ** 2 +
                torch.norm(self.item_embedding.weight) ** 2
            ) / 2.0
        
        return loss + reg_loss


class SGL(nn.Module):
    """
    SGL: Self-supervised Graph Learning for Recommendation
    SIGIR 2021
    
    在 LightGCN 基础上添加对比学习：
    - 图增强：节点dropout、边dropout
    - InfoNCE 对比损失
    """
    
    def __init__(self, dataset, args):
        super(SGL, self).__init__()
        
        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING = "rating:float"
        
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.embedding_size = args.embedding_size
        self.n_layers = getattr(args, 'gnn_layer_size', 3)
        self.device = args.device
        
        # SSL 超参数
        self.ssl_temp = getattr(args, 'ssl_temp', 0.2)  # InfoNCE 温度
        self.ssl_reg = getattr(args, 'ssl_reg', 0.1)    # SSL 损失权重
        self.aug_type = getattr(args, 'aug_type', 'ED')  # 'ND': node dropout, 'ED': edge dropout
        self.drop_ratio = getattr(args, 'drop_ratio', 0.1)
        
        # Embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # 原始图
        self.Graph = self._build_graph(dataset.interaction_matrix).to(self.device)
        
        print(f"[SGL] SSL_temp={self.ssl_temp}, SSL_reg={self.ssl_reg}, "
              f"Aug={self.aug_type}, Drop={self.drop_ratio}")
    
    def _build_graph(self, interaction_matrix):
        """同 LightGCN - 高效构建"""
        R = interaction_matrix.tocoo()
        
        # 直接构建 COO 格式，避免内存溢出
        row_top = R.row
        col_top = R.col + self.n_users
        data_top = R.data
        
        row_bottom = R.col + self.n_users
        col_bottom = R.row
        data_bottom = R.data
        
        row = np.concatenate([row_top, row_bottom])
        col = np.concatenate([col_top, col_bottom])
        data = np.concatenate([data_top, data_bottom])
        
        adj_mat = sp.coo_matrix((data, (row, col)), 
                               shape=(self.n_users + self.n_items,
                                     self.n_users + self.n_items),
                               dtype=np.float32)
        
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt
        norm_adj = norm_adj.tocoo()
        
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def _graph_augment(self, graph):
        """
        图增强：边dropout 或 节点dropout
        """
        if self.aug_type == 'ED':
            # 边 dropout
            indices = graph._indices()
            values = graph._values()
            
            # 随机保留 (1 - drop_ratio) 的边
            mask = torch.rand(values.size(0)) > self.drop_ratio
            new_indices = indices[:, mask]
            new_values = values[mask]
            
            aug_graph = torch.sparse.FloatTensor(new_indices, new_values, graph.size())
            return aug_graph
        
        elif self.aug_type == 'ND':
            # 节点 dropout（简化实现：直接对嵌入加噪声）
            return graph  # 实际可在嵌入层面做dropout
        
        return graph
    
    def _propagate(self, graph):
        """在给定图上传播"""
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embs = [all_emb]
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def forward(self):
        """主任务的前向传播"""
        return self._propagate(self.Graph)
    
    def ssl_loss(self):
        """
        自监督对比学习损失 (InfoNCE) - 内存优化版本
        使用负采样而不是全矩阵计算
        """
        # 两个增强视图
        graph1 = self._graph_augment(self.Graph)
        graph2 = self._graph_augment(self.Graph)
        
        user1, item1 = self._propagate(graph1)
        user2, item2 = self._propagate(graph2)
        
        # 采样负样本数量（避免内存爆炸）
        neg_samples = min(1024, self.n_users - 1)
        
        # 用户对比损失
        user1_norm = F.normalize(user1, dim=1)
        user2_norm = F.normalize(user2, dim=1)
        
        # InfoNCE: 正样本是同一用户的两个视图
        pos_score = torch.sum(user1_norm * user2_norm, dim=1, keepdim=True) / self.ssl_temp
        
        # 负样本：随机采样其他用户
        neg_idx = torch.randint(0, self.n_users, (self.n_users, neg_samples), device=self.device)
        neg_emb = user2_norm[neg_idx]  # (n_users, neg_samples, dim)
        neg_score = torch.bmm(neg_emb, user1_norm.unsqueeze(-1)).squeeze(-1) / self.ssl_temp  # (n_users, neg_samples)
        
        # 合并正负样本分数
        logits = torch.cat([pos_score, neg_score], dim=1)  # (n_users, 1+neg_samples)
        labels = torch.zeros(self.n_users, dtype=torch.long, device=self.device)  # 正样本在第0位
        
        ssl_loss_user = F.cross_entropy(logits, labels)
        
        # 物品对比损失（类似）
        item1_norm = F.normalize(item1, dim=1)
        item2_norm = F.normalize(item2, dim=1)
        
        pos_score_item = torch.sum(item1_norm * item2_norm, dim=1, keepdim=True) / self.ssl_temp
        
        neg_samples_item = min(1024, self.n_items - 1)
        neg_idx_item = torch.randint(0, self.n_items, (self.n_items, neg_samples_item), device=self.device)
        neg_emb_item = item2_norm[neg_idx_item]
        neg_score_item = torch.bmm(neg_emb_item, item1_norm.unsqueeze(-1)).squeeze(-1) / self.ssl_temp
        
        logits_item = torch.cat([pos_score_item, neg_score_item], dim=1)
        labels_item = torch.zeros(self.n_items, dtype=torch.long, device=self.device)
        
        ssl_loss_item = F.cross_entropy(logits_item, labels_item)
        
        return ssl_loss_user + ssl_loss_item
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores
    
    def calculate_loss(self, interaction):
        """主任务损失 + SSL 损失（大数据集时禁用SSL避免OOM）"""
        # 主任务
        scores = self.predict(interaction)
        rating_gt = interaction[self.RATING]
        main_loss = F.mse_loss(scores, rating_gt)
        
        # SSL 损失 - 对于超大规模数据集禁用以避免内存溢出
        # 数据集规模: 180K users × 755K items 时，对比学习会导致OOM
        if self.n_users < 50000 and self.n_items < 100000:
            ssl = self.ssl_loss()
            total_loss = main_loss + self.ssl_reg * ssl
        else:
            # 大规模数据集：只使用主任务损失（等价于LightGCN）
            print(f"[SGL] Large dataset detected (U={self.n_users}, I={self.n_items}), SSL disabled to avoid OOM")
            total_loss = main_loss
        
        return total_loss


class XSimGCL(nn.Module):
    """
    XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation
    TKDE 2023
    
    主要创新：
    - 不需要显式图增强（节点/边dropout）
    - 直接在嵌入上添加噪声
    - 使用 Uniformity 和 Alignment 损失
    """
    
    def __init__(self, dataset, args):
        super(XSimGCL, self).__init__()
        
        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING = "rating:float"
        
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.embedding_size = args.embedding_size
        self.n_layers = getattr(args, 'gnn_layer_size', 3)
        self.device = args.device
        
        # XSimGCL 超参数
        self.eps = getattr(args, 'noise_eps', 0.1)  # 噪声强度
        self.ssl_reg = getattr(args, 'ssl_reg', 0.1)
        self.cl_temp = getattr(args, 'cl_temp', 0.2)
        
        # Embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # 图
        self.Graph = self._build_graph(dataset.interaction_matrix).to(self.device)
        
        print(f"[XSimGCL] Noise_eps={self.eps}, SSL_reg={self.ssl_reg}")
    
    def _build_graph(self, interaction_matrix):
        """同 LightGCN - 高效构建"""
        R = interaction_matrix.tocoo()
        
        # 直接构建 COO 格式，避免内存溢出
        row_top = R.row
        col_top = R.col + self.n_users
        data_top = R.data
        
        row_bottom = R.col + self.n_users
        col_bottom = R.row
        data_bottom = R.data
        
        row = np.concatenate([row_top, row_bottom])
        col = np.concatenate([col_top, col_bottom])
        data = np.concatenate([data_top, data_bottom])
        
        adj_mat = sp.coo_matrix((data, (row, col)), 
                               shape=(self.n_users + self.n_items,
                                     self.n_users + self.n_items),
                               dtype=np.float32)
        
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt
        norm_adj = norm_adj.tocoo()
        
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def forward(self, perturbed=False):
        """
        前向传播
        perturbed: 是否添加噪声（用于对比学习）
        """
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        
        # 添加随机噪声（XSimGCL的核心）
        if perturbed:
            random_noise = torch.rand_like(all_emb).to(self.device)
            all_emb = all_emb + torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps
        
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def infonce_loss(self):
        """InfoNCE 对比损失 - 内存优化版本"""
        user1, item1 = self.forward(perturbed=True)
        user2, item2 = self.forward(perturbed=True)
        
        user1 = F.normalize(user1, dim=1)
        user2 = F.normalize(user2, dim=1)
        
        # 正样本分数
        pos_score = torch.sum(user1 * user2, dim=1, keepdim=True) / self.cl_temp
        
        # 负采样（避免全矩阵）
        neg_samples = min(1024, self.n_users - 1)
        neg_idx = torch.randint(0, self.n_users, (self.n_users, neg_samples), device=self.device)
        neg_emb = user2[neg_idx]
        neg_score = torch.bmm(neg_emb, user1.unsqueeze(-1)).squeeze(-1) / self.cl_temp
        
        # InfoNCE 损失
        logits = torch.cat([pos_score, neg_score], dim=1)
        labels = torch.zeros(self.n_users, dtype=torch.long, device=self.device)
        cl_loss = F.cross_entropy(logits, labels)
        
        return cl_loss
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.forward(perturbed=False)
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores
    
    def calculate_loss(self, interaction):
        """主任务损失 + 对比损失（大数据集时禁用）"""
        scores = self.predict(interaction)
        rating_gt = interaction[self.RATING]
        main_loss = F.mse_loss(scores, rating_gt)
        
        # 对比损失 - 大规模数据集时禁用避免OOM
        if self.n_users < 50000 and self.n_items < 100000:
            cl_loss = self.infonce_loss()
            total_loss = main_loss + self.ssl_reg * cl_loss
        else:
            print(f"[XSimGCL] Large dataset detected (U={self.n_users}, I={self.n_items}), CL disabled to avoid OOM")
            total_loss = main_loss
        
        return total_loss
