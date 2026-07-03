import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .RecSSN import RecSSN
# from .SocialMF import SocialMF
# from .TDRec import TDRec

###########################################
# 2. ESSRec模型 (融合高阶聚合 + 对比学习)
###########################################
class ESSRec(nn.Module):
    def __init__(self, dataset, args):
        super(ESSRec, self).__init__()

        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING = "rating:float"

        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        print(f"Number of users: {self.n_users}, Number of items: {self.n_items}")

        self.interaction_matrix = dataset.interaction_matrix.astype(np.float32)
        self.net_matrix = dataset.net_matrix

        print(f"Interaction matrix shape: {self.interaction_matrix.shape}")
        print(f"Net matrix shape: {self.net_matrix.shape}")

        # 给符号网络加对角（I）
        self.net_matrix = self._prepare_net_matrix(self.net_matrix)

        # 从 args 中获取参数
        self.embedding_size = args.embedding_size
        self.gnn_layer_size = args.gnn_layer_size       # 原本多层卷积层数
        self.gnn_layer_size_k = args.gnn_layer_size_k   # 新增: 再叠加多阶聚合
        self.sim_threshold = args.sim_threshold
        self.device = args.device
        self.criterion = nn.MSELoss()
        self.alpha = getattr(args, "alpha", 0.5)
        self.ablation = getattr(args, "ablation", "full")
        self.contrastive_samples = getattr(args, "contrastive_samples", 4096)

        # 多阶卷积矩阵
        self.ACM = self.get_convolution_matrix(self.interaction_matrix).to(self.device)
        self.ACM_T = self.get_convolution_matrix(self.interaction_matrix.transpose()).to(self.device)
        self.SCM = self.get_convolution_matrix(self.net_matrix).to(self.device)
        self.SCM_abs = self.get_convolution_matrix(self._abs_net_matrix(self.net_matrix)).to(self.device)

        # Embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size).to(self.device)

        # 一些线性层
        self.sim_layer = nn.Linear(self.embedding_size * 2, 2).to(self.device)
        self.user_concat = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.item_concat = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.concat_user_item_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.concat_PON_layer = nn.Linear(self.embedding_size * 3, self.embedding_size).to(self.device)
        self.concat_PN_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.concat_user_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.concat_social_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)

        # Contrastive 超参
        self.contrast_margin = getattr(args, "contrast_margin", 1.0)

        # 预计算每个用户交互过的物品索引列表（用于打分方案A）
        inter_csr = self.interaction_matrix.tocsr()
        self.user_items_index = []
        for u in range(self.n_users):
            start, end = inter_csr.indptr[u], inter_csr.indptr[u + 1]
            items = inter_csr.indices[start:end]
            self.user_items_index.append(torch.tensor(items, dtype=torch.long, device=self.device))

        self._cache_net_edges()

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def forward(self):
        # U-I图上的卷积
        user_init_embedding, item_init_embedding = self.user_item_graph_convolution()
        # item 进一步聚合
        item_rep = torch.sparse.mm(self.ACM, item_init_embedding.to(self.device))

        # 拼接
        E = self.concat_user_item_layer(torch.cat([user_init_embedding, item_rep], dim=1))

        if self.ablation == "no_pno":
            user_social = torch.sparse.mm(self.SCM_abs, E)
            user_final_embedding = self.concat_social_layer(torch.cat([E, user_social], dim=1))
        else:
            # 对符号网络做多阶聚合
            P_total, N_total, O_total = None, None, None
            current_E = E
            for k in range(self.gnn_layer_size_k):
                P, N, O = self.graph_convolution(self.net_matrix, current_E)
                if P_total is None:
                    P_total = P
                    N_total = N
                    O_total = O
                else:
                    P_total += P
                    N_total += N
                    O_total += O
                current_E = (P + N + O) / 3

            # 取平均
            P = P_total / self.gnn_layer_size_k
            N = N_total / self.gnn_layer_size_k
            O = O_total / self.gnn_layer_size_k

            user_pn_embedding = self.concat_PN_layer(torch.cat([P, N], dim=1))
            user_final_embedding = self.concat_user_layer(torch.cat([user_pn_embedding, O], dim=1))
        
        # 修复：让item embedding也经过聚合，确保user和item在同一表示空间
        # item通过ACM_T从user侧聚合信息
        item_from_users = torch.sparse.mm(self.ACM_T, user_final_embedding)
        item_final_embedding = self.item_concat(torch.cat([item_init_embedding, item_from_users], dim=1))

        return user_final_embedding, item_final_embedding

    # 提供不改变 forward 接口的情况下，获取用户/物品嵌入（以及可选的 P/N/O）
    def get_user_item_embeddings(self, return_pno: bool = False):
        # U-I图上的卷积
        user_init_embedding, item_init_embedding = self.user_item_graph_convolution()
        # item 进一步聚合
        item_rep = torch.sparse.mm(self.ACM, item_init_embedding.to(self.device))

        # 拼接
        E = self.concat_user_item_layer(torch.cat([user_init_embedding, item_rep], dim=1))

        if self.ablation == "no_pno":
            user_social = torch.sparse.mm(self.SCM_abs, E)
            user_final_embedding = self.concat_social_layer(torch.cat([E, user_social], dim=1))
            P, N, O = None, None, None
        else:
            # 对符号网络做多阶聚合
            P_total, N_total, O_total = None, None, None
            current_E = E
            for k in range(self.gnn_layer_size_k):
                P, N, O = self.graph_convolution(self.net_matrix, current_E)
                if P_total is None:
                    P_total = P
                    N_total = N
                    O_total = O
                else:
                    P_total += P
                    N_total += N
                    O_total += O
                current_E = (P + N + O) / 3

            # 取平均
            P = P_total / self.gnn_layer_size_k
            N = N_total / self.gnn_layer_size_k
            O = O_total / self.gnn_layer_size_k

            user_pn_embedding = self.concat_PN_layer(torch.cat([P, N], dim=1))
            user_final_embedding = self.concat_user_layer(torch.cat([user_pn_embedding, O], dim=1))
        
        # 修复：让item embedding也经过聚合
        item_from_users = torch.sparse.mm(self.ACM_T, user_final_embedding)
        item_final_embedding = self.item_concat(torch.cat([item_init_embedding, item_from_users], dim=1))

        if return_pno:
            return user_final_embedding, item_final_embedding, P, N, O
        return user_final_embedding, item_final_embedding

    def calculate_loss(self, interaction, return_dict: bool = False):
        rating_gt = interaction[self.RATING].to(self.device)
        scores = self.predict(interaction)
        rec_loss = F.mse_loss(scores, rating_gt)

        cl_loss = torch.tensor(0.0, device=self.device)
        if self.alpha > 0 and self.ablation != "no_contrastive":
            user_emb, _ = self.get_user_item_embeddings(return_pno=False)
            cl_loss = self._contrastive_loss(user_emb)

        total_loss = (1 - self.alpha) * rec_loss + self.alpha * cl_loss
        if return_dict:
            return total_loss, {
                "rec_loss": rec_loss.detach().item(),
                "cl_loss": cl_loss.detach().item(),
                "total_loss": total_loss.detach().item(),
            }
        return total_loss

    @torch.no_grad()
    def score_edge(self, u_idx, v_idx, method: str = "A", tau: float = 1.0):
        """
        计算用户-用户有符号边 (u,v) 的打分与校准概率。
        - method: "A" 使用物品加权的相似度（推荐）
        - tau: 温度缩放系数

        输入可以是 int 或形如 (B,) 的 LongTensor；返回 dict:
          { 's': Tensor(B,), 'p_pos': Tensor(B,), 'p_neg': Tensor(B,) }
        """
        if isinstance(u_idx, int):
            u_idx = torch.tensor([u_idx], dtype=torch.long, device=self.device)
        if isinstance(v_idx, int):
            v_idx = torch.tensor([v_idx], dtype=torch.long, device=self.device)

        assert u_idx.shape == v_idx.shape

        U, I = self.get_user_item_embeddings(return_pno=False)
        E_u = U[u_idx]  # (B, d)
        E_v = U[v_idx]  # (B, d)

        if method.upper() == "A":
            # 物品加权上下文: 对每个样本 b，用 u_b 的物品集合 I(u_b) 做 softmax 加权，query 为 v_b
            d = E_u.size(1)
            B = u_idx.size(0)
            E_v_cond = torch.zeros(B, d, device=self.device)
            for b in range(B):
                u = int(u_idx[b].item())
                items = self.user_items_index[u]
                if items.numel() == 0:
                    # 无物品上下文时退化为 v 的自身表示
                    E_v_cond[b] = E_v[b]
                    continue
                e_items = I[items]  # (m, d)
                logits = torch.matmul(e_items, E_v[b])  # (m,)
                alpha = torch.softmax(logits, dim=0)  # (m,)
                ctx = torch.matmul(alpha, e_items)  # (d,)
                E_v_cond[b] = ctx
            # 余弦相似度
            s = F.cosine_similarity(E_u, E_v_cond, dim=1)
        else:
            # 退化到简单余弦（若未来启用B方案，可改为 P/N/O 融合）
            s = F.cosine_similarity(E_u, E_v, dim=1)

        # 温度缩放后的概率
        s_scaled = s / max(1e-6, tau)
        p_pos = torch.sigmoid(s_scaled)
        p_neg = torch.sigmoid(-s_scaled)
        return {"s": s, "p_pos": p_pos, "p_neg": p_neg}

    @torch.no_grad()
    def score_edges(self, uv_index: torch.Tensor, method: str = "A", tau: float = 1.0):
        """
        批量打分：uv_index 形状为 (2, N)，返回 (N, 4) 的张量: [u, v, p_pos, p_neg]
        """
        assert uv_index.dim() == 2 and uv_index.size(0) == 2
        u = uv_index[0].long().to(self.device)
        v = uv_index[1].long().to(self.device)
        out = self.score_edge(u, v, method=method, tau=tau)
        p_pos = out["p_pos"].unsqueeze(1)
        p_neg = out["p_neg"].unsqueeze(1)
        return torch.cat([u.unsqueeze(1).float(), v.unsqueeze(1).float(), p_pos.float(), p_neg.float()], dim=1)

    def user_item_graph_convolution(self):
        item_rep = torch.sparse.mm(self.ACM, self.item_embedding.weight.to(self.device))
        user_rep = torch.sparse.mm(self.ACM_T, self.user_embedding.weight.to(self.device))

        item_init_embedding = self.item_concat(
            torch.cat([self.item_embedding.weight.to(self.device),
                       torch.sparse.mm(self.ACM_T, item_rep)], dim=1))

        user_init_embedding = self.user_concat(
            torch.cat([self.user_embedding.weight.to(self.device),
                       torch.sparse.mm(self.ACM, user_rep)], dim=1))
        return user_init_embedding, item_init_embedding

    def graph_convolution(self, sparse_matrix, E):
        # P,N,O 拆分
        P, N, O = self.get_signed_split_matrices(sparse_matrix, E)
        PCM = self.get_convolution_matrix(P).to(self.device)
        OCM = self.get_convolution_matrix(O).to(self.device)
        NCM = self.get_convolution_matrix(N).to(self.device)

        p_now, n_now, o_now = None, None, None
        for i in range(self.gnn_layer_size):
            if i == 0:
                p_now = torch.mm(PCM, E)
                n_now = torch.mm(NCM, E)
                o_now = torch.mm(OCM, E)
            else:
                p1 = 0.5 * (torch.mm(PCM, p_now) + torch.mm(NCM, n_now))
                n1 = 0.5 * (torch.mm(NCM, p_now) + torch.mm(PCM, n_now))
                o1 = torch.mm(OCM, o_now)
                p_now, n_now, o_now = p1, n1, o1

        return p_now, n_now, o_now

    def get_signed_split_matrices(self, sparse_matrix, user_rep):
        row = torch.LongTensor(sparse_matrix.row).to(self.device)
        col = torch.LongTensor(sparse_matrix.col).to(self.device)
        data = sparse_matrix.data

        source_rep = user_rep[row]
        target_rep = user_rep[col]
        cos_sim = F.cosine_similarity(source_rep, target_rep).cpu().detach().numpy()

        sim_index = np.argwhere(cos_sim > self.sim_threshold).squeeze(axis=1)
        diff_index = np.argwhere(cos_sim <= self.sim_threshold).squeeze(axis=1)
        trust_index = np.argwhere(data == 1).squeeze(axis=1)
        distrust_index = np.argwhere(data == -1).squeeze(axis=1)
        total = np.union1d(trust_index, distrust_index)

        p_index = np.intersect1d(trust_index, sim_index)
        n_index = np.intersect1d(distrust_index, diff_index)
        o_index = np.setdiff1d(total, np.union1d(p_index, n_index))

        P = self.get_split_matrix(sparse_matrix, p_index)
        N = self.get_split_matrix(sparse_matrix, n_index)
        O = self.get_split_matrix(sparse_matrix, o_index)
        return P, N, O

    def get_split_matrix(self, sparse_matrix, index):
        row = sparse_matrix.row
        col = sparse_matrix.col
        row = row[index]
        col = col[index]
        data = np.ones(len(index))
        matrix = sp.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_users))
        return matrix

    def get_convolution_matrix(self, matrix):
        # mean聚合
        D = self.get_inverse_degree_matrix(matrix)
        L = D * matrix
        L = sp.coo_matrix(L)
        return self.get_sparse_tensor(L)

    def get_inverse_degree_matrix(self, matrix):
        sumArr = (matrix != 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        return sp.coo_matrix(sp.diags(diag), dtype=np.float32)

    def get_sparse_tensor(self, sparse_matrix):
        # 将 sparse_matrix.row, .col, .data 转换为 numpy 数组
        row = np.array(sparse_matrix.row)
        col = np.array(sparse_matrix.col)
        indices = np.vstack((row, col))
        data = np.array(sparse_matrix.data)
        # 使用 torch.sparse_coo_tensor
        i = torch.LongTensor(indices)
        d = torch.FloatTensor(data)
        return torch.sparse_coo_tensor(i, d, sparse_matrix.shape, device=self.device)

    def _abs_net_matrix(self, sparse_matrix: sp.coo_matrix) -> sp.coo_matrix:
        if not sp.isspmatrix_coo(sparse_matrix):
            sparse_matrix = sparse_matrix.tocoo()
        abs_mat = sparse_matrix.copy()
        abs_mat.data = np.ones_like(abs_mat.data)
        return abs_mat

    def _prepare_net_matrix(self, net_matrix: sp.coo_matrix) -> sp.coo_matrix:
        if not sp.isspmatrix_coo(net_matrix):
            net_matrix = net_matrix.tocoo()
        diag = np.ones(self.n_users)
        I = sp.coo_matrix(sp.diags(diag), dtype=np.float32)
        net_matrix = sp.coo_matrix(net_matrix + I)
        return net_matrix

    def _cache_net_edges(self):
        net = self.net_matrix.tocoo()
        self._net_rows = net.row.astype(np.int64)
        self._net_cols = net.col.astype(np.int64)
        self._net_signs = net.data.astype(np.int64)

    def update_net_matrix(self, new_net_matrix: sp.coo_matrix):
        self.net_matrix = self._prepare_net_matrix(new_net_matrix)
        self.SCM = self.get_convolution_matrix(self.net_matrix).to(self.device)
        self.SCM_abs = self.get_convolution_matrix(self._abs_net_matrix(self.net_matrix)).to(self.device)
        self._cache_net_edges()

    def _contrastive_loss(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        if self.net_matrix.nnz == 0:
            return torch.tensor(0.0, device=self.device)

        num_edges = len(self._net_rows)
        sample_size = min(num_edges, self.contrastive_samples)
        if sample_size == 0:
            return torch.tensor(0.0, device=self.device)

        idx = np.random.choice(num_edges, size=sample_size, replace=False)
        u = torch.tensor(self._net_rows[idx], dtype=torch.long, device=self.device)
        v = torch.tensor(self._net_cols[idx], dtype=torch.long, device=self.device)
        sign = torch.tensor(self._net_signs[idx], dtype=torch.long, device=self.device)

        e_u = user_embeddings[u]
        e_v = user_embeddings[v]
        cos_sim = F.cosine_similarity(e_u, e_v, dim=1)

        pos_mask = sign > 0
        neg_mask = sign < 0

        loss_pos = torch.tensor(0.0, device=self.device)
        if pos_mask.any():
            loss_pos = 1.0 - cos_sim[pos_mask]
            loss_pos = loss_pos.mean()

        loss_neg = torch.tensor(0.0, device=self.device)
        if neg_mask.any():
            loss_neg = torch.relu(cos_sim[neg_mask] + self.contrast_margin)
            loss_neg = loss_neg.mean()

        return loss_pos + loss_neg

class GraphRec(nn.Module):
    """
    简化版 GraphRec：
    - 用户表示 = 融合(从物品聚合的表示, 从社交图聚合的表示)
    - 物品表示 = 从用户聚合
    - 多层传播（gnn_layer_size），层间残差；最终做点积回归
    - 复用你现有的字段名与稀疏算子，确保与数据集/训练循环完全兼容
    """
    def __init__(self, dataset, args):
        super(GraphRec, self).__init__()

        # -------- 与 ESSRec 对齐的字段名 --------
        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING  = "rating:float"

        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        self.device          = args.device
        self.embedding_size  = args.embedding_size
        self.gnn_layer_size  = args.gnn_layer_size
        self.lr              = getattr(args, "lr", 1e-3)
        self.criterion       = nn.MSELoss()

        # 用户-物品交互 & 社交图（已在你的数据集里构好）
        self.interaction_matrix = dataset.interaction_matrix.astype(np.float32)
        self.net_matrix         = dataset.net_matrix

        # 给社交图加自环（常见做法）
        Iu = sp.eye(self.n_users, dtype=np.float32, format="coo")
        self.net_matrix = sp.coo_matrix(self.net_matrix + Iu)

        # 预计算归一化卷积矩阵（延迟GPU分配以节省初始化内存）
        # 注意：_get_conv已经在内部调用.to(device)，无需重复
        self.A_ui = self._get_conv(self.interaction_matrix)           # (U x I)
        self.A_iu = self._get_conv(self.interaction_matrix.transpose()) # (I x U)
        self.A_ss = self._get_conv(self.net_matrix)                   # (U x U)

        # Embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size).to(self.device)

        # 融合层：把 UI 与 Social 两路聚合的用户表示拼接再压回 d
        self.user_fuse = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)

        # 可选的层间融合（残差 + 线性），更稳
        self.user_res = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.item_res = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)

        # 评分回归就用点积（与你原 predict 一致）；也可以在此处加一层 MLP，但为保持对比公平先不加

    # ------- 训练/评估共用接口 -------
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        U, I = self.forward()              # (U_all, I_all)
        u = U[user]                        # (B, d)
        i = I[item]                        # (B, d)
        scores = (u * i).sum(dim=1)        # 点积
        return scores

    def forward(self):
        # 初始嵌入
        u0 = self.user_embedding.weight.to(self.device)  # (U, d)
        i0 = self.item_embedding.weight.to(self.device)  # (I, d)

        u = u0
        i = i0

        for _ in range(self.gnn_layer_size):
            # 物品从“用户->物品”聚合
            i_from_users = torch.sparse.mm(self.A_iu, u)         # (I, d)

            # 用户两路：从物品聚合 + 从社交图聚合
            u_from_items  = torch.sparse.mm(self.A_ui, i)        # (U, d)
            u_from_social = torch.sparse.mm(self.A_ss, u)        # (U, d)

            # 融合用户两路
            u_new = self.user_fuse(torch.cat([u_from_items, u_from_social], dim=1))  # (U, d)
            i_new = i_from_users                                                    # (I, d)

            # 残差融合（更稳）
            u = self.user_res(torch.cat([u_new, u], dim=1))
            i = self.item_res(torch.cat([i_new, i], dim=1))

        return u, i

    # ------- 工具函数（与 ESSRec 的卷积构建保持一致） -------
    def _get_conv(self, mat_coo):
        D = self._inv_deg(mat_coo)
        L = D * mat_coo   # D^{-1}A
        L = sp.coo_matrix(L)
        return self._to_torch_sparse(L)

    def _inv_deg(self, mat):
        # 行归一化：按“出边数”做平均
        deg = (mat != 0).sum(axis=1)
        diag = np.array(deg).flatten() + 1e-7
        diag = np.power(diag, -1)
        return sp.coo_matrix(sp.diags(diag), dtype=np.float32)

    def _to_torch_sparse(self, spm):
        row = np.array(spm.row)
        col = np.array(spm.col)
        idx = np.vstack([row, col])
        data = np.array(spm.data, dtype=np.float32)
        i = torch.LongTensor(idx)
        v = torch.FloatTensor(data)
        return torch.sparse_coo_tensor(i, v, spm.shape, device=self.device)


class RecSSN(nn.Module):

    def __init__(self, dataset, args):
        super(RecSSN, self).__init__()

        # fields
        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING  = "rating:float"

        # sizes
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # matrices
        self.interaction_matrix = dataset.interaction_matrix.astype(np.float32)
        self.net_matrix = dataset.net_matrix

        # device & hparams
        self.device = args.device
        self.embedding_size = getattr(args, "embedding_size", 64)
        self.criterion = nn.MSELoss(reduction='sum')
        self.reg_weight = 0.01
        self.sign_loss_weight = 0.01

        # embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size).to(self.device)

        # simple heads (kept for parity with original code structure)
        self.sim_layer = nn.Linear(self.embedding_size * 2, 2).to(self.device)
        self.concat_user_item_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.concat_PON_layer = nn.Linear(self.embedding_size * 3, self.embedding_size).to(self.device)
        self.concat_PN_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)

        # precompute global weight via exponential rank on transposed social graph
        self.global_weight = self._exponential_rank()

    # ====== losses ======
    def reg_loss(self):
        l2_reg = torch.pow(self.user_embedding.weight.norm(), 2) + torch.pow(self.item_embedding.weight.norm(), 2)
        return self.reg_weight * l2_reg

    def rating_loss(self, interaction):
        rating = interaction[self.RATING]
        user = interaction[self.USER_ID]
        weight = self.global_weight[user]
        weight = torch.mul(weight, rating)
        scores = self.predict_norm(interaction)  # in [0,1]
        rating_norm = rating / torch.mul(torch.ones_like(scores), 5)
        mse_each = F.mse_loss(rating_norm, scores, reduction='none')
        weighted = torch.mul(mse_each, weight)
        return torch.sum(weighted)

    def sign_loss(self, interaction):
        user = interaction[self.USER_ID]
        P, N = self._get_signed_matrices(self.net_matrix)
        PCM = self._get_conv(P)
        NCM = self._get_conv(N)
        tmp_criterion = nn.L1Loss(reduction='none')
        UP = torch.sparse.mm(PCM, self.user_embedding.weight)
        UN = torch.sparse.mm(NCM, self.user_embedding.weight)
        p_gap = tmp_criterion(UP, self.user_embedding.weight)[user]
        n_gap = tmp_criterion(UN, self.user_embedding.weight)[user]
        p_gap = torch.pow(p_gap, 2)
        n_gap = torch.pow(n_gap, 2)
        loss = p_gap - n_gap
        loss = torch.relu(loss)
        return torch.sum(loss) * self.sign_loss_weight

    def calculate_loss(self, interaction):
        rating_l = self.rating_loss(interaction)
        sign_l = self.sign_loss(interaction)
        reg_l = self.reg_loss()
        return rating_l + sign_l + reg_l

    # ====== inference ======
    def predict_norm(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all, item_all = self.forward()
        u = user_all[user]
        i = item_all[item]
        scores = torch.sigmoid(torch.mul(u, i).sum(dim=1))
        return scores

    def predict(self, interaction):
        scores = self.predict_norm(interaction)
        return torch.mul(scores, 5)

    def forward(self):
        return self.user_embedding.weight, self.item_embedding.weight

    # ====== graph utils ======
    def _inv_deg(self, matrix):
        # use unweighted degree by default (align to original code variant)
        sumArr = (matrix > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        return sp.coo_matrix(sp.diags(diag), dtype=np.float32)

    def _to_torch_sparse(self, spm):
        row = np.array(spm.row)
        col = np.array(spm.col)
        idx = np.vstack([row, col])
        data = np.array(spm.data, dtype=np.float32)
        i = torch.LongTensor(idx)
        v = torch.FloatTensor(data)
        return torch.sparse_coo_tensor(i, v, spm.shape, device=self.device)

    def _get_conv(self, matrix):
        D = self._inv_deg(matrix)
        L = D * matrix
        L = sp.coo_matrix(L)
        return self._to_torch_sparse(L)

    def _get_signed_matrices(self, sparse_matrix):
        data = sparse_matrix.data
        trust_index = np.argwhere(data == 1).squeeze(axis=1)
        distrust_index = np.argwhere(data == -1).squeeze(axis=1)
        P = self._split_matrix(sparse_matrix, trust_index)
        N = self._split_matrix(sparse_matrix, distrust_index)
        return P, N

    def _split_matrix(self, sparse_matrix, index):
        if len(index) == 0:
            return sp.coo_matrix((self.n_users, self.n_users), dtype=np.float32)
        row = sparse_matrix.row[index]
        col = sparse_matrix.col[index]
        data = np.ones(len(index))
        return sp.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_users))

    def _exponential_rank(self):
        A = self.net_matrix.T
        p = np.ones(self.n_users) / self.n_users
        steps = 1000
        u = 5
        for _ in range(steps):
            # matrix-vector exp(u * A * p) approximately via numpy broadcasting on dense may be heavy;
            # Here treat A as sparse and perform A * p first.
            Ap = A.dot(p)
            p = np.exp(u * Ap)
            s = p.sum()
            if s == 0:
                break
            p = p / s
        r = np.argsort(p) + 1
        w = 1 / np.log1p(r)
        return torch.tensor(w, dtype=torch.float32, device=self.device)

class SocialMF(nn.Module):

    def __init__(self, dataset, args):
        super(SocialMF, self).__init__()
        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING  = "rating:float"

        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.interaction_matrix = dataset.interaction_matrix.astype(np.float32)
        self.net_matrix = dataset.net_matrix

        self.device = args.device
        self.embedding_size = getattr(args, "embedding_size", 64)
        self.criterion = nn.MSELoss(reduction='sum')
        self.reg_weight = 1e-3
        self.trust_weight = 0.0

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size).to(self.device)

        # positive neighbor aggregation matrix
        P, _ = self._get_signed_matrices(self.net_matrix)
        # cache propagation operator; compute UP per-iteration without tracking graph
        self.PCM = self._get_conv(P)

    def reg_loss(self):
        l2_reg = torch.pow(self.user_embedding.weight.norm(), 2) + torch.pow(self.item_embedding.weight.norm(), 2)
        return self.reg_weight * l2_reg

    def calculate_loss(self, interaction):
        rating = interaction[self.RATING].to(self.device)
        scores = self.predict_norm(interaction)  # [0,1]
        rating_norm = rating / torch.mul(torch.ones_like(scores), 5)
        rating_loss = self.criterion(rating_norm, scores)
        # compute UP detached to avoid building a backward graph across iterations
        with torch.no_grad():
            UP = torch.sparse.mm(self.PCM, self.user_embedding.weight.detach())
        trust_loss = self.criterion(self.user_embedding.weight, UP) * self.trust_weight
        reg = self.reg_loss()
        return (rating_loss + trust_loss + reg) * 0.5

    def predict(self, interaction):
        scores = self.predict_norm(interaction)
        return torch.mul(scores, 5)

    def predict_norm(self, interaction):
        user = interaction[self.USER_ID].to(self.device)
        item = interaction[self.ITEM_ID].to(self.device)
        u_all, i_all = self.forward()
        u = u_all[user]
        i = i_all[item]
        scores = torch.sigmoid(torch.mul(u, i).sum(dim=1))
        return scores

    def forward(self):
        return self.user_embedding.weight, self.item_embedding.weight

    # utils
    def _inv_deg(self, matrix):
        sumArr = (matrix > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        return sp.coo_matrix(sp.diags(diag), dtype=np.float32)

    def _to_torch_sparse(self, spm):
        row = np.array(spm.row)
        col = np.array(spm.col)
        idx = np.vstack([row, col])
        data = np.array(spm.data, dtype=np.float32)
        i = torch.LongTensor(idx)
        v = torch.FloatTensor(data)
        return torch.sparse_coo_tensor(i, v, spm.shape, device=self.device)

    def _get_conv(self, matrix):
        D = self._inv_deg(matrix)
        L = D * matrix
        L = sp.coo_matrix(L)
        return self._to_torch_sparse(L)

    def _get_signed_matrices(self, sparse_matrix):
        data = sparse_matrix.data
        trust_index = np.argwhere(data == 1).squeeze(axis=1)
        distrust_index = np.argwhere(data == -1).squeeze(axis=1)
        P = self._split_matrix(sparse_matrix, trust_index)
        N = self._split_matrix(sparse_matrix, distrust_index)
        return P, N

    def _split_matrix(self, sparse_matrix, index):
        if len(index) == 0:
            return sp.coo_matrix((self.n_users, self.n_users), dtype=np.float32)
        row = sparse_matrix.row[index]
        col = sparse_matrix.col[index]
        data = np.ones(len(index))
        return sp.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_users))

# class TrustMF(nn.Module):

class TDRec(nn.Module):

    def __init__(self, dataset, args):
        super(TDRec, self).__init__()
        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING  = "rating:float"

        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.interaction_matrix = dataset.interaction_matrix.astype(np.float32)
        self.net_matrix = dataset.net_matrix

        self.device = args.device
        self.embedding_size = getattr(args, "embedding_size", 64)
        self.criterion = nn.MSELoss(reduction='sum')
        self.reg_weight = 1e-3
        self.distrust_weight = 0.0
        self.trust_weight = 1.0
        self.beta = 0.4

        # split trust/distrust
        self.trust_matrix, self.distrust_matrix = self._get_signed_matrices(self.net_matrix)

        # embeddings
        self.P = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.Q = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.V = nn.Embedding(self.n_items, self.embedding_size).to(self.device)

    def reg_loss(self):
        l2_reg = (
            torch.pow(self.P.weight.norm(), 2)
            + torch.pow(self.Q.weight.norm(), 2)
            + torch.pow(self.V.weight.norm(), 2)
        )
        return self.reg_weight * l2_reg

    def trust_loss(self):
        if self.trust_matrix.nnz == 0:
            return torch.tensor(0.0, device=self.device)
        row = torch.LongTensor(self.trust_matrix.row).to(self.device)
        col = torch.LongTensor(self.trust_matrix.col).to(self.device)
        data = torch.FloatTensor(self.trust_matrix.data).to(self.device)
        tmp_P = self.P.weight[row]
        tmp_Q = self.Q.weight[col]
        pred = torch.sigmoid(torch.mul(tmp_P, tmp_Q).sum(dim=1))
        loss = self.criterion(pred, data)
        return loss * self.trust_weight

    def distrust_loss(self):
        if self.distrust_matrix.nnz == 0:
            return torch.tensor(0.0, device=self.device)
        DCM = self._get_conv(self.distrust_matrix)
        PN = torch.sparse.mm(DCM, self.P.weight)
        QN = torch.sparse.mm(DCM, self.Q.weight)
        p_gap = self.criterion(PN, self.P.weight)
        q_gap = self.criterion(QN, self.Q.weight)
        loss = (p_gap + q_gap) / max(1, len(self.P.weight))
        return loss * self.distrust_weight

    def rating_loss(self, interaction):
        rating = interaction[self.RATING].to(self.device)
        scores = self.predict_norm(interaction)
        rating_norm = rating / torch.mul(torch.ones_like(scores), 5)
        return self.criterion(rating_norm, scores)

    def calculate_loss(self, interaction):
        return self.rating_loss(interaction) + self.trust_loss() - self.distrust_loss() + self.reg_loss()

    def predict(self, interaction):
        scores = self.predict_norm(interaction)
        return torch.mul(scores, 5)

    def predict_norm(self, interaction):
        user = interaction[self.USER_ID].to(self.device)
        item = interaction[self.ITEM_ID].to(self.device)
        P, Q, V = self.forward()
        p = P[user]
        q = Q[user]
        v = V[item]
        scores = torch.sigmoid(torch.mul(torch.mul(p, v).sum(dim=1), self.beta) + torch.mul(
            torch.mul(q, v).sum(dim=1), 1 - self.beta))
        return scores

    def forward(self):
        return self.P.weight, self.Q.weight, self.V.weight

    # utils
    def _inv_deg(self, matrix):
        sumArr = (matrix > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        return sp.coo_matrix(sp.diags(diag), dtype=np.float32)

    def _to_torch_sparse(self, spm):
        row = np.array(spm.row)
        col = np.array(spm.col)
        idx = np.vstack([row, col])
        data = np.array(spm.data, dtype=np.float32)
        i = torch.LongTensor(idx)
        v = torch.FloatTensor(data)
        return torch.sparse_coo_tensor(i, v, spm.shape, device=self.device)

    def _get_conv(self, matrix):
        D = self._inv_deg(matrix)
        L = D * matrix
        L = sp.coo_matrix(L)
        return self._to_torch_sparse(L)

    def _get_signed_matrices(self, sparse_matrix):
        data = sparse_matrix.data
        trust_index = np.argwhere(data == 1).squeeze(axis=1)
        distrust_index = np.argwhere(data == -1).squeeze(axis=1)
        P = self._split_matrix(sparse_matrix, trust_index)
        N = self._split_matrix(sparse_matrix, distrust_index)
        return P, N

    def _split_matrix(self, sparse_matrix, index):
        if len(index) == 0:
            return sp.coo_matrix((self.n_users, self.n_users), dtype=np.float32)
        row = sparse_matrix.row[index]
        col = sparse_matrix.col[index]
        data = np.ones(len(index))
        return sp.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_users))


# Alias for paper naming compatibility
SSC_Loop = ESSRec