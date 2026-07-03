import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm
import random

###########################################
# 1. SlashdotDataset - 将信任关系转换为"用户推荐用户"的评分
###########################################
class SlashdotDataset:
    """
    Slashdot 数据集加载类 - 将符号社交网络转换为"用户推荐用户"场景
    
    转换策略：
    - 用户-用户信任关系 (src, tgt, sign) → 用户对用户的"信任评分"
    - sign = +1 → rating = 5.0 (完全信任)
    - sign = -1 → rating = 1.0 (不信任)
    
    这样可以将推荐系统模型应用于符号社交网络分析
    """

    def __init__(self, path_net, train_ratio=0.8, rating_pos=5.0, rating_neg=1.0):
        self.user_map = {}
        self.user_cnt = 0
        self.rating_pos = rating_pos  # 信任边的评分
        self.rating_neg = rating_neg  # 不信任边的评分

        self.user_user_sign = {}  # {(src, tgt): sign}
        self.user_item_dict = defaultdict(list)  # 复用字段名: {user: [(target_user, rating)]}

        self._load_net(path_net)
        self.interaction_matrix = self._build_interaction_matrix()
        self.net_matrix = self._build_net_matrix()
        self.train_data, self.test_data = self._train_test_split(ratio=train_ratio)

    def _map_user_id(self, old_u):
        if old_u not in self.user_map:
            self.user_map[old_u] = self.user_cnt
            self.user_cnt += 1
        return self.user_map[old_u]

    def _load_net(self, path_net):
        """读取 Slashdot 网络文件"""
        print(f"Loading Slashdot network from {path_net}...")
        num_lines = sum(1 for _ in open(path_net, "r"))
        
        with open(path_net, "r") as f:
            for line in tqdm(f, total=num_lines, desc="Reading network file"):
                if line.strip() == "":
                    continue
                try:
                    parts = line.strip().split(',')
                    if len(parts) != 3:
                        continue
                    
                    src, tgt, sign = int(parts[0]), int(parts[1]), int(parts[2])
                    src_new = self._map_user_id(src)
                    tgt_new = self._map_user_id(tgt)
                    
                    self.user_user_sign[(src_new, tgt_new)] = sign
                    # 转换: sign -> rating
                    rating = self.rating_pos if sign > 0 else self.rating_neg
                    self.user_item_dict[src_new].append((tgt_new, rating))
                    
                except ValueError as e:
                    print(f"Skipping invalid line: {line.strip()} - Error: {e}")

    def _build_interaction_matrix(self):
        """构建评分矩阵（用户->目标用户的信任评分）"""
        print("Constructing interaction matrix...")
        rows, cols, data = [], [], []
        for u, interactions in tqdm(self.user_item_dict.items(), desc="Building matrix"):
            for i, r in interactions:
                rows.append(u)
                cols.append(i)
                data.append(r)
        interaction_matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self.user_cnt, self.user_cnt),  # Slashdot: 方阵，用户数 x 用户数
            dtype=np.float32
        )
        return interaction_matrix

    def _build_net_matrix(self):
        """构建符号网络矩阵（用于ESSRec的符号图卷积）"""
        print("Constructing signed network matrix...")
        rows, cols, data = [], [], []
        for (u1, u2), sign in tqdm(self.user_user_sign.items(), desc="Building net matrix"):
            rows.append(u1)
            cols.append(u2)
            data.append(sign)
        # 加对角线保持自连接
        diag = np.ones(self.user_cnt)
        net_matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self.user_cnt, self.user_cnt),
            dtype=np.float32
        )
        I = sp.coo_matrix(sp.diags(diag), dtype=np.float32)
        net_matrix = net_matrix + I
        return sp.coo_matrix(net_matrix)

    def _train_test_split(self, ratio=0.8):
        """划分训练集和测试集"""
        print("Splitting dataset into train and test sets...")
        all_ratings = []
        for u, interactions in self.user_item_dict.items():
            for i, r in interactions:
                all_ratings.append((u, i, r))
        random.shuffle(all_ratings)

        split_idx = int(len(all_ratings) * ratio)
        train_data = all_ratings[:split_idx]
        test_data = all_ratings[split_idx:]
        return train_data, test_data

    def num(self, field):
        if field == "user_id:token":
            return self.user_cnt
        elif field == "item_id:token":
            return self.user_cnt  # Slashdot: item 就是 user
        else:
            raise ValueError(f"Unknown field: {field}")


###########################################
# 2. InteractionDataset - 保持与 ESSRec 兼容
###########################################
class InteractionDataset(Dataset):
    def __init__(self, data_list, user_key, item_key, rating_key):
        self.records = []
        self.user_key = user_key
        self.item_key = item_key
        self.rating_key = rating_key

        for (u, i, r) in data_list:
            rec = {self.user_key: u, self.item_key: i, self.rating_key: r}
            self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    @staticmethod
    def collate_fn(batch):
        result = defaultdict(list)
        for b in batch:
            for k, v in b.items():
                result[k].append(v)
        out = {}
        for k, vals in result.items():
            if "rating" in k:
                out[k] = torch.tensor(vals, dtype=torch.float32)
            else:
                out[k] = torch.tensor(vals, dtype=torch.long)
        return out
