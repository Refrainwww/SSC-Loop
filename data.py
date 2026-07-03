import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm
import random

###########################################
# 1. 定义 InteractionDataset - 用于 DataLoader
###########################################
class InteractionDataset(Dataset):
    """
    将 (user, item, rating) 列表封装成 PyTorch Dataset 的简易示例。
    """
    def __init__(self, data_list, user_key, item_key, rating_key):
        # data_list 里的每条数据形如 (u, i, r)
        # user_key, item_key, rating_key 分别对应 ESSRec 里的 self.USER_ID, self.ITEM_ID, self.RATING
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
            for k,v in b.items():
                result[k].append(v)
        # 将 user_id, item_id 转成 long; rating 转成 float
        out = {}
        for k, vals in result.items():
            if "rating" in k:
                out[k] = torch.tensor(vals, dtype=torch.float32)
            else:
                out[k] = torch.tensor(vals, dtype=torch.long)
        return out



class EpinionsDataset:
    """
    Epinions 数据集加载和预处理类，适配 ESSRec 模型的要求。
    """

    def __init__(self, path_inter="epinions.inter", path_net="epinions.net",
                 train_ratio=0.7, val_ratio=0.1, seed: int = 42):
        # 用户和物品的 ID 映射表
        self.user_map = {}
        self.item_map = {}
        self.user_cnt = 0
        self.item_cnt = 0

        # 用户-物品评分数据：格式 {user_id: [(item_id, rating), ...]}
        self.user_item_dict = defaultdict(list)

        # 用户-用户符号社交网络：格式 {(u1, u2): sign}
        self.user_user_sign = {}

        # 加载交互数据和社交网络数据
        self._load_inter(path_inter)
        self._load_net(path_net)

        # 构建评分矩阵 (interaction_matrix) 和符号网络矩阵 (net_matrix)
        self.interaction_matrix = self._build_interaction_matrix()
        self.net_matrix = self._build_net_matrix()

        # 划分训练集、验证集、测试集
        self.seed = seed
        self.train_data, self.val_data, self.test_data = self._train_val_test_split(
            train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )

    def _map_user_id(self, old_u):
        """用户 ID 重映射"""
        if old_u not in self.user_map:
            self.user_map[old_u] = self.user_cnt
            self.user_cnt += 1
        return self.user_map[old_u]

    def _map_item_id(self, old_i):
        """物品 ID 重映射"""
        if old_i not in self.item_map:
            self.item_map[old_i] = self.item_cnt
            self.item_cnt += 1
        return self.item_map[old_i]

    def _load_inter(self, path_inter):
        """
        读取 epinions.inter 文件，构建 user-item 数据字典。
        格式: user_id item_id rating
        """
        print(f"Loading interaction data from {path_inter}...")
        num_lines = sum(1 for _ in open(path_inter, "r"))

        with open(path_inter, "r") as f:
            for line in tqdm(f, total=num_lines, desc="Reading interaction file"):
                # 跳过可能的表头和空行
                if line.startswith("user_id") or line.strip() == "":
                    continue

                # 尝试解析每行数据
                try:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue

                    old_u, old_i, r = parts
                    old_u = int(old_u)
                    old_i = int(old_i)
                    r = float(r)

                    new_u = self._map_user_id(old_u)
                    new_i = self._map_item_id(old_i)
                    self.user_item_dict[new_u].append((new_i, r))

                except ValueError as e:
                    print(f"Skipping invalid line: {line.strip()} - Error: {e}")
                except Exception as e:
                    print(f"Unexpected error when reading line: {line.strip()} - Error: {e}")

    def _load_net(self, path_net):
        """
        读取 epinions.net 文件，构建用户-用户符号网络字典。
        格式: user_id1 user_id2 sign
        """
        print(f"Loading social network data from {path_net}...")
        num_lines = sum(1 for _ in open(path_net, "r"))

        with open(path_net, "r") as f:
            for line in tqdm(f, total=num_lines, desc="Reading social network file"):
                # 跳过可能的表头和空行
                if line.startswith("source_id") or line.strip() == "":
                    continue

                # 尝试解析每行数据
                try:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue

                    old_u1, old_u2, sign = parts
                    old_u1 = int(old_u1)
                    old_u2 = int(old_u2)
                    sign = int(sign)  # 1 或 -1

                    # 映射为连续 ID
                    new_u1 = self._map_user_id(old_u1)
                    new_u2 = self._map_user_id(old_u2)

                    self.user_user_sign[(new_u1, new_u2)] = sign

                except ValueError as e:
                    print(f"Skipping invalid line: {line.strip()} - Error: {e}")
                except Exception as e:
                    print(f"Unexpected error when reading line: {line.strip()} - Error: {e}")


    def _build_interaction_matrix(self):
        """
        构建用户-物品评分矩阵（稀疏格式）。
        行：用户，列：物品，值：评分。
        """
        print("Constructing interaction matrix...")
        rows, cols, data = [], [], []
        for u, interactions in tqdm(self.user_item_dict.items(), desc="Building interaction matrix"):
            for i, r in interactions:
                rows.append(u)
                cols.append(i)
                data.append(r)
        # 构建稀疏矩阵
        interaction_matrix = sp.coo_matrix((data, (rows, cols)),
                                           shape=(self.user_cnt, self.item_cnt),
                                           dtype=np.float32)
        return interaction_matrix

    def _build_net_matrix(self):
        """
        构建用户-用户符号社交网络的邻接矩阵。
        行列：用户，值：±1。
        """
        print("Constructing net matrix...")
        rows, cols, data = [], [], []
        for (u1, u2), sign in tqdm(self.user_user_sign.items(), desc="Building net matrix"):
            rows.append(u1)
            cols.append(u2)
            data.append(sign)
        # 构建用户-用户网络矩阵
        net_matrix = sp.coo_matrix((data, (rows, cols)),
                                   shape=(self.user_cnt, self.user_cnt),
                                   dtype=np.float32)
        return net_matrix

    def _train_val_test_split(self, train_ratio=0.7, val_ratio=0.1, seed: int = 42):
        """
        将用户-物品评分数据集划分为训练集、验证集和测试集。
        - 先按 train_ratio / (1 - train_ratio) 拆分 train/test
        - 再从 train 中抽出 val_ratio 作为验证集
        """
        print("Splitting dataset into train/val/test sets...")
        all_ratings = []
        for u, interactions in self.user_item_dict.items():
            for i, r in interactions:
                all_ratings.append((u, i, r))

        rng = random.Random(seed)
        rng.shuffle(all_ratings)

        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

        train_end = int(len(all_ratings) * train_ratio)
        val_end = train_end + int(len(all_ratings) * val_ratio)
        train_data = all_ratings[:train_end]
        val_data = all_ratings[train_end:val_end]
        test_data = all_ratings[val_end:]
        return train_data, val_data, test_data

    def update_net_matrix(self, new_net_matrix: sp.coo_matrix):
        """
        更新符号社交网络矩阵（例如 ESA-DA 增强后）。
        """
        if not sp.isspmatrix_coo(new_net_matrix):
            new_net_matrix = new_net_matrix.tocoo()
        self.net_matrix = new_net_matrix

    def num(self, field):
        """
        返回字段对应的计数：用户或物品数。
        """
        if field == "user_id:token":
            return self.user_cnt
        elif field == "item_id:token":
            return self.item_cnt
        else:
            raise ValueError(f"Unknown field: {field}")
