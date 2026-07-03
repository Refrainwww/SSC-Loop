"""
排序评估指标模块
实现 Recall@K, NDCG@K, HR@K 等指标
"""
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


def _safe_get_embeddings(model):
    """
    安全地从任意模型的 forward() 获取 user_emb 和 item_emb。
    兼容以下所有返回格式：
    - (users, items)                    → LightGCN/SGL/XSimGCL/GraphRec/RecSSN/ESSRec
    - (P, Q, V)                         → TDRec
    - users/items as single tensor       → 某些模型
    - model.user_embedding + item_embedding → 回退方案
    """
    out = model.forward()

    # 情况1: 2元素元组（标准格式）
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]

    # 情况2: 3元素元组（如 TDRec: P, Q, V）
    if isinstance(out, tuple) and len(out) == 3:
        return out[0], out[2]  # P (user), V (item)

    # 情况3: 已经是张量对
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        return out[0], out[1]

    raise ValueError(f"无法解析 model.forward() 返回值: type={type(out)}, "
                     f"value={str(out)[:200]}")


def evaluate_ranking(model, test_data, args, K_list=[5, 10, 20], rating_threshold=4.0):
    """
    评估 Top-K 排序指标（修正版）

    使用候选集评估方式：
    - 每个正样本搭配99个随机负样本，在100个候选中排序
    - 这比全物品排序更贴近推荐系统的真实场景

    Args:
        model: 训练好的推荐模型
        test_data: 测试数据 [(user, item, rating), ...]
        args: 参数配置
        K_list: 需要评估的K值列表
        rating_threshold: 将评分>=该值的物品视为正样本

    Returns:
        dict: 包含 Recall@K, NDCG@K, HR@K 的字典
    """
    model.eval()

    # 按用户组织测试数据 - 只保留高分物品作为正样本
    user_pos_items = defaultdict(list)
    for u, i, r in test_data:
        if r >= rating_threshold:
            user_pos_items[u].append(i)

    # 过滤掉没有正样本的用户
    user_pos_items = {u: items for u, items in user_pos_items.items() if len(items) > 0}

    if len(user_pos_items) == 0:
        print("Warning: No positive samples in test data!")
        results = {}
        for k in K_list:
            results[f'Recall@{k}'] = 0.0
            results[f'NDCG@{k}'] = 0.0
            results[f'HR@{k}'] = 0.0
            results[f'Precision@{k}'] = 0.0
        return results

    all_metrics = {f'Recall@{k}': [] for k in K_list}
    all_metrics.update({f'NDCG@{k}': [] for k in K_list})
    all_metrics.update({f'HR@{k}': [] for k in K_list})
    all_metrics.update({f'Precision@{k}': [] for k in K_list})

    max_k = max(K_list)

    # 判断是否使用候选集评估
    # 推荐任务（interaction_matrix + n_items != n_users）→ 候选集评估
    # 符号链接任务（n_items == n_users）             → 全排序评估
    use_candidate = (
        hasattr(model, 'interaction_matrix') and
        hasattr(model, 'n_items') and
        hasattr(model, 'n_users') and
        model.n_items != model.n_users
    )

    if use_candidate:
        train_interacted = defaultdict(set)
        inter_csr = model.interaction_matrix.tocsr()
        for u in range(model.n_users):
            start, end = inter_csr.indptr[u], inter_csr.indptr[u + 1]
            train_interacted[u] = set(inter_csr.indices[start:end])

        with torch.no_grad():
            user_emb, item_emb = _safe_get_embeddings(model)

            for user, pos_items in tqdm(list(user_pos_items.items()), desc="Evaluating Ranking", leave=False):
                u_emb = user_emb[user].unsqueeze(0)

                for pos_item in pos_items:
                    neg_items = []
                    attempts = 0
                    while len(neg_items) < 99 and attempts < 1000:
                        cand = np.random.randint(0, model.n_items)
                        if cand != pos_item and cand not in train_interacted[user]:
                            neg_items.append(cand)
                        attempts += 1
                    while len(neg_items) < 99:
                        cand = np.random.randint(0, model.n_items)
                        if cand != pos_item:
                            neg_items.append(cand)

                    candidate_items = [pos_item] + neg_items
                    cand_tensor = torch.tensor(candidate_items, device=user_emb.device)
                    scores = torch.matmul(u_emb, item_emb[cand_tensor].T).squeeze()
                    _, ranked_indices = torch.topk(scores, min(max_k, len(candidate_items)))
                    ranked_items = np.array(candidate_items)[ranked_indices.cpu().numpy()]

                    pos_set = {pos_item}
                    hits = sum(1 for item in ranked_items[:max_k] if item in pos_set)

                    for k in K_list:
                        recall = hits / len(pos_set)
                        all_metrics[f'Recall@{k}'].append(recall)
                        precision = hits / k
                        all_metrics[f'Precision@{k}'].append(precision)
                        hr = 1.0 if hits > 0 else 0.0
                        all_metrics[f'HR@{k}'].append(hr)
                        dcg = sum([1.0 / np.log2(idx + 2) for idx, item in enumerate(ranked_items[:k])
                                  if item in pos_set])
                        idcg = 1.0 / np.log2(2)
                        ndcg = dcg / idcg if idcg > 0 else 0.0
                        all_metrics[f'NDCG@{k}'].append(ndcg)

    else:
        # 全物品排序：用于符号链接预测（n_items == n_users）或无 interaction_matrix 的模型
        try:
            user_emb, item_emb = _safe_get_embeddings(model)
        except Exception as e:
            print(f"Warning: 无法从 forward() 获取嵌入，尝试直接访问 embedding: {e}")
            try:
                user_emb = model.user_embedding.weight
                item_emb = model.item_embedding.weight
            except Exception:
                raise RuntimeError(f"无法获取嵌入进行排序评估: {e}")

        n_items = item_emb.shape[0]

        with torch.no_grad():
            for user, pos_items in tqdm(list(user_pos_items.items()), desc="Evaluating Ranking", leave=False):
                u_emb = user_emb[user].unsqueeze(0)
                scores = torch.matmul(u_emb, item_emb.T).squeeze()
                _, topk_items = torch.topk(scores, min(max_k, n_items))
                topk_items_np = topk_items.cpu().numpy()
                pos_set = set(pos_items)

                for k in K_list:
                    pred_k = set(topk_items_np[:k])
                    hits = len(pred_k & pos_set)
                    recall = hits / len(pos_items)
                    all_metrics[f'Recall@{k}'].append(recall)
                    precision = hits / k
                    all_metrics[f'Precision@{k}'].append(precision)
                    hr = 1.0 if hits > 0 else 0.0
                    all_metrics[f'HR@{k}'].append(hr)
                    dcg = sum([1.0 / np.log2(idx + 2) for idx, item in enumerate(topk_items_np[:k])
                              if item in pos_set])
                    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(pos_items), k))])
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    all_metrics[f'NDCG@{k}'].append(ndcg)

    results = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in all_metrics.items()}
    return results


def evaluate_cold_start(model, dataset, test_data, args, cold_threshold=5, K_list=[5, 10, 20]):
    """
    分别评估冷启动用户和热启动用户的性能（修正版）
    
    Args:
        model: 训练好的模型
        dataset: 数据集对象
        test_data: 测试数据
        args: 参数配置
        cold_threshold: 交互数小于该值的用户被视为冷启动用户
        K_list: 评估的K值列表
    
    Returns:
        tuple: (cold_metrics, warm_metrics, cold_users, warm_users)
    """
    # 统计每个用户在训练集中的交互数
    user_inter_count = defaultdict(int)
    for u, i, r in dataset.train_data:
        user_inter_count[u] += 1
    
    # 分组
    cold_users = set([u for u, cnt in user_inter_count.items() if cnt < cold_threshold])
    warm_users = set([u for u, cnt in user_inter_count.items() if cnt >= cold_threshold])
    
    # 分离测试数据
    cold_test = [(u, i, r) for u, i, r in test_data if u in cold_users]
    warm_test = [(u, i, r) for u, i, r in test_data if u in warm_users]
    
    print(f"\n{'='*60}")
    print(f"Cold-Start Evaluation (threshold={cold_threshold})")
    print(f"{'='*60}")
    print(f"Cold-start users: {len(cold_users)} ({len(cold_test)} test samples)")
    print(f"Warm users: {len(warm_users)} ({len(warm_test)} test samples)")
    
    # 分别评估
    cold_metrics = {}
    warm_metrics = {}
    
    if len(cold_test) > 0:
        print("\n[Cold-Start Users]")
        cold_metrics = evaluate_ranking(model, cold_test, args, K_list)
        for k in K_list:
            print(f"  Recall@{k}: {cold_metrics.get(f'Recall@{k}', 0.0):.4f}, "
                  f"NDCG@{k}: {cold_metrics.get(f'NDCG@{k}', 0.0):.4f}, "
                  f"HR@{k}: {cold_metrics.get(f'HR@{k}', 0.0):.4f}")
    else:
        # 冷启动测试集为空时返回全0（而不是报错）
        print("\n[Cold-Start Users] No cold-start test samples found.")
        cold_metrics = {f'Recall@{k}': 0.0 for k in K_list}
        cold_metrics.update({f'NDCG@{k}': 0.0 for k in K_list})
        cold_metrics.update({f'HR@{k}': 0.0 for k in K_list})
    
    if len(warm_test) > 0:
        print("\n[Warm Users]")
        warm_metrics = evaluate_ranking(model, warm_test, args, K_list)
        for k in K_list:
            print(f"  Recall@{k}: {warm_metrics.get(f'Recall@{k}', 0.0):.4f}, "
                  f"NDCG@{k}: {warm_metrics.get(f'NDCG@{k}', 0.0):.4f}, "
                  f"HR@{k}: {warm_metrics.get(f'HR@{k}', 0.0):.4f}")
    else:
        print("\n[Warm Users] No warm test samples found.")
        warm_metrics = {f'Recall@{k}': 0.0 for k in K_list}
        warm_metrics.update({f'NDCG@{k}': 0.0 for k in K_list})
        warm_metrics.update({f'HR@{k}': 0.0 for k in K_list})
    
    return cold_metrics, warm_metrics, cold_users, warm_users


def evaluate_by_interaction_groups(model, dataset, test_data, args, K_list=[10]):
    """
    按用户交互数分组评估
    
    分组：
    - Extreme Cold [0, 3): 极度冷启动
    - Cold [3, 10): 冷启动
    - Medium [10, 50): 中等活跃
    - Active [50, +∞): 高活跃用户
    """
    # 统计用户交互数
    user_inter_count = defaultdict(int)
    for u, i, r in dataset.train_data:
        user_inter_count[u] += 1
    
    # 定义分组
    groups = {
        'Extreme Cold [0,3)': (0, 3),
        'Cold [3,10)': (3, 10),
        'Medium [10,50)': (10, 50),
        'Active [50,+∞)': (50, float('inf'))
    }
    
    print(f"\n{'='*60}")
    print(f"Evaluation by Interaction Groups")
    print(f"{'='*60}")
    
    all_group_results = {}
    
    for group_name, (low, high) in groups.items():
        # 找到属于该组的用户
        group_users = set([u for u, cnt in user_inter_count.items()
                          if low <= cnt < high])
        
        if len(group_users) == 0:
            print(f"\n{group_name}: No users in this group")
            continue
        
        # 筛选测试数据（使用传入的 test_data 而不是 dataset.test_data）
        group_test = [(u, i, r) for u, i, r in test_data
                     if u in group_users]
        
        if len(group_test) == 0:
            print(f"\n{group_name}: No test samples")
            continue
        
        print(f"\n{group_name}: {len(group_users)} users, {len(group_test)} test samples")
        
        # 评估
        metrics = evaluate_ranking(model, group_test, args, K_list)
        all_group_results[group_name] = metrics
        
        # 打印结果
        for k in K_list:
            print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f}, "
                  f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}, "
                  f"HR@{k}: {metrics[f'HR@{k}']:.4f}")
    
    return all_group_results


def evaluate_rating_prediction(model, test_loader, args):
    """
    评估评分预测指标 (RMSE, MAE)
    """
    model.eval()
    preds = []
    gts = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            rating_gt = batch_data[model.RATING].to(args.device)
            scores = model.predict(batch_data)
            preds.append(scores.cpu())
            gts.append(rating_gt.cpu())
    
    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    
    rmse = torch.sqrt(torch.mean((preds - gts) ** 2))
    mae = torch.mean(torch.abs(preds - gts))
    
    return {
        'RMSE': rmse.item(),
        'MAE': mae.item()
    }
