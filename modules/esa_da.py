import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class ESADAConfig:
    """Configuration for ESA-DA structural consistency refinement."""
    delete_ratio: float = 0.05
    add_ratio: float = 0.05
    knn_k: int = 200
    top_k_candidate: int = 20000
    d_pos_max: int = 50
    d_neg_max: int = 50
    delta_min: int = 1
    balance_guard: bool = True
    tau: float = 1.0


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return embeddings / norm


def _build_knn(embeddings: torch.Tensor, knn_k: int, device: torch.device) -> np.ndarray:
    """
    Return knn indices for each user. Falls back to torch top-k when faiss is unavailable.
    """
    try:
        import faiss  # type: ignore
        use_faiss = True
    except Exception:
        use_faiss = False

    emb = embeddings.detach().cpu().float().numpy()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12

    if use_faiss:
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        _, knn = index.search(emb, knn_k + 1)
        return knn

    # torch-based fallback (works on CPU/GPU)
    emb_t = torch.from_numpy(emb).to(device)
    n_users = emb_t.size(0)
    knn = np.zeros((n_users, knn_k + 1), dtype=np.int64)
    block = max(1024, min(8192, n_users))
    for s in tqdm(range(0, n_users, block), desc="knn blocks", leave=False):
        e = min(n_users, s + block)
        sim = torch.matmul(emb_t[s:e], emb_t.T)
        k = min(knn_k + 1, sim.size(1))
        _, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)
        knn[s:e] = idx.detach().cpu().numpy()
    return knn


def _unique_undirected_edges(rows: np.ndarray, cols: np.ndarray, signs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]
    signs = signs[mask]

    u_min = np.minimum(rows, cols)
    v_max = np.maximum(rows, cols)
    uv = np.stack([u_min, v_max], axis=1)
    uniq_uv, uniq_idx = np.unique(uv, axis=0, return_index=True)
    uniq_signs = signs[uniq_idx]
    return uniq_uv[:, 0], uniq_uv[:, 1], uniq_signs


def _build_degree_counts(n_users: int, u: np.ndarray, v: np.ndarray, signs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_deg = np.zeros(n_users, dtype=np.int32)
    neg_deg = np.zeros(n_users, dtype=np.int32)
    total_deg = np.zeros(n_users, dtype=np.int32)
    for uu, vv, ss in zip(u, v, signs):
        if ss > 0:
            pos_deg[uu] += 1
            pos_deg[vv] += 1
        else:
            neg_deg[uu] += 1
            neg_deg[vv] += 1
        total_deg[uu] += 1
        total_deg[vv] += 1
    return pos_deg, neg_deg, total_deg


def _edge_balance_guard(
    u: int,
    v: int,
    sign: int,
    abs_csr: sp.csr_matrix,
    abs_csc: sp.csc_matrix,
    signed_csr: sp.csr_matrix,
    signed_csc: sp.csc_matrix,
) -> bool:
    """Check structural balance guard using sparse two-hop counts."""
    m_abs = float(abs_csr[u].dot(abs_csc[:, v])[0, 0])
    m_signed = float(signed_csr[u].dot(signed_csc[:, v])[0, 0])
    delta_neg = m_abs - sign * m_signed
    return delta_neg <= 0


def refine_signed_graph(
    net_matrix: sp.coo_matrix,
    user_embeddings: torch.Tensor,
    config: ESADAConfig,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Tuple[sp.coo_matrix, Dict[str, float]]:
    """
    ESA-DA refinement for signed graph. Returns refined net_matrix and stats.
    """
    if device is None:
        device = user_embeddings.device

    if not sp.isspmatrix_coo(net_matrix):
        net_matrix = net_matrix.tocoo()

    rows, cols, signs = _unique_undirected_edges(net_matrix.row, net_matrix.col, net_matrix.data)
    n_users = net_matrix.shape[0]

    if not np.all(np.isin(signs, [1, -1])):
        raise ValueError("ESA-DA expects signed edges with labels in {+1, -1}.")

    num_edges_before = len(rows)
    pos_mask = signs > 0
    neg_mask = signs < 0

    # score existing edges for deletion
    if model is not None:
        uv = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long, device=device)
        with torch.no_grad():
            scored = model.score_edges(uv, method="A", tau=config.tau).detach().cpu().numpy()
        p_pos = scored[:, 2]
        p_neg = scored[:, 3]
    else:
        emb = _normalize_embeddings(user_embeddings)
        sim = (emb[rows] * emb[cols]).sum(dim=1).cpu().numpy()
        p_pos = 1 / (1 + np.exp(-sim))
        p_neg = 1 / (1 + np.exp(sim))

    pos_scores = p_pos[pos_mask]
    neg_scores = p_neg[neg_mask]
    pos_edges = np.stack([rows[pos_mask], cols[pos_mask]], axis=1)
    neg_edges = np.stack([rows[neg_mask], cols[neg_mask]], axis=1)

    num_pos_del = int(len(pos_edges) * config.delete_ratio)
    num_neg_del = int(len(neg_edges) * config.delete_ratio)

    pos_del_idx = np.argsort(pos_scores)[:num_pos_del] if num_pos_del > 0 else np.array([], dtype=np.int64)
    neg_del_idx = np.argsort(neg_scores)[:num_neg_del] if num_neg_del > 0 else np.array([], dtype=np.int64)

    pos_del_edges = pos_edges[pos_del_idx] if len(pos_del_idx) else np.empty((0, 2), dtype=np.int64)
    neg_del_edges = neg_edges[neg_del_idx] if len(neg_del_idx) else np.empty((0, 2), dtype=np.int64)

    # build sets for fast lookup
    edge_set = {(int(u), int(v)): int(s) for u, v, s in zip(rows, cols, signs)}

    pos_deg, neg_deg, total_deg = _build_degree_counts(n_users, rows, cols, signs)

    deleted_pos = 0
    deleted_neg = 0
    for u, v in pos_del_edges:
        if total_deg[u] - 1 < config.delta_min or total_deg[v] - 1 < config.delta_min:
            continue
        edge_set.pop((int(u), int(v)), None)
        pos_deg[u] -= 1
        pos_deg[v] -= 1
        total_deg[u] -= 1
        total_deg[v] -= 1
        deleted_pos += 1

    for u, v in neg_del_edges:
        if total_deg[u] - 1 < config.delta_min or total_deg[v] - 1 < config.delta_min:
            continue
        edge_set.pop((int(u), int(v)), None)
        neg_deg[u] -= 1
        neg_deg[v] -= 1
        total_deg[u] -= 1
        total_deg[v] -= 1
        deleted_neg += 1

    # candidate generation using kNN
    emb = _normalize_embeddings(user_embeddings)
    knn = _build_knn(emb, config.knn_k, device)

    pos_heap: List[Tuple[float, int, int]] = []
    neg_heap: List[Tuple[float, int, int]] = []

    for u in tqdm(range(n_users), desc="ESA-DA kNN", leave=False):
        for v in knn[u]:
            v = int(v)
            if v == u:
                continue
            uu, vv = (u, v) if u < v else (v, u)
            if uu != u:
                continue
            if (uu, vv) in edge_set:
                continue
            sim = torch.dot(emb[uu], emb[vv]).item()
            p_pos = 1 / (1 + np.exp(-sim / max(1e-6, config.tau)))
            p_neg = 1 / (1 + np.exp(sim / max(1e-6, config.tau)))

            if len(pos_heap) < config.top_k_candidate:
                heapq.heappush(pos_heap, (p_pos, uu, vv))
            elif p_pos > pos_heap[0][0]:
                heapq.heapreplace(pos_heap, (p_pos, uu, vv))

            if len(neg_heap) < config.top_k_candidate:
                heapq.heappush(neg_heap, (p_neg, uu, vv))
            elif p_neg > neg_heap[0][0]:
                heapq.heapreplace(neg_heap, (p_neg, uu, vv))

    pos_candidates = sorted(pos_heap, key=lambda x: -x[0])
    neg_candidates = sorted(neg_heap, key=lambda x: -x[0])

    num_add_total = int(max(1, round(num_edges_before * config.add_ratio)))
    num_pos_add = min(len(pos_candidates), num_add_total)
    num_neg_add = min(len(neg_candidates), num_add_total)

    # structural balance guard matrices
    guard_rows: List[int] = []
    guard_cols: List[int] = []
    guard_signs: List[int] = []
    for (u, v), sign in edge_set.items():
        guard_rows.extend([u, v])
        guard_cols.extend([v, u])
        guard_signs.extend([sign, sign])

    abs_net = sp.coo_matrix(
        (np.ones(len(guard_rows), dtype=np.float32), (guard_rows, guard_cols)),
        shape=(n_users, n_users),
    )
    signed_net = sp.coo_matrix(
        (np.asarray(guard_signs, dtype=np.float32), (guard_rows, guard_cols)),
        shape=(n_users, n_users),
    )

    abs_csr = abs_net.tocsr()
    abs_csc = abs_net.tocsc()
    signed_csr = signed_net.tocsr()
    signed_csc = signed_net.tocsc()

    added_pos = 0
    added_neg = 0
    balance_accept = 0
    balance_total = 0

    for _, u, v in pos_candidates[:num_pos_add]:
        if pos_deg[u] + 1 > config.d_pos_max or pos_deg[v] + 1 > config.d_pos_max:
            continue
        if config.balance_guard:
            balance_total += 1
            if not _edge_balance_guard(u, v, 1, abs_csr, abs_csc, signed_csr, signed_csc):
                continue
            balance_accept += 1
        edge_set[(u, v)] = 1
        pos_deg[u] += 1
        pos_deg[v] += 1
        total_deg[u] += 1
        total_deg[v] += 1
        added_pos += 1

    for _, u, v in neg_candidates[:num_neg_add]:
        if neg_deg[u] + 1 > config.d_neg_max or neg_deg[v] + 1 > config.d_neg_max:
            continue
        if config.balance_guard:
            balance_total += 1
            if not _edge_balance_guard(u, v, -1, abs_csr, abs_csc, signed_csr, signed_csc):
                continue
            balance_accept += 1
        edge_set[(u, v)] = -1
        neg_deg[u] += 1
        neg_deg[v] += 1
        total_deg[u] += 1
        total_deg[v] += 1
        added_neg += 1

    # rebuild symmetric net matrix
    final_rows = []
    final_cols = []
    final_signs = []
    for (u, v), s in edge_set.items():
        final_rows.extend([u, v])
        final_cols.extend([v, u])
        final_signs.extend([s, s])

    refined = sp.coo_matrix((final_signs, (final_rows, final_cols)), shape=(n_users, n_users))

    stats = {
        "num_edges_before": float(num_edges_before),
        "num_edges_after": float(len(edge_set)),
        "deleted_pos": float(deleted_pos),
        "deleted_neg": float(deleted_neg),
        "added_pos": float(added_pos),
        "added_neg": float(added_neg),
        "balance_accept_ratio": float(balance_accept / balance_total) if balance_total > 0 else 1.0,
    }

    return refined, stats
