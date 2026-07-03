"""Small end-to-end check that does not require a downloaded dataset."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import EpinionsDataset, InteractionDataset
from models import ESSRec
from modules import ESADAConfig, refine_signed_graph


def main() -> None:
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="ssc_loop_smoke_") as tmp:
        tmp_path = Path(tmp)
        inter_path = tmp_path / "tiny.inter"
        net_path = tmp_path / "tiny.net"

        interactions = [
            "user_id item_id rating",
            "1 10 5", "1 11 4", "1 12 2",
            "2 10 4", "2 13 1", "2 14 3",
            "3 11 5", "3 12 4", "3 14 2",
            "4 10 1", "4 13 5", "4 14 4",
        ]
        signed_edges = [
            "source_id target_id sign",
            "1 2 1", "2 1 1", "1 3 -1", "3 1 -1",
            "2 4 -1", "4 2 -1", "3 4 1", "4 3 1",
        ]
        inter_path.write_text("\n".join(interactions) + "\n", encoding="utf-8")
        net_path.write_text("\n".join(signed_edges) + "\n", encoding="utf-8")

        dataset = EpinionsDataset(inter_path, net_path, train_ratio=0.7, val_ratio=0.1, seed=7)
        split_sizes = (len(dataset.train_data), len(dataset.val_data), len(dataset.test_data))
        if split_sizes != (8, 1, 3):
            raise RuntimeError(f"Unexpected 70/10/20 split for 12 records: {split_sizes}")
        args = SimpleNamespace(
            embedding_size=8,
            gnn_layer_size=1,
            gnn_layer_size_k=1,
            sim_threshold=0.0,
            device=torch.device("cpu"),
            alpha=0.2,
            ablation="full",
            contrastive_samples=32,
            contrast_margin=1.0,
        )
        model = ESSRec(dataset, args)

        batch_records = dataset.train_data[:4]
        batch = InteractionDataset.collate_fn(
            [
                {model.USER_ID: u, model.ITEM_ID: i, model.RATING: r}
                for u, i, r in batch_records
            ]
        )
        loss, parts = model.calculate_loss(batch, return_dict=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {parts}")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            user_embeddings, _ = model.get_user_item_embeddings()
        config = ESADAConfig(
            delete_ratio=0.1,
            add_ratio=0.1,
            knn_k=2,
            top_k_candidate=8,
            d_pos_max=10,
            d_neg_max=10,
            delta_min=1,
            balance_guard=True,
        )
        refined, stats = refine_signed_graph(
            dataset.net_matrix,
            user_embeddings,
            config,
            model=model,
            device=torch.device("cpu"),
        )
        if refined.shape != dataset.net_matrix.shape:
            raise RuntimeError("Refined graph shape changed unexpectedly.")

        print("SSC-Loop smoke test passed.")
        print(f"loss={loss.item():.6f} parts={parts}")
        print(f"refinement={stats}")


if __name__ == "__main__":
    main()
