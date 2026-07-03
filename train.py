
import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import EpinionsDataset, InteractionDataset
from models import ESSRec
from modules import ESADAConfig, refine_signed_graph


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(dataset: EpinionsDataset, model: ESSRec, args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = InteractionDataset(dataset.train_data, model.USER_ID, model.ITEM_ID, model.RATING)
    val_dataset = InteractionDataset(dataset.val_data, model.USER_ID, model.ITEM_ID, model.RATING)
    test_dataset = InteractionDataset(dataset.test_data, model.USER_ID, model.ITEM_ID, model.RATING)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=InteractionDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=InteractionDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=InteractionDataset.collate_fn)
    return train_loader, val_loader, test_loader


def train_one_epoch(model: ESSRec, train_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
                    ) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_rec = 0.0
    total_cl = 0.0

    for batch_data in tqdm(train_loader, desc="train", unit="batch", dynamic_ncols=True, leave=False):
        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                if torch.is_tensor(v):
                    batch_data[k] = v.to(device)

        loss, loss_dict = model.calculate_loss(batch_data, return_dict=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rec += loss_dict.get("rec_loss", 0.0)
        total_cl += loss_dict.get("cl_loss", 0.0)

    denom = max(1, len(train_loader))
    return total_loss / denom, {
        "rec_loss": total_rec / denom,
        "cl_loss": total_cl / denom,
        "total_loss": total_loss / denom,
    }


def evaluate_model(model: ESSRec, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for batch_data in data_loader:
            rating_gt = batch_data[model.RATING].to(device)
            scores = model.predict(batch_data)
            preds.append(scores.cpu())
            gts.append(rating_gt.cpu())

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    rmse = torch.sqrt(torch.mean((preds - gts) ** 2))
    mae = torch.mean(torch.abs(preds - gts))
    return rmse.item(), mae.item()


def train_ssc_loop(dataset: EpinionsDataset, args, model_cls=ESSRec) -> Tuple[ESSRec, Dict[str, float]]:
    print(f"Training model: {model_cls.__name__}")
    model = model_cls(dataset, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader, test_loader = build_dataloaders(dataset, model, args)

    best_val_rmse = float("inf")
    best_val_mae = float("inf")
    best_ckpt_path = os.path.join(args.ckpt_dir, "best_val.pt")
    patience_counter = 0

    if args.ablation == "no_contrastive":
        args.alpha = 0.0

    effective_outer_loops = 1 if args.outer_loops <= 0 else args.outer_loops
    if args.ablation == "one_shot":
        effective_outer_loops = 1

    if args.warmup_epochs > 0:
        print(f"[Warmup] epochs={args.warmup_epochs}")
        for epoch in range(1, args.warmup_epochs + 1):
            avg_loss, loss_dict = train_one_epoch(model, train_loader, optimizer, args.device)
            val_rmse, val_mae = evaluate_model(model, val_loader, args.device)
            print(f"[Warmup {epoch}/{args.warmup_epochs}] loss={avg_loss:.4f} rec={loss_dict['rec_loss']:.4f} "
                  f"cl={loss_dict['cl_loss']:.4f} val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f}")

            if val_rmse < best_val_rmse:
                best_val_rmse, best_val_mae = val_rmse, val_mae
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), best_ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                print("[EarlyStop] triggered during warmup")
                break

    patience_counter = 0
    for outer in range(effective_outer_loops):
        do_refine = args.ablation != "no_esa_da" and args.outer_loops > 0
        if args.ablation == "one_shot" and outer > 0:
            do_refine = False

        if do_refine:
            print(f"[Outer {outer + 1}/{effective_outer_loops}] ESA-DA refining graph...")
            with torch.no_grad():
                user_emb, _ = model.get_user_item_embeddings(return_pno=False)
            config = ESADAConfig(
                delete_ratio=args.delete_ratio,
                add_ratio=args.add_ratio,
                knn_k=args.knn_k,
                top_k_candidate=args.top_k_candidate,
                d_pos_max=args.d_pos_max,
                d_neg_max=args.d_neg_max,
                delta_min=args.delta_min,
                balance_guard=args.balance_guard,
                tau=args.tau,
            )
            refined_net, stats = refine_signed_graph(dataset.net_matrix, user_emb, config, model=model,
                                                    device=args.device)
            dataset.update_net_matrix(refined_net)
            model.update_net_matrix(refined_net)
            print(
                f"[ESA-DA] edges {stats['num_edges_before']} -> {stats['num_edges_after']} | "
                f"+pos {stats['added_pos']} -pos {stats['deleted_pos']} | "
                f"+neg {stats['added_neg']} -neg {stats['deleted_neg']} | "
                f"balance_ratio {stats['balance_accept_ratio']:.4f}"
            )

        for epoch in range(1, args.inner_epochs + 1):
            avg_loss, loss_dict = train_one_epoch(model, train_loader, optimizer, args.device)
            val_rmse, val_mae = evaluate_model(model, val_loader, args.device)

            print(
                f"[Outer {outer + 1}/{effective_outer_loops} | Epoch {epoch}/{args.inner_epochs}] "
                f"loss={avg_loss:.4f} rec={loss_dict['rec_loss']:.4f} cl={loss_dict['cl_loss']:.4f} "
                f"val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f}"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse, best_val_mae = val_rmse, val_mae
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), best_ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print("[EarlyStop] triggered")
                break

        if patience_counter >= args.patience:
            break

    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=args.device), strict=False)

    test_rmse, test_mae = evaluate_model(model, test_loader, args.device)
    summary = {
        "best_val_rmse": best_val_rmse,
        "best_val_mae": best_val_mae,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }
    return model, summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESSRec SSC-Loop Training")
    parser.add_argument("--inter_path", type=str, required=True, help="Epinions .inter path")
    parser.add_argument("--net_path", type=str, required=True, help="Epinions .net path (original)")
    parser.add_argument("--ckpt_dir", type=str, default=str(Path("runs") / "checkpoints"))
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--gnn_layer_size", type=int, default=2)
    parser.add_argument("--gnn_layer_size_k", type=int, default=2)
    parser.add_argument("--sim_threshold", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--outer_loops", type=int, default=3)
    parser.add_argument("--inner_epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--contrast_margin", type=float, default=1.0)
    parser.add_argument("--contrastive_samples", type=int, default=4096)

    parser.add_argument("--delete_ratio", type=float, default=0.05)
    parser.add_argument("--add_ratio", type=float, default=0.05)
    parser.add_argument("--knn_k", type=int, default=200)
    parser.add_argument("--top_k_candidate", type=int, default=20000)
    parser.add_argument("--d_pos_max", type=int, default=50)
    parser.add_argument("--d_neg_max", type=int, default=50)
    parser.add_argument("--delta_min", type=int, default=1)
    parser.add_argument("--balance_guard", type=int, default=1)
    parser.add_argument("--tau", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_esa_da", "no_pno", "no_contrastive", "one_shot"])
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.balance_guard = bool(args.balance_guard)
    args.device = torch.device(args.device)

    for run in range(args.repeat):
        run_seed = args.seed + run
        set_random_seed(run_seed)

        print(f"\n[Run {run + 1}/{args.repeat}] seed={run_seed} ablation={args.ablation}")
        dataset = EpinionsDataset(
            args.inter_path,
            args.net_path,
            train_ratio=0.7,
            val_ratio=0.1,
            seed=run_seed,
        )

        print("Dataset summary:")
        print(f" - Users: {dataset.num('user_id:token')}")
        print(f" - Items: {dataset.num('item_id:token')}")
        print(f" - Interaction matrix shape: {dataset.interaction_matrix.shape}")
        print(f" - Net matrix shape: {dataset.net_matrix.shape}")
        print(f" - Train interactions: {len(dataset.train_data)}")
        print(f" - Val interactions: {len(dataset.val_data)}")
        print(f" - Test interactions: {len(dataset.test_data)}")

        _, summary = train_ssc_loop(dataset, args, model_cls=ESSRec)
        print(
            f"[Summary] best_val_RMSE={summary['best_val_rmse']:.4f} "
            f"best_val_MAE={summary['best_val_mae']:.4f} "
            f"test_RMSE={summary['test_rmse']:.4f} "
            f"test_MAE={summary['test_mae']:.4f}"
        )


if __name__ == "__main__":
    main()
