import os
import torch
import torch.distributed as dist
import numpy as np


def save_checkpoint(hllm, optimizer, scheduler, epoch, best_metric, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # DDP 模式下保存 module 的 state_dict
    model_to_save = hllm.module if hasattr(hllm, "module") else hllm
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, "latest.pt"))
    print(f"Checkpoint saved: epoch {epoch}, best Recall@10 = {best_metric:.4f}")


def load_checkpoint(checkpoint_path, hllm, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # DDP 模式下加载 module 的 state_dict
    model_to_load = hllm.module if hasattr(hllm, "module") else hllm
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", 0.0)
    print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, best Recall@10 = {best_metric:.4f}")
    return start_epoch, best_metric


def distributed_concat(tensor, num_total_examples):
    """跨 GPU 聚合指标（与原版 trainer.py 一致）"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


def metric(hllm, test_loader, all_item_ids, all_item_cu_lens, all_item_position_ids,
           list_k=[10, 20, 50, 100], device=None, distributed=False):
    """评估HLLM的召回率和NDCG（支持分布式）"""
    hllm.eval()
    recall_dict = {k: [] for k in list_k}
    ndcg_dict = {k: [] for k in list_k}
    with torch.no_grad():
        # 访问真实模型（DDP 包装下用 module）
        real_model = hllm.module if hasattr(hllm, "module") else hllm
        real_model._non()  # 清除缓存的所有物品嵌入
        for batch in test_loader:
            seq = batch["history"].to(device)
            pos_id = batch["pos_id"].to(device)
            mask = batch["mask"].to(device)
            score = real_model._predict(seq, mask, all_item_ids, all_item_cu_lens, all_item_position_ids)
            for k in list_k:
                top_k_indices = torch.topk(score, k=k, dim=1).indices
                top_k_item_ids = top_k_indices.cpu().numpy()
                pos_id_expanded = pos_id.unsqueeze(1).cpu().numpy()
                hits = (top_k_item_ids == pos_id_expanded).astype(float)
                recall = hits.sum(axis=1)
                ndcg = hits / np.log2(np.arange(2, k + 2).astype(float))
                ndcg = ndcg.sum(axis=1)
                recall_dict[k].extend(recall)
                ndcg_dict[k].extend(ndcg)

    # 跨卡聚合指标
    if distributed and dist.is_initialized() and dist.get_world_size() > 1:
        num_total = len(test_loader.dataset)
        for k in list_k:
            recall_t = torch.tensor(recall_dict[k], device=device, dtype=torch.float64)
            ndcg_t = torch.tensor(ndcg_dict[k], device=device, dtype=torch.float64)
            recall_t = distributed_concat(recall_t, num_total)
            ndcg_t = distributed_concat(ndcg_t, num_total)
            recall_dict[k] = recall_t.cpu().numpy()
            ndcg_dict[k] = ndcg_t.cpu().numpy()

    for k in list_k:
        recall_dict[k] = np.mean(recall_dict[k])
        ndcg_dict[k] = np.mean(ndcg_dict[k])
    return recall_dict, ndcg_dict


def train(hllm, train_loader, test_loader, all_item_ids, all_item_cu_lens, all_item_position_ids,
          optimizer, scheduler, epoch, device, config, start_epoch=0, best_metric=0.0, train_sampler=None):
    """训练（支持分布式）"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    distributed = config.distributed

    if rank == 0:
        print("Starting training...")
    hllm.train()
    list_k = [10, 20, 50, 100]

    for e in range(start_epoch, epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(e)

        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            for key in batch:
                batch[key] = batch[key].to(device)
            dic = hllm(batch)
            loss = dic["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hllm.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            with open("train_loss.txt", "a") as f:
                f.write(f"Epoch {e}: Average Loss = {avg_loss}\n")

        recall_dict, ndcg_dict = metric(
            hllm, test_loader, all_item_ids, all_item_cu_lens, all_item_position_ids,
            list_k, device, distributed=distributed,
        )

        if rank == 0:
            with open("test_metrics.txt", "a") as f:
                for k in list_k:
                    f.write(f"Epoch {e}, K={k}: Recall = {recall_dict[k] * 100} %, NDCG = {ndcg_dict[k] * 100}%\n")
            save_checkpoint(hllm, optimizer, scheduler, e, best_metric, config.checkpoint_dir)
            if recall_dict[10] > best_metric:
                best_metric = recall_dict[10]
                model_to_save = hllm.module if hasattr(hllm, "module") else hllm
                best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                torch.save(model_to_save.state_dict(), best_path)
                print(f"New best model saved: Recall@10 = {best_metric:.4f}")

        # 同步所有卡，确保 checkpoint 保存完成
        if distributed and dist.is_initialized():
            dist.barrier()
