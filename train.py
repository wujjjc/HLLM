import os
import torch
import numpy as np
from model import HLLM


def save_checkpoint(hllm, optimizer, scheduler, epoch, best_metric, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": hllm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, "latest.pt"))
    print(f"Checkpoint saved: epoch {epoch}, best Recall@10 = {best_metric:.4f}")


def load_checkpoint(checkpoint_path, hllm, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hllm.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", 0.0)
    print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, best Recall@10 = {best_metric:.4f}")
    return start_epoch, best_metric


def metric(hllm, test_loader, all_item_ids, all_item_cu_lens, all_item_position_ids, list_k=[10, 20, 50, 100], device=None):
    """评估HLLM的召回率和NDCG
    Args:
        hllm(_type_): 模型
        test_loader (_type_): 测试数据加载器
        all_item_ids: 所有物品的ID token列表
        all_item_cu_lens: 所有物品ID token列表的累积长度列表
        all_item_position_ids: 所有物品ID token列表对应的位置ID列表
        list_k (_type_): Top-k 列表
        device (_type_): 设备
    Returns:
        recall_dict: 召回率字典，key为k，value为对应的平均召回率
        ndcg_dict: NDCG字典，key为k，value为对应的平均NDCG值
    """
    hllm.eval()
    recall_dict = {k: [] for k in list_k}
    ndcg_dict = {k: [] for k in list_k}
    with torch.no_grad():
        max_k = max(list_k)
        for batch in test_loader:
            seq = batch["history"].to(device)  # [N, S] 用户历史序列
            pos_id = batch["pos_id"].to(device)  # [N] 正样本物品ID
            mask = batch["mask"].to(device)  # [N, S] 序列掩码
            topk_scores, topk_indices = hllm._predict(seq, mask, all_item_ids, all_item_cu_lens, all_item_position_ids, topk=max_k)
            topk_item_ids = topk_indices.cpu().numpy()  # [N, max_k]
            pos_id_expanded = pos_id.unsqueeze(1).cpu().numpy()  # [N, 1]
            for k in list_k:
                hits = (topk_item_ids[:, :k] == pos_id_expanded).astype(float)  # [N, k]
                recall = hits.sum(axis=1) # [N]
                ndcg = hits / np.log2(np.arange(2, k + 2).astype(float))  # [N, k]
                ndcg = ndcg.sum(axis=1)  # [N]
                recall_dict[k].extend(recall)
                ndcg_dict[k].extend(ndcg)
    for k in list_k:
        recall_dict[k] = np.mean(recall_dict[k])
        ndcg_dict[k] = np.mean(ndcg_dict[k])
    return recall_dict, ndcg_dict


def train(hllm, train_loader, test_loader, all_item_ids, all_item_cu_lens, all_item_position_ids, optimizer, scheduler, epoch, device, config, start_epoch=0, best_metric=0.0):
    """
    训练
    Args:
        hllm (_type_): HLLM 模型实例
        train_loader (_type_): 训练数据加载器
        test_loader (_type_): 测试数据加载器
        all_item_ids: 所有物品的ID token列表
        all_item_cu_lens: 所有物品ID token列表的累积长度列表
        all_item_position_ids: 所有物品ID token列表对应的位置ID列表
        optimizer (_type_): 优化器
        scheduler (_type_): 学习率调度器
        epoch (_type_): 训练轮数
        device (_type_): 设备
        config: 配置对象
        start_epoch: 起始 epoch（续训时使用）
        best_metric: 最佳 Recall@10（续训时使用）
    """
    print("Starting training...")
    hllm.train()
    list_k = [10, 20, 50, 100]
    for e in range(start_epoch, epoch):
        total_loss = 0.0
        i = 0
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
            # with open(f"train_loss_batch.txt", "a") as f:
            #     f.write(f"Epoch {e}, Batch: {i}: Loss = {loss.item()}\n")
            # i += 1
        avg_loss = total_loss / len(train_loader)
        with open(f"train_loss.txt", "a") as f:
            f.write(f"Epoch {e}: Average Loss = {avg_loss}")
        torch.save(hllm.state_dict(), "latest_model.pt")
        recall_dict, ndcg_dict = metric(hllm, test_loader, all_item_ids, all_item_cu_lens, all_item_position_ids, list_k, device)
        with open(f"test_metrics.txt", "a") as f:
            for k in list_k:
                f.write(f"Epoch {e}, K={k}: Recall = {recall_dict[k] * 100} %, NDCG = {ndcg_dict[k] * 100}%\n")
        save_checkpoint(hllm, optimizer, scheduler, e, best_metric, config.checkpoint_dir)
        if recall_dict[10] > best_metric:
            best_metric = recall_dict[10]
            best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            torch.save(hllm.state_dict(), best_path)
            print(f"New best model saved: Recall@10 = {best_metric:.4f}")
