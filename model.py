import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "HLLM", "code"))
from REC.model.HLLM.modeling_llama import LlamaForCausalLM


# ============================================================
# 工具函数
# ============================================================

def all_gather(data, group=None, sync_grads=False):
    """跨 GPU 聚合数据（与原版 basemodel.py 一致）"""
    group = group if group is not None else dist.group.WORLD
    if dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed import nn
        if sync_grads:
            return torch.stack(nn.functional.all_gather(data, group=group), dim=0)
        with torch.no_grad():
            return torch.stack(nn.functional.all_gather(data, group=group), dim=0)
    else:
        return data.unsqueeze(0)


# ============================================================
# HLLM 模型
# ============================================================

class HLLM(nn.Module):

    def __init__(self, config):
        """
        任务:
          1. self.item_llm = self._create_llm(config.item_pretrain_dir)
          2. self.user_llm = self._create_llm(config.user_pretrain_dir)
          3. self.item_emb_tokens = nn.Parameter(torch.zeros(1, 1, hidden_size))
             初始化用 .normal_(mean=0, std=0.02)
          4. self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        """
        super().__init__()
        self.config = config
        self.item_llm = self._create_llm(config.item_pretrain_dir)
        self.user_llm = self._create_llm(config.user_pretrain_dir)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.all_item_emb = None  # 存储所有 item 的 embedding，评估时使用
        self.item_emb_tokens = nn.Parameter(torch.zeros(1, 1, self.item_llm.config.hidden_size))
        self.item_emb_tokens.data.normal_(mean=0, std=0.02)


    # ── 创建 LLM ────────────────────────────────

    def _create_llm(self, pretrain_dir, init=True):
        """
        任务:
          1. hf_config = AutoConfig.from_pretrained(pretrain_dir)
          2. 设置 hf_config 字段:
             - gradient_checkpointing
             - use_cache = False
             - output_hidden_states = True
             - return_dict = True
             - use_ft_flash_attn
          3. 加载权重:
             - LlamaForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
             或
             - LlamaForCausalLM(hf_config).cuda()  (init=False 时随机初始化)
        返回: LLM 实例
        """
        hf_config = AutoConfig.from_pretrained(pretrain_dir) #加载预训练模型的配置文件
        hf_config.gradient_checkpointing = self.config.gradient_checkpointing #启用梯度检查点以节省显存
        hf_config.use_cache = False #禁用缓存，因为我们在训练过程中不需要使用缓存
        hf_config.output_hidden_states = True #确保模型在前向传播时返回所有层的隐藏状态，这对于我们提取 item embedding 是必要的
        hf_config.return_dict = True #确保模型返回一个字典而不是元组，这样我们可以通过键来访问输出
        hf_config.use_ft_flash_attn = self.config.use_ft_flash_attn #如果配置中启用了使用 Flash Attention，则在模型配置中设置相应的标志
        model = LlamaForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
        return model
        

    # ── Item Embedding ───────────────────────────

    def forward_item_emb(self, input_ids, position_ids, cu_input_lens, llm):
        """
        输入:
          input_ids:     [total_tokens]  所有 item 文本拼接
          position_ids:  [total_tokens]  每个 token 在各自 item 内的位置(从0开始)
          cu_input_lens: [num_items]     每个 item 的 token 数量
          llm:           Item LLM

        步骤:
          1. inputs_embeds = llm.get_input_embeddings()(input_ids)    → [T, D]
          2. emb_pos = cu_input_lens.cumsum(dim=0) - 1                → [num_items]
             把 inputs_embeds[emb_pos] 替换成 self.item_emb_tokens
          3. output = llm(inputs_embeds=..., cu_input_lens=..., position_ids=...)
             取 output.hidden_states[-1]                               → [T, D]
          4. 在 emb_pos 位置切出 embedding                             → [num_items, D]

        返回: item embeddings [num_items, hidden_size]
        """
        input_embeds = llm.get_input_embeddings()(input_ids)    # [T, D]
        """
        llm() 的输入参数说明:
        input_ids	torch.LongTensor (batch, seq_len)	是*	输入 token 的 ID 序列，由 tokenizer 生成。
        attention_mask	torch.LongTensor (batch, seq_len)	否	标记哪些位置是真实 token（1） vs 填充 token（0）。若不提供，模型会默认所有位置都有效。
        labels	torch.LongTensor (batch, seq_len)	否	用于计算语言模型损失（交叉熵）。通常是将 input_ids 右移一位（预测下一个 token）。提供 labels 时，forward() 会返回 loss。
        position_ids	torch.LongTensor (batch, seq_len)	否	手动指定每个 token 的位置索引。若不提供，模型会自动从 0 开始生成连续位置。
        past_key_values	tuple(tuple(torch.FloatTensor)) 或 Cache 对象	否	缓存先前 token 的 key/value 对，用于自回归生成时加速。通常由模型内部或 generate() 方法管理。
        inputs_embeds	torch.FloatTensor (batch, seq_len, hidden_size)	否	直接提供嵌入向量，代替 input_ids。两者只能提供一个。
        use_cache	bool	否	是否返回 past_key_values。默认为 None（使用模型配置中的值）。
        output_attentions	bool	否	是否返回所有层的注意力权重。默认为 False。
        output_hidden_states	bool	否	是否返回所有层的隐藏状态。默认为 False。
        return_dict	bool	否	是否返回 CausalLMOutputWithPast 等命名元组而非普通 tuple。默认为 True。
        """
        emb_pos = cu_input_lens.cumsum(dim=0) - 1 # [num_items]
        input_embeds[emb_pos] = self.item_emb_tokens.squeeze(0) # 替换成可训练的 item embedding token
        output = llm(inputs_embeds=input_embeds.unsqueeze(0), position_ids=position_ids.unsqueeze(0), cu_input_lens=cu_input_lens) # [T, D]
        return (output.hidden_states[-1]).squeeze(0)[emb_pos] # 在 emb_pos 位置切出 embedding                             → [num_items, D]

    # ── NCE Loss ────────────────────────────────

    def nce_loss(self, user_embs, target_pos, target_neg, mask):
        """
        输入:
          user_embs:   [N, S, D]   用户序列表示
          target_pos:  [N, S, D]   正样本 item embedding
          target_neg:  [N, K, D]   负样本 item embedding
          mask:        [N, S]      1=有效, 0=padding

        与原版一致: 对 target_neg 做 all_gather 跨卡扩大负样本池
        """
        with torch.no_grad():
            self.temperature.clamp_(0, np.log(100))
        scale = self.temperature.exp()
        D = target_neg.size(-1)

        user_embs = F.normalize(user_embs, dim=-1)  # [N, S, D]
        target_pos = F.normalize(target_pos, dim=-1)  # [N, S, D]

        shape0, shape1 = torch.nonzero(mask, as_tuple=True)
        user_embs = user_embs[shape0, shape1]  # [V, D]
        target_pos = target_pos[shape0, shape1]  # [V, D]
        target_neg = target_neg[shape0]  # [V, K, D]

        pos_logits = (user_embs * target_pos).sum(dim=-1).unsqueeze(-1)  # [V, 1]

        # 跨卡聚合负样本（与原版 nce_loss 一致）
        target_neg_flat = target_neg.reshape(-1, D)  # [V*K, D]
        neg_embedding_all = all_gather(target_neg_flat, sync_grads=True).reshape(-1, D)  # [world*V*K, D]
        neg_embedding_all = F.normalize(neg_embedding_all, dim=-1)
        neg_embedding_all = neg_embedding_all.transpose(-1, -2)  # [D, world*V*K]

        neg_logits = torch.matmul(user_embs, neg_embedding_all)  # [V, world*V*K]
        fix_logits = torch.matmul(target_pos, neg_embedding_all)  # [V, world*V*K]
        neg_logits[fix_logits > self.config.nce_thres] = torch.finfo(neg_logits.dtype).min

        logits = torch.cat([pos_logits, neg_logits], dim=-1) * scale  # [V, 1+world*V*K]
        label = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, label)
        return loss
    # ── 训练 Forward ────────────────────────────

    def forward(self, batch, mode='train'):
        """
        batch 字段: attention_mask, pos_input_ids, pos_cu_lens, pos_position_ids,
                    neg_input_ids, neg_cu_lens, neg_position_ids

        步骤:
          1. pos_emb = forward_item_emb(pos_items, item_llm)     → [N*(S+1), D]
          2. neg_emb = forward_item_emb(neg_items, item_llm)     → [N*K, D]
          3. pos_emb.reshape(N, S+1, D)
             target_pos = pos_emb[:, 1:, :]   用户历史 → 真实下一个
             user_input = pos_emb[:, :-1, :]  喂给 User LLM
          4. user_emb = user_llm(inputs_embeds=user_input, attention_mask=mask)
                       .hidden_states[-1]                        → [N, S, D]
          5. loss = nce_loss(user_emb, target_pos, neg_emb, mask)

        返回: {"loss": loss}
        """
        pos_item_emb = self.forward_item_emb(batch["pos_input_ids"], batch["pos_position_ids"], batch["pos_cu_lens"], self.item_llm) # [N*(S+1), D]
        neg_item_emb = self.forward_item_emb(batch["neg_input_ids"], batch["neg_position_ids"], batch["neg_cu_lens"], self.item_llm) # [N*K, D]
        pos_item_emb = pos_item_emb.view(-1, self.config.MAX_ITEM_LIST_LENGTH + 1, pos_item_emb.shape[-1]) # [N, S+1, D]
        neg_item_emb = neg_item_emb.view(-1, self.config.num_negatives, neg_item_emb.shape[-1]) # [N, K, D]
        target_pos = pos_item_emb[:, 1:, :]   # 用户历史 → 真实下一个 [N, S, D]
        user_input = pos_item_emb[:, :-1, :]  # 喂给 User LLM [N, S, D]
        user_emb = self.user_llm(inputs_embeds=user_input, attention_mask=batch["attention_mask"]).hidden_states[-1] # [N, S, D]
        loss = self.nce_loss(user_emb, target_pos, neg_item_emb, batch["attention_mask"])
        return {"loss": loss}

    # ── 推理 ────────────────────────────────────

    @torch.no_grad()
    def _predict(self, item_seq, mask, all_item_ids, all_item_cu_lens, all_item_position_ids):
        """
        item_seq: [N, S] 用户历史 item id 序列
        mask:     [N, S] 1=有效, 0=padding
        步骤:
          1. pos_emb = item_feature[item_seq]
          2. user_emb = user_llm(inputs_embeds=pos_emb, attention_mask=mask)
                         .hidden_states[-1][:, -1]
          3. L2 归一化后 return matmul(user_emb, item_feature.T)

        返回: scores [N, num_items]
        """
        pos_emb = self._compute_item(all_item_ids, all_item_cu_lens, all_item_position_ids)[item_seq] # [N, S, D]
        user_emb = self.user_llm(inputs_embeds=pos_emb, attention_mask=mask).hidden_states[-1][:, -1] # [N, D]
        user_emb = F.normalize(user_emb, dim=-1) # [N, D]
        item_emb = F.normalize(self._compute_item(all_item_ids, all_item_cu_lens, all_item_position_ids), dim=-1) # [num_items, D]
        scores = user_emb @ item_emb.transpose(-2, -1) # [N, num_items]
        return scores

    @torch.no_grad()
    def _compute_item(self, all_item_ids, all_item_cu_lens, all_item_position_ids, chunk_size=64):
        """
        预计算所有 item 的 embedding（评估用），分批处理避免 OOM
        返回: item embeddings [num_items, D]
        """
        if self.all_item_emb is not None:
            return self.all_item_emb

        num_items = len(all_item_cu_lens)
        cu_lens_cumsum = all_item_cu_lens.cumsum(dim=0)
        all_embs = []

        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            token_start = cu_lens_cumsum[start - 1].item() if start > 0 else 0
            token_end = cu_lens_cumsum[end - 1].item()

            batch_ids = all_item_ids[token_start:token_end]
            batch_position = all_item_position_ids[token_start:token_end]
            batch_cu_lens = all_item_cu_lens[start:end]

            emb = self.forward_item_emb(batch_ids, batch_position, batch_cu_lens, self.item_llm)
            all_embs.append(emb)

        self.all_item_emb = torch.cat(all_embs, dim=0)  # [num_items, D]
        return self.all_item_emb

    @torch.no_grad()
    def _non(self):
      self.all_item_emb = None
