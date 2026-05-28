import torch    
import pandas as pd                                                                                                                                                               
from torch.utils.data import Dataset, DataLoader                                                                                                                                  
from transformers import AutoTokenizer  
import os

def new_item_id_mapping(config):
    """将原来的itemid映射到从0开始的整数
    Args:
        config (_type_): 配置对象
    returns:
        dict: 原 item_id 到新 item_id 的映射字典
    """
    df = pd.read_csv(os.path.join(config.data_path, config.dataset_name + ".csv"))
    unique_items = df['item_id'].unique()
    return {item_id: idx for idx, item_id in enumerate(unique_items)}


def split_user_histories(config, test_ratio=0.1, item_id_mapping=None):
    """
    读取csv文件产生用户交互字典，格式为 {user_id: [(item_id, timestamp), ...]}，并按时间戳排序，然后划分为测试集和数据集

    Args:
        config (_type_): 配置对象
        test_ratio (_type_): 测试集占比
        item_id_mapping (_type_): 原 item_id 到新 item_id 的映射字典，如果提供则将 item_id 替换为映射后的 id
    Returns:
        train_histories: 训练集用户交互字典
        test_histories: 测试集用户交互字典
    """
    file_path = os.path.join(config.data_path, config.dataset_name + ".csv")
    df = pd.read_csv(file_path)
    user_histories = {}
    for row in df.itertuples():
        item_id = item_id_mapping.get(row.item_id, row.item_id) if item_id_mapping else row.item_id
        user_histories.setdefault(row.user_id, []).append((item_id, int(row.timestamp)))
    for _, val in user_histories.items():
        val.sort(key=lambda x: x[1])  # 按时间戳排序
    user_id = list(user_histories.keys())
    test_size = int(len(user_id) * test_ratio)
    test_user_id = set(user_id[:test_size])
    train_histories = {user_id: history for user_id, history in user_histories.items() if user_id not in test_user_id}
    test_histories = {user_id: history for user_id, history in user_histories.items() if user_id in test_user_id}
    return train_histories, test_histories

class HLLMTestDataset(Dataset):
    """测试集 Dataset，构造用户历史序列和正样本
    每条样本：用户的前 N-1 个物品作为历史序列，第 N 个物品作为正样本     
    | item_id | description   | title   | tag (optional) |
    |---------|---------------|---------|----------------|
    | item_i  | description_i | title_i | tag_i          |     
    
    | item_id | user_id | timestamp |
    |---------|---------|-----------|
    | item_i  | user_j  | time_k    |                                                                                                    
    """   
    def __init__(self, config, user_histories):
        self.config = config
        self.user_histories = user_histories
        self.user_histories_id = list(self.user_histories.keys())  # 存储用户的id列表
    def __len__(self):
        return len(self.user_histories)
    def __getitem__(self, idx):
        user_id = self.user_histories_id[idx]
        history = [item_id for item_id, _ in self.user_histories[user_id]]
        pos_id = history[-1] # 最后一个物品作为正样本
        seq = history[:-1]
        mask = []
        if len(seq) > self.config.MAX_ITEM_LIST_LENGTH:
            seq = seq[-self.config.MAX_ITEM_LIST_LENGTH:]
            mask = [1] * self.config.MAX_ITEM_LIST_LENGTH
        else:
            mask = [0] * (self.config.MAX_ITEM_LIST_LENGTH - len(seq)) + [1] * len(seq)
            seq = [0] * (self.config.MAX_ITEM_LIST_LENGTH - len(seq)) + seq  # 前面补0表示 padding 的 item_id
        return {
            "pos_id": pos_id,
            "history": seq,
            "mask" : mask
        }
    
    def collate_fn(self, batch):
        seq = [item["history"] for item in batch]
        pos_id = [item["pos_id"] for item in batch]
        mask = [item["mask"] for item in batch]
        return {
            "pos_id": torch.tensor(pos_id, dtype=torch.long),
            "history": torch.tensor(seq, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool)
        }


class HLLMDataset(Dataset):
    """                                                                                                                                                                           
    加载交互数据 + 物品文本，构造训练样本                                                                                                                                         
    每条样本：用户的前 N-1 个物品作为历史序列，第 N 个物品作为正样本     
    | item_id | description   | title   | tag (optional) |
    |---------|---------------|---------|----------------|
    | item_i  | description_i | title_i | tag_i          |     
    
    | item_id | user_id | timestamp |
    |---------|---------|-----------|
    | item_i  | user_j  | time_k    |                                                                                                    
    """   
    def __init__(self, config, user_histories, item_id_mapping):
        self.config = config
        df_items = pd.read_csv(os.path.join(config.text_path, config.dataset_name + ".csv"))
        self.user_histories = user_histories
        self.item_texts = {}
        self.all_items = set()
        self.max_len = config.MAX_ITEM_LIST_LENGTH
        self.negative_num = config.num_negatives
        self.user_histories_id = list(self.user_histories.keys())  # 存储用户的id列表
        for row in df_items.itertuples():
            item_id = item_id_mapping.get(row.item_id, row.item_id) if item_id_mapping else row.item_id
            self.all_items.add(item_id)
            parts = []                                                                                                                                                                    
            for key in config.text_keys:                                                                                                                                                  
                val = getattr(row, key, "")                                                                                                                                               
                if val:                                                                                                                                                                   
                    parts.append(f"{key}: {val}")
            self.item_texts[item_id] = ", ".join(parts) # 将多个文本字段合并成一个字符串
        self.tokenizer = AutoTokenizer.from_pretrained(config.item_pretrain_dir)  
        self.tokenizer.pad_token = self.tokenizer.eos_token  
    def __len__(self):
        return len(self.user_histories)
    def get_all_items(self):
        """返回所有item的token表示，相对位置，长度
        """
        all_items_token_ids = []
        all_items_position_ids = []
        all_items_cu_lens = []
        for item_id in range(len(self.all_items)):
            input_ids, position_ids = self.tokenize_item(item_id)
            all_items_token_ids.append(input_ids)
            all_items_position_ids.append(position_ids)
            all_items_cu_lens.append(len(input_ids))
        all_items_token_ids = torch.cat(all_items_token_ids, dim=0)
        all_items_position_ids = torch.cat(all_items_position_ids, dim=0)
        all_items_cu_lens = torch.tensor(all_items_cu_lens, dtype=torch.int32)
        return all_items_token_ids, all_items_position_ids, all_items_cu_lens
        
    def tokenize_item(self, item_id):
        """
        将item_id进行token化，返回item_id对应的token id和位置 id。
        """
        
        text = self.config.item_prompt + self.item_texts[item_id]
        """
        tokens.input_ids：形状为 (1, seq_len) 的 PyTorch 张量（因为 return_tensors="pt"），存储 token 对应的整数 ID。

        tokens.attention_mask：同样形状的张量，标记有效 token（1）和填充位置（0）。

        tokens.token_type_ids：可选，仅用于 BERT 等需要区分两个句子的模型。在你的例子中由于是单句输入，通常不会包含。

        tokens.keys()：查看返回的字段名。

        tokens.to(device)：可以将所有张量迁移到指定设备（如 GPU）。

        tokens['input_ids'] 或 tokens.input_ids 均可访问。
        """
        tokens = self.tokenizer(
            text, 
            max_length=self.config.MAX_TEXT_LENGTH,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )
        input_id = tokens.input_ids.squeeze(0)  # 去掉批次维度，变成 (seq_len,)
        position_ids = torch.arange(len(input_id))  # 位置 ID 从 0 到 seq_len-1
        return input_id, position_ids
    
    def _sample_negatives(self, history):
        all_items = self.all_items.copy()
        for item_id in history:
            all_items.discard(item_id)
        negatives = list(all_items)
        indices = torch.randperm(len(negatives))[:self.negative_num]                                                                                                                      
        neg_samples = [negatives[i] for i in indices.tolist()]  
        return neg_samples
    
    def __getitem__(self, idx):
        user_id = self.user_histories_id[idx]
        history = [item_id for item_id, _ in self.user_histories[user_id]]
        pos_id = history[-1] # 最后一个物品作为正样本
        seq = history[:-1]
        padding_len = self.max_len - len(seq)
        if padding_len > 0:
            mask = [0] * padding_len + [1] * len(seq)
            seq = [-1] * padding_len + seq  # 前面补-1表示 padding 的 item_id
        else:
            seq = seq[-self.max_len:]
            mask = [1] * self.max_len
        neg_samples = self._sample_negatives(history)
        return {
            "pos_id": pos_id,
            "neg_ids": neg_samples,
            "attention_mask": mask,
            "history": seq
        }
        
    def collate_fn(self, batch):
        """

        Args:
            batch (_type_): 字典列表，每个字典包含 "pos_id", "neg_ids", "attention_mask", "history" 四个键
                - "pos_id": 正样本的 item_id
                - "neg_ids": 负样本的 item_id 列表
                - "attention_mask": 历史序列的 attention mask 列表
                - "history": 历史序列的 item_id 列表

        Returns:
            dict: 包含以下键的字典：
                - "pos_input_ids": 正样本的 token id 张量，形状为 (total_pos_samples, seq_len)
                - "pos_cu_lens": 每个正样本的 token 长度张量，形状为 (total_pos_samples,)
                - "pos_position_ids": 正样本的相对位置 id 张量，形状为 (total_pos_samples, seq_len)
                - "neg_input_ids": 负样本的 token id 张量，形状为 (total_neg_samples, seq_len)
                - "neg_cu_lens": 每个负样本的 token 长度张量，形状为 (total_neg_samples,)
                - "neg_position_ids": 负样本的相对位置 id 张量，形状为 (total_neg_samples, seq_len)
                - "attention_mask": 历史序列的 attention mask 张量，形状为 (batch_size, max_history_len)
        """
        attention_mask = []
        pos_all_ids = []
        neg_all_ids = []
        for item in batch:
            attention_mask.append(item["attention_mask"])
            pos_all_ids.append(item["history"] + [item["pos_id"]])
            neg_all_ids.append(item["neg_ids"])
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)  # [N, S]
        pos_input_ids_list = [] # 每个 item 的 token id 列表
        pos_cu_lens = []       # 每个 item 的 token 长度
        pos_position_ids_list = [] # 每个 item 的位置 id 列表
        for pos_id_list in pos_all_ids:
            for pos_id in pos_id_list:
                if pos_id == -1:  # 跳过 padding 的 item_id
                    input_ids, position_ids = torch.tensor([self.tokenizer.pad_token_id]), torch.tensor([0]) # padding token 的 id 和位置 id
                else:
                    input_ids, position_ids = self.tokenize_item(pos_id) # 将 item_id 转换为文本并进行 tokenization
                pos_input_ids_list.append(input_ids) # 将每个 item 的 token id 列表添加到总列表中
                pos_cu_lens.append(len(input_ids)) # 记录每个 item 的 token 长度
                pos_position_ids_list.append(position_ids)  # 将每个 item 的相对位置 id 列表添加到总列表中
        pos_input_ids = torch.cat(pos_input_ids_list, dim=0)  # 拼接成一个大张量
        pos_cu_lens = torch.tensor(pos_cu_lens, dtype=torch.int32)
        pos_position_ids = torch.cat(pos_position_ids_list, dim=0)
        neg_input_ids_list = []
        neg_cu_lens = []
        neg_position_ids_list = []
        for neg_ids in neg_all_ids:
            for neg_id in neg_ids:
                input_ids, position_ids = self.tokenize_item(neg_id)
                neg_input_ids_list.append(input_ids)
                neg_cu_lens.append(len(input_ids))
                neg_position_ids_list.append(position_ids)
        neg_input_ids = torch.cat(neg_input_ids_list, dim=0)
        neg_cu_lens = torch.tensor(neg_cu_lens, dtype=torch.int32)
        neg_position_ids = torch.cat(neg_position_ids_list, dim=0)
        return {
            "pos_input_ids": pos_input_ids,
            "pos_cu_lens": pos_cu_lens,
            "pos_position_ids": pos_position_ids,
            "neg_input_ids": neg_input_ids,
            "neg_cu_lens": neg_cu_lens,
            "neg_position_ids": neg_position_ids,
            "attention_mask": attention_mask
        }
        
