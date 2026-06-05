import os
import torch
from config import get_device, HLLMConfig
from data import HLLMDataset, HLLMTestDataset
from train import *
config = HLLMConfig()

# CUDA 默认按 FASTEST_FIRST 排序（H100 优先），和 nvidia-smi 物理编号不同
# 设为 PCI_BUS_ID 让 CUDA 编号和 nvidia-smi 一致
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 必须在任何 CUDA 操作之前设置 CUDA_VISIBLE_DEVICES，否则会占用多张卡显存
os.environ["CUDA_VISIBLE_DEVICES"] = str(get_device())
device = torch.device("cuda:0")  # 环境变量生效后只有一张卡可见，固定 cuda:0
torch.cuda.set_device(device)
print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, device={device}")

from model import *
from data import *
from train import train, load_checkpoint
from transformers import get_cosine_schedule_with_warmup
dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
net = HLLM(config).to(device).to(dtype)
item_map = new_item_id_mapping(config)
train_history, test_history = split_user_histories(config, 0.1, item_map)
train_dataset = HLLMDataset(config, train_history, item_map)
test_dataset = HLLMTestDataset(config, test_history)
all_item_ids, all_item_cu_lens, all_item_position_ids = train_dataset.get_all_items()
all_item_ids = all_item_ids.to(device)
all_item_cu_lens = all_item_cu_lens.to(device)
all_item_position_ids = all_item_position_ids.to(device)
train_loader = HLLMDataset(train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
test_loader = HLLMTestDataset(test_dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader) * config.epochs * config.warmup_ratio),\
                                            num_training_steps=len(train_loader) * config.epochs)

start_epoch, best_metric = 0, 0.0
checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pt")
if os.path.exists(checkpoint_path):
    start_epoch, best_metric = load_checkpoint(checkpoint_path, net, optimizer, scheduler)

# train(net, train_loader, test_loader, all_item_ids, all_item_cu_lens, all_item_position_ids, optimizer, scheduler, config.epochs, device, config, start_epoch, best_metric)
metric(net, test_loader, all_item_ids, all_item_cu_lens, all_item_position_ids, device=device)