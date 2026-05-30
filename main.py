import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from config import HLLMConfig

# ==== 分布式初始化 ====
local_rank = int(os.environ.get("LOCAL_RANK", -1))
distributed = local_rank >= 0

if distributed:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = 1
    rank = 0

print(f"[Rank {rank}] device={device}, world_size={world_size}, distributed={distributed}")

from model import HLLM
from data import (new_item_id_mapping, split_user_histories,
                  HLLMDataset, HLLMTestDataset, NonConsecutiveSequentialDistributedSampler)
from train import train, load_checkpoint
from transformers import get_cosine_schedule_with_warmup

config = HLLMConfig()
config.local_rank = local_rank
config.world_size = world_size
config.distributed = distributed

dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
net = HLLM(config).to(device).to(dtype)

if distributed:
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)

item_map = new_item_id_mapping(config)
train_history, test_history = split_user_histories(config, 0.1, item_map)
train_dataset = HLLMDataset(config, train_history, item_map)
test_dataset = HLLMTestDataset(config, test_history)
if rank == 0:
    print("数据集构建完成")

all_item_ids, all_item_cu_lens, all_item_position_ids = train_dataset.get_all_items()
all_item_ids = all_item_ids.to(device)
all_item_cu_lens = all_item_cu_lens.to(device)
all_item_position_ids = all_item_position_ids.to(device)

# ==== DataLoader ====
train_sampler = DistributedSampler(train_dataset) if distributed else None
train_loader = DataLoader(
    train_dataset,
    batch_size=config.train_batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    collate_fn=train_dataset.collate_fn,
)

if distributed:
    test_sampler = NonConsecutiveSequentialDistributedSampler(test_dataset)
else:
    test_sampler = None
test_loader = DataLoader(
    test_dataset,
    batch_size=config.eval_batch_size,
    shuffle=False,
    sampler=test_sampler,
    collate_fn=test_dataset.collate_fn,
)

if rank == 0:
    print("数据加载完成")

optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(len(train_loader) * config.epochs * config.warmup_ratio),
    num_training_steps=len(train_loader) * config.epochs,
)

start_epoch, best_metric = 0, 0.0
checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pt")
if os.path.exists(checkpoint_path):
    start_epoch, best_metric = load_checkpoint(checkpoint_path, net, optimizer, scheduler)

train(
    net, train_loader, test_loader,
    all_item_ids, all_item_cu_lens, all_item_position_ids,
    optimizer, scheduler, config.epochs, device, config,
    start_epoch, best_metric, train_sampler=train_sampler,
)

if distributed:
    dist.destroy_process_group()
