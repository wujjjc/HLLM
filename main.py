from model import *
from data import *
from config import *
from train import train
config = HLLMConfig()
net = HLLM(config)
item_map = new_item_id_mapping(os.path.join(config.text_path, config.dataset_name + "csv"))
train_history, test_history = split_user_histories(config, 0.1, item_map)
train_dataset = HLLMDataset(config, train_history, item_map)
test_dataset = HLLMTestDataset(config, test_history)
all_item_ids, all_item_cu_lens, all_item_position_ids = train_dataset.get_all_items()
train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
train(net, train_loader, test_loader, all_item_ids, all_item_cu_lens, all_item_position_ids, optimizer, scheduler, config.epochs)