from dataclasses import dataclass, field
from typing import List, Optional                                                                                                                                                 
                                                                                                                                                                                
@dataclass                                                                                                                                                                        
class HLLMConfig:                                                                                                                                                                 
    # ==== 路径 ====                                                                                                                                                              
    dataset_name: str = "amazon_books"        # 数据文件名（不含 .csv）                                                                                                           
    data_path: str = "./dataset"              # 交互数据目录                                                                                                                      
    text_path: str = "./information"          # 物品文本目录（绝对路径）                                                                                                          
    item_pretrain_dir: str = "./TinyLlama-1.1B-Chat-v1.0"                                                                                                                         
    user_pretrain_dir: str = "./TinyLlama-1.1B-Chat-v1.0"                                                                                                                         
    checkpoint_dir: str = "./checkpoints"                                                                                                                                         
                                                                                                                                                                                
    # ==== 数据参数 ====                                                                                                                                                          
    MAX_TEXT_LENGTH: int = 256                # 物品文本最大 token 数                                                                                                             
    MAX_ITEM_LIST_LENGTH: int = 50            # 用户序列最大长度                                                                                                                  
    text_keys: List[str] = field(default_factory=lambda: ["title", "description"])                                                                                                
    item_prompt: str = "Compress the following sentence into embedding: "    #提示词                                                                                                 
    item_emb_token_n: int = 1                 # item embedding token 数量                                                                                                         
                                                                                                                                                                                
    # ==== 训练参数 ====                                                                                                                                                          
    epochs: int = 5                                                                                                                                                               
    train_batch_size: int = 8                                                                                                                                                     
    eval_batch_size: int = 256
    learning_rate: float = 1e-4                                                                                                                                                   
    weight_decay: float = 0.01                                                                                                                                                    
    warmup_ratio: float = 0.1                                                                                                                                                     
    max_grad_norm: float = 1.0                                                                                                                                                    
                                                                                                                                                                                
    # ==== NCE loss 参数 ====                                                                                                                                                     
    nce_thres: float = 0.99                                                                                                                                                       
    num_negatives: int = 512                                                                                                                                                      
                                                                                                                                                                                
    # ==== 评估 ====                                                                                                                                                              
    topk: List[int] = field(default_factory=lambda: [5, 10, 50, 200])                                                                                                             
                                                                                                                                                                                
    # ==== 分布式 ====                                                                                                                                                            
    local_rank: int = 0                                                                                                                                                           
    world_size: int = 1                                                                                                                                                           
                                                                                                                                                                                
    # ==== 其他 ====                                                                                                                                                              
    seed: int = 2020                                                                                                                                                              
    gradient_checkpointing: bool = True     # 省显存                                                                                                                              
    use_ft_flash_attn: bool = True                                                                                                                                                
    precision: str = "bf16"                                                                                                                                                       
    val_only: bool = False     