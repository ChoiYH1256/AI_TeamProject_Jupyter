import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import intel_extension_for_pytorch as ipex
import time
import gc

# DDP 초기화 함수
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# DDP 종료 함수
def cleanup():
    dist.destroy_process_group()

# 모델 생성 함수
def create_model(num_classes=91):
    model = torchvision.models.detection.retinanet_resnext50_32x4d_fpn(weights='ResNeXt50_32X4D_Weights.DEFAULT')
    model = ipex.optimize(model, dtype=torch.bfloat16)  # Intel CPU 최적화
    return model

# 학습 함수
def train(rank, world_size, config):
    setup(rank, world_size)
    
    # 모델과 DDP 설정
    model = create_model(num_classes=91).to(rank)
    model = DDP(model, device_ids=[rank])

    # 데이터셋과 데이터로더 설정
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, 
                              num_workers=config['worker'], collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['Adam_lr'])
    
    # 학습 루프
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(rank) for img in images]
            targets = [{k: v.to(rank) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
        
        print(f"Rank {rank}, Epoch {epoch} Loss: {total_loss / len(train_loader)}")
    
    cleanup()

# 메인 함수
def main():
    world_size = 2  # 사용할 프로세스 개수 설정
    config = {
        'batch_size': 4,
        'worker': 8,
        'epochs': 1,
        'Adam_lr': 0.001,
    }

    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
