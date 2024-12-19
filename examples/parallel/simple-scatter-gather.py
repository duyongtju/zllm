import torch
import torch.nn as nn
import torch.distributed as dist
import os

def main():
    world_size = torch.cuda.device_count()
    local_rank = os.environ.get("LOCAL_RANK")
    if not local_rank:
        local_rank = 0
    device = f'cuda:{local_rank}'
    dist.init_process_group(backend='nccl')
    
    tensor_size = 2

    output_tensor = torch.zeros(tensor_size, device=device)
    
    if dist.get_rank() == 0:
        scatter_list = [torch.ones(tensor_size, device=device) * (i+1) for i in range(world_size)]
        # t_ones = torch.ones(tensor_size, device=device)
        # t_fives = torch.ones(tensor_size, device=device) * 5
        
        # scatter_list = [t_ones, t_fives]
    else:
        scatter_list = None

    dist.scatter(output_tensor, scatter_list, src=0)

    print(f'local rank: {local_rank}', output_tensor)

    output_tensor += 1

    if dist.get_rank() == 0:
        scatter_list = [torch.ones(tensor_size, device=device) for _ in range(world_size)]
        # t_ones1 = torch.ones(tensor_size, device=device)
        # t_ones2 = torch.ones(tensor_size, device=device)
        # scatter_list = [t_ones1, t_ones2]
    else:
        scatter_list = None
    
    dist.gather(output_tensor, scatter_list, dst=0)
    if dist.get_rank() == 0:
        print(scatter_list)

if __name__ == "__main__":
    main()

