import os
from decimal import Decimal
import torch
from torch_geometric.loader import DataListLoader, DataLoader
import time

# Function to get memory usage
def get_memory_usage(device, gpu_id):
    if device == 'cpu':
        print('CPU memory usage not supported')
        raise NotImplementedError
    else:
        result = os.popen(f'nvidia-smi -i {gpu_id} | grep MiB').readlines()
        result = result[0].split()
        mem_usage = result[8]
        max_mem = result[10]
        mem_usage = mem_usage[:-3]
        max_mem = max_mem[:-3]
        mem_usage = float(Decimal(mem_usage))*1.048576/1000
        max_mem = float(Decimal(max_mem))*1.048576/1000
        return mem_usage, max_mem

def find_batch_size(_model, device, gpu_id, on_disk_data):
    
    mem_usage, mem_capacity = get_memory_usage(device, gpu_id)
    print(f'Memory capacity: {mem_capacity}')
    
    batch_size = 64
    num_trys = 0
    while True:
        num_trys += 1            
        mem_usage_max = 0
        model = _model
        model.to(device)
        model.train()
        loader = DataLoader(on_disk_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            
        iter = 0
        for data in loader:
            iter += 1
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            time.sleep(1)
            mem_usage, _ = get_memory_usage(device, gpu_id)
            
            if iter == 1:
                break

        #if mem_usage > mem_usage_max:
        mem_usage_max = mem_usage
        # Remove the model from gpu
        data = data.detach()
        out = out.detach()
        model = model.cpu()

        if mem_usage_max*1.06 >= mem_capacity:
            break
        else:
            while mem_usage_max*2.05 < mem_capacity:
                mem_usage_max = mem_usage_max*2
                batch_size = batch_size*2
            while mem_usage_max*1.35 < mem_capacity:
                mem_usage_max = mem_usage_max*1.3
                batch_size = batch_size*1.3
                batch_size = int(batch_size)
            while mem_usage_max*1.15 < mem_capacity:
                mem_usage_max = mem_usage_max*1.1
                batch_size = batch_size*1.1
                batch_size = int(batch_size)
            if num_trys > 2:
                break

        
        del out
        del data
        del model
        del loader
        time.sleep(1)
        torch.cuda.empty_cache()
        
    print(f'Max batch size: {batch_size}', f'Max memory usage: {mem_usage_max}', f'Memory usage (current): {mem_usage}')

    return batch_size
