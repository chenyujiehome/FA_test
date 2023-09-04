import pandas as pd
import torch
import time
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# 检查是否有可用的GPU并设为默认设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 从Excel文件中读取数据
df = pd.read_excel('test_dim_data.xlsx', engine='openpyxl')

# 创建一个新的数据框用于保存运行时间
results = {
    'batchsize': [],
    'nheads': [],
    'headdim': [],
    'seqlen': [],
    'average_time': [],
    'all_times': []
}

# 遍历每一行数据
for _, row in df.iterrows():
    batch_size = int(row['batchsize'])
    nheads = int(row['nheads'])
    headdim = int(row['headdim'])
    seqlen = int(row['seqlen'])
    
    times = []

    for _ in range(100):
        # 创建张量并移动到GPU上，并转换为fp16
        q = torch.randn(batch_size, seqlen, nheads, headdim).to(device).half()
        k = torch.randn(batch_size, seqlen, nheads, headdim).to(device).half()
        v = torch.randn(batch_size, seqlen, nheads, headdim).to(device).half()

        start_time = time.time()
        flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
        end_time = time.time()
        times.append(end_time - start_time)
    
    average_time = sum(times) / 100

    # 保存结果
    results['batchsize'].append(batch_size)
    results['nheads'].append(nheads)
    results['headdim'].append(headdim)
    results['seqlen'].append(seqlen)
    results['average_time'].append(average_time)
    results['all_times'].append(times)

# 将结果保存到新的Excel文件中
results_df = pd.DataFrame(results)
results_df.to_excel('test_dim_data_results.xlsx', index=False)
