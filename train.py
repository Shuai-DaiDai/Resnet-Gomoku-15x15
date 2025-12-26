from typing import Any
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from model import Net, device # 确保云端 model.py 的 device 为 "cuda"
from mcts_pure import BitBoard, MCTS
import multiprocessing as mp
import resource
import time # 在文件顶端添加

def get_equi_data(play_data, width, height):
    """数据增强：通过旋转和翻转，将1局棋的数据量提升8倍"""
    extend_data = []
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            # 旋转镜像操作
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(mcts_prob.reshape(height, width), i)
            extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
            # 水平翻转操作
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
    return extend_data

def collect_self_play_data(width, height, n_in_row, n_playout, net_weights, device):
    """子进程工人：运行一局自我对弈并返回数据"""
    # 局部初始化，确保每个进程有自己的资源空间
    local_net = Net(width, height, n_res_blocks=40).to(device)
    local_net.load_state_dict(net_weights)
    local_net.eval()
    
    def policy_fn(b):
        state = torch.FloatTensor(b.current_state()).unsqueeze(0).to(device)
        with torch.no_grad():
            log_p, v = local_net(state)
        probs = np.exp(log_p.cpu().numpy().flatten())
        return zip(b.availables, probs[b.availables]), v.item()

    board = BitBoard(width, height, n_in_row=n_in_row)
    mcts = MCTS(policy_fn, n_playout)
    states, probs, players = [], [], []
    step_count = 0
    
    while True:
        temp = 1.0 if step_count < 30 else 1e-3
        acts, p = mcts.get_move_probs(board, temp)
        
        full_p = np.zeros(width * height)
        full_p[list(acts)] = p
        states.append(board.current_state())
        probs.append(full_p)
        players.append(board.current_player)

        # 强制转换为 float64 并重新归一化，确保严丝合缝等于 1，消除精度报错
        p = np.array(p).astype('float64')
        p /= p.sum()    
        move = np.random.choice(acts, p=p)
        board.do_move(move)
        step_count += 1
        
        end, winner = board.game_end()              
        if end:
            z = np.zeros(len(players))
            if winner != -1:
                z[np.array(players) == winner] = 1.0
                z[np.array(players) != winner] = -1.0
            return list(zip(states, probs, z))

def train():
    # --- 15x15 配置区 ---
    width, height, n_in_row = 15, 15, 5
    # 5090 显存够大，直接上 20 层残差块提取深层逻辑
    net = Net(width, height, n_res_blocks=40).to(device)
    net = torch.compile(net) # 只需要加这一行
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    model_file = './models/gomoku_latest.pth'
    num_workers = 8  # 5090 性能强劲，建议开 6 个并行下棋进程
    pool = mp.Pool(processes=num_workers)
    data_tasks = []

    if os.path.exists(model_file):
        print(f"--- 发现预训练模型 {model_file}，正在加载并继续训练 ---")
        net.load_state_dict(torch.load(model_file, map_location=device))
    else:
        print("--- 未发现存档文件，将从随机初始化开始训练 ---")
    buffer = deque(maxlen=200000) # 增大经验池容量
    
    # 5090 硬件加速：混合精度缩放器
    scaler = torch.amp.GradScaler('cuda')

    # 确保保存目录存在
    if not os.path.exists('./models'): os.makedirs('./models')

    print(f"RTX 5090 专家级训练启动，设备: {device}, 棋盘: 15x15")
    
    for i in range(100000): # 15x15 建议起步 20000 轮
        # --- A. 派发新任务 ---
        # 如果当前运行的任务少于设定的并行数，则补齐任务
        # 只有当旧任务全部完成后，才开启新一轮并行下棋
        # 将 if len(data_tasks) == 0: 改为 while 循环
        while len(data_tasks) < num_workers:
            weights_cpu = {k: v.cpu() for k, v in net.state_dict().items()}
            task = pool.apply_async(collect_self_play_data, 
                           args=(width, height, n_in_row, 2000, weights_cpu, device))
            data_tasks.append(task)

        # --- B. 检查并收集已完成的任务数据 ---
        for task in data_tasks[:]:
            new_data_received = False # 在 B 循环开始前初始化
            if task.ready():
                curr_play_data = task.get()
                # 依然保持数据增强逻辑
                enhanced_data = get_equi_data(curr_play_data, width, height)
                buffer.extend(enhanced_data)
                data_tasks.remove(task)
                new_data_received = True # 核心：标记收到了新棋谱
        
        # --- C. 神经网络参数更新 ---
        if len(buffer) > 6144:
            net.train()
            # 5090 核心：直接开启 512 或 1024 大 Batch 训练
            batch = random.sample(buffer, 6144)
            s_b, p_b, z_b = zip(*batch)
            s_t = torch.FloatTensor(np.array(s_b)).to(device)
            p_t = torch.FloatTensor(np.array(p_b)).to(device)
            z_t = torch.FloatTensor(np.array(z_b)).to(device)
            
            optimizer.zero_grad()
            
        else:
            # 【关键修复】如果 Buffer 不够且没有新数据进来，休眠 1 秒，防止空转
            if not new_data_received:
                time.sleep(1)
                continue # 跳过本次循环，不触发下面的 i 计数和存盘

            # 混合精度上下文，自动分配 5090 算力
            with torch.amp.autocast('cuda'):
                lp, v = net(s_t)
                loss = -torch.mean(torch.sum(p_t * lp, dim=1)) + F.mse_loss(v.view(-1), z_t)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # --- 增加这一行，防止主进程跑太快导致句柄堆积 ---
            time.sleep(0.01)
            
            if (i+1) % 10 == 0:
                print(f"轮次 {i+1}, Buffer大小: {len(buffer)}, 损失Loss: {loss.item():.4f}")

        # --- 存盘逻辑：适配云端路径 ---
        if (i + 1) % 100 == 0:
            save_path = f'./models/gomoku_15x15_{i+1}.pth'
            torch.save(net.state_dict(), save_path)
            # 同时更新一个最新版
            torch.save(net.state_dict(), './models/gomoku_latest.pth')
            print(f"--- 存档完成: {save_path} ---")

if __name__ == "__main__":
    # 1. 提升系统允许同时打开的文件句柄数，解决 "Too many open files" 报错
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
    
    # 2. 强制设置进程启动模式为 spawn，这是 CUDA 多进程的硬性要求
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    # 3. 启动主训练逻辑
    train()