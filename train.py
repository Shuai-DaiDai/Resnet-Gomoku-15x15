import torch
import torch.multiprocessing as mp
import multiprocessing
import resource

# --- 必须在所有 torch.cuda 操作之前调用 ---
if __name__ == '__main__':
    # 2. 提升系统句柄上限，解决 "Too many open files"
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, hard))
    # 1. 强制设置启动模式，防止 CUDA 冲突
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass



from collections import deque
import torch.nn.functional as F
import random
import os
import time
import gc
import numpy as np
from model import Net
from mcts_pure import BitBoard, MCTS

# 1. 基础设置：彻底缓解句柄压力
mp.set_sharing_strategy('file_system')

def get_equi_data(play_data, width, height):
    extend_data = []
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(mcts_prob.reshape(height, width), i)
            extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
    return extend_data

def worker_loop(width, height, n_in_row, n_playout, input_queue, output_queue,dev):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_net = Net(width, height, n_res_blocks=40).to(dev)
    """持久化工人：启动后常驻显存，避免反复创建进程产生的句柄堆积"""
    # 每个进程内部初始化自己的网络副本
    local_net = Net(width, height, n_res_blocks=40).to(dev)
    local_net.eval()
    
    while True:
        # 1. 获取主进程发来的最新权重
        weights = input_queue.get()
        if weights is None: break # 退出信号
        local_net.load_state_dict(weights)
        
        # 2. 进行一局自我对弈
        play_data = run_self_play(width, height, n_in_row, n_playout, local_net, dev)
        
        # 3. 将数据送回主进程
        output_queue.put(play_data)

def run_self_play(width, height, n_in_row, n_playout, net, dev):
    """单局对弈逻辑，包含报错修复的归一化"""
    def policy_fn(b):
        state = torch.FloatTensor(b.current_state()).unsqueeze(0).to(dev)
        with torch.no_grad():
            log_p, v = net(state)
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

        # 核心修复：强制高精度归一化，防止 ValueError
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
    # --- 参数区 (5090 极致压榨版) ---
    width, height = 15, 15
    n_in_row = 5
    n_playout = 2000
    num_workers = 8        # 5090 建议 6-8 个工人
    batch_size = 6144      # 5090 甜点位 Batch
    buffer_maxlen = 100000
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('./models'): os.makedirs('./models')
    
    net = Net(width, height, n_res_blocks=40).to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    buffer = deque(maxlen=buffer_maxlen)

    # --- 启动持久化工进程 ---
    input_queues = [mp.Queue(maxsize=1) for _ in range(num_workers)]
    output_queue = mp.Queue()
    
    for j in range(num_workers):
        p = mp.Process(target=worker_loop, args=(width, height, n_in_row, n_playout, input_queues[j], output_queue, dev))
        p.daemon = True
        p.start()
        # 初始权重同步
        w = {k: v.cpu().clone().detach() for k, v in net.state_dict().items()}
        input_queues[j].put(w)

    print("--- 5090 炼丹炉已开启 (持久化进程模式) ---")

    for i in range(100000):
        # 1. 尽力收集产出的数据
        new_data_count = 0
        while not output_queue.empty():
            data = output_queue.get()
            buffer.extend(get_equi_data(data, width, height))
            new_data_count += 1
        
        # 2. 如果 Buffer 够了，进行高强度训练
        if len(buffer) >= batch_size:
            net.train()
            batch = random.sample(buffer, batch_size)
            s_b, p_b, z_b = zip(*batch)
            
            s_t = torch.FloatTensor(np.array(s_b)).to(dev)
            p_t = torch.FloatTensor(np.array(p_b)).to(dev)
            z_t = torch.FloatTensor(np.array(z_b)).to(dev)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                lp, v = net(s_t)
                loss = -torch.mean(torch.sum(p_t * lp, dim=1)) + F.mse_loss(v.view(-1), z_t)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 权重更新后，通知工人领新权重 (每 5 轮同步一次即可)
            if i % 5 == 0:
                current_w = {k: v.cpu().clone().detach() for k, v in net.state_dict().items()}
                for q in input_queues:
                    if q.empty(): # 只有工人闲着才塞新权重，不堵塞队列
                        q.put(current_w)

            if (i+1) % 10 == 0:
                print(f"轮次 {i+1}, Buffer: {len(buffer)}, Loss: {loss.item():.4f}")
        else:
            time.sleep(1) # Buffer 不足，强制冷静
            
        # 3. 存档与清理
        if (i + 1) % 100 == 0:
            torch.save(net.state_dict(), './models/gomoku_latest.pth')
            gc.collect()
            
if __name__ == '__main__':
    train()