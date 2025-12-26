from typing import Any
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from model import Net, device # 确保云端 model.py 的 device 为 "cuda"
from mcts_pure import BitBoard, MCTS

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

def train():
    # --- 15x15 配置区 ---
    width, height, n_in_row = 15, 15, 5
    # 5090 显存够大，直接上 20 层残差块提取深层逻辑
    net = Net(width, height, n_res_blocks=40).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    model_file = './models/gomoku_latest.pth'
    if os.path.exists(model_file):
        print(f"--- 发现预训练模型 {model_file}，正在加载并继续训练 ---")
        net.load_state_dict(torch.load(model_file, map_location=device))
    else:
        print("--- 未发现存档文件，将从随机初始化开始训练 ---")
    buffer = deque(maxlen=10000) # 增大经验池容量
    
    # 5090 硬件加速：混合精度缩放器
    scaler = torch.amp.GradScaler('cuda')

    # 确保保存目录存在
    if not os.path.exists('./models'): os.makedirs('./models')

    print(f"RTX 5090 专家级训练启动，设备: {device}, 棋盘: 15x15")
    
    for i in range(20000): # 15x15 建议起步 20000 轮
        board = BitBoard(width, height, n_in_row=n_in_row)
        # 自我对弈逻辑定义
        def policy_fn(b):
            state = torch.FloatTensor(b.current_state()).unsqueeze(0).to(device)
            net.eval()
            with torch.no_grad():
                log_p, v = net(state)
            probs = np.exp(log_p.cpu().numpy().flatten())
            availables = b.availables 
            return zip(availables, probs[availables]), v.item()

        mcts = MCTS(policy_fn, 2000)# 自我对弈搜索量，5090可适当调高
        play_data = [] # 暂存本局数据
        states, probs, players = [], [], []
        step_count = 0
        
        while True:
            # 探索与收敛策略
            temp = 1.0 if step_count < 30 else 1e-3
            acts, p = mcts.get_move_probs(board, temp)
            
            p = np.array(p).astype('float64')
            p /= np.sum(p)

            full_p = np.zeros(width*height)
            full_p[list(acts)] = p
            states.append(board.current_state())
            probs.append(full_p)
            players.append(board.current_player)
                
            move = np.random.choice(acts, p=p)
            board.do_move(move)
            step_count += 1
            
            end, winner = board.game_end()              
            if end:
                z = np.zeros(len(players))
                if winner != -1:
                    z[np.array(players) == winner] = 1.0
                    z[np.array(players) != winner] = -1.0
                
                # 进行数据增强并存入 Buffer
                curr_play_data = list(zip(states, probs, z))
                enhanced_data = get_equi_data(curr_play_data, width, height)
                buffer.extend(enhanced_data)
                break
        
        # --- 神经网络参数更新 ---
        if len(buffer) > 2048:
            net.train()
            # 5090 核心：直接开启 512 或 1024 大 Batch 训练
            batch = random.sample(buffer, 2048)
            s_b, p_b, z_b = zip(*batch)
            s_t = torch.FloatTensor(np.array(s_b)).to(device)
            p_t = torch.FloatTensor(np.array(p_b)).to(device)
            z_t = torch.FloatTensor(np.array(z_b)).to(device)
            
            optimizer.zero_grad()
            
            # 混合精度上下文，自动分配 5090 算力
            with torch.amp.autocast('cuda'):
                lp, v = net(s_t)
                loss = -torch.mean(torch.sum(p_t * lp, dim=1)) + F.mse_loss(v.view(-1), z_t)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
    train()