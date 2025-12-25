# Resnet-Gomoku-15x15

**Expert-level Gomoku AI based on AlphaZero, optimized for 15x15 board with C++ MCTS Engine.**
**基于 AlphaZero 算法的专家级五子棋 AI，针对 15x15 棋盘及 C++ MCTS 引擎进行了深度优化。**

---

## 🌟 Features | 项目特性

* **Deep Residual Learning**: Utilizes ResNet architecture for policy and value estimation.
* **High-Performance C++ Engine**: MCTS logic implemented in C++ with `pybind11` for maximum simulation speed.
* **Hybrid Board Logic**: Robust matrix-based board management to handle 15x15 state space without overflow.
* **RTX 5090 Optimized**: Fully supports CUDA acceleration for rapid training and self-play.

---

## 🚀 Getting Started | 快速开始

### 1. Prerequisites | 环境要求
* Python 3.10+
* PyTorch (with CUDA support)
* G++ (supporting C++17)
* Pybind11

### 2. Compilation | 编译 C++ 引擎
To enable the high-speed MCTS, compile the C++ module on your target machine:
为了开启高速 MCTS，请在目标机器上编译 C++ 模块：

```Bash
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) mcts_fast.cpp -o mcts_fast$(python3-config --extension-suffix)
```


### 3. Training | 启动训练

Start the self-play training process: 启动自我对弈训练流程：

```Bash
python3 train.py
```

## 🛠 Project Structure | 项目结构
* train.py: Main entry for self-play and neural network optimization.

* mcts_pure.py: Python wrapper and board logic for the game.

* mcts_fast.cpp: Core MCTS simulation engine written in C++.

* model.py: ResNet architecture definitions.

## 📈 Training Progress | 训练进展
Currently training on NVIDIA RTX 5090. The model is capable of generating expert-level moves on a 15x15 board through continuous self-play. 当前正在 NVIDIA RTX 5090 上进行训练。通过持续的自我对弈，模型能够在 15x15 棋盘上产生专家级的落子方案。

## 📄 License | 开源协议
This project is licensed under the Apache-2.0 License. 本项目遵循 Apache-2.0 开源协议。
