import numpy as np
import torch
# 导入通过 g++ 编译生成的 C++ 模块
try:
    import mcts_fast
except ImportError:
    print("错误：未找到 mcts_fast 模块。请先编译 mcts_fast.cpp 生成 .so 文件。")

class BitBoard:
    def __init__(self, width=15, height=15, n_in_row=5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.init_board()

    def init_board(self):
        self.bitboards = {1: 0, 2: 0} # 使用 Python 大整数作为位棋盘
        self.current_player = 1
        self.move_count = 0
        self.last_move = -1

    def do_move(self, move):
        self.bitboards[self.current_player] |= (1 << move)
        self.last_move = move
        self.current_player = 3 - self.current_player
        self.move_count += 1

    def undo_move(self, move):
        self.current_player = 3 - self.current_player
        self.bitboards[self.current_player] &= ~(1 << move)
        self.move_count -= 1

    def game_end(self):
        """高效位运算判断：支持 15x15 棋盘五连珠判断"""
        w = self.width
        for p, b in self.bitboards.items():
            # 1. 横向判断
            v = b & (b >> 1)
            v &= (v >> 2)
            if v & (v >> 1): return True, p
            # 2. 纵向判断
            v = b & (b >> w)
            v &= (v >> (2 * w))
            if v & (v >> w): return True, p
            # 3. 斜下判断 (\)
            v = b & (b >> (w + 1))
            v &= (v >> (2 * (w + 1)))
            if v & (v >> (w + 1)): return True, p
            # 4. 斜上判断 (/)
            v = b & (b >> (w - 1))
            v &= (v >> (2 * (w - 1)))
            if v & (v >> (w - 1)): return True, p
            
        if self.move_count == self.width * self.height:
            return True, -1 # 平局
        return False, -1

    def current_state(self):
        """返回 4x15x15 的特征图供神经网络使用"""
        square_state = np.zeros((4, self.width, self.height))
        if self.move_count > 0:
            # 这里的逻辑与你之前的 model 保持一致
            # 0层：当前玩家棋子；1层：对手棋子；2层：最后落子位置；3层：当前玩家标识
            p1 = self.bitboards[self.current_player]
            p2 = self.bitboards[3 - self.current_player]
            for i in range(self.width * self.height):
                r, c = i // self.width, i % self.width
                if (p1 >> i) & 1: square_state[0, r, c] = 1.0
                if (p2 >> i) & 1: square_state[1, r, c] = 1.0
            if self.last_move != -1:
                square_state[2, self.last_move // self.width, self.last_move % self.width] = 1.0
        if self.current_player == 1:
            square_state[3, :, :] = 1.0
        return square_state

# --- 关键修改：将 MCTS 接口重定向到 C++ 模块 ---
# 现在 train.py 中调用 MCTS 时，实际上是在调用 C++ 编写的高速引擎
MCTS = mcts_fast.MCTS