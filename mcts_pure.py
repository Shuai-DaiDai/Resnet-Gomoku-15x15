import numpy as np
import torch
import copy  # 必须导入用于深度克隆

try:
    import mcts_fast
except ImportError:
    print("提示：未找到 mcts_fast 模块。")

class BitBoard:
    def __init__(self, width=15, height=15, n_in_row=5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.init_board()

    def init_board(self):
        self.bitboards = {1: 0, 2: 0} 
        self.current_player = 1
        self.move_count = 0
        self.last_move = -1

    def copy(self):
        """深度克隆：这是 C++ 模拟成功的核心接口"""
        new_board = BitBoard(self.width, self.height, self.n_in_row)
        new_board.bitboards = self.bitboards.copy()
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        new_board.last_move = self.last_move
        return new_board

    @property
    def availables(self):
        """核心接口：确保 C++ 永远能拿到可落子位置"""
        occupied = self.bitboards[1] | self.bitboards[2]
        # 遍历 225 个位置，检查哪些位是 0（空位）
        return [i for i in range(self.width * self.height) if not (occupied >> i) & 1]

    def do_move(self, move):
        self.bitboards[self.current_player] |= (1 << move)
        self.last_move = move
        self.current_player = 3 - self.current_player
        self.move_count += 1

    def game_end(self):
        """位运算五连珠判断逻辑"""
        w = self.width
        for p, b in self.bitboards.items():
            for shift in [1, w, w + 1, w - 1]:
                v = b & (b >> shift)
                v &= (v >> (2 * shift))
                if v & (v >> shift): return True, p
        if self.move_count == w * self.height:
            return True, -1
        return False, -1

    def current_state(self):
        """神经网络输入特征图构造"""
        square_state = np.zeros((4, self.width, self.height))
        p1, p2 = self.bitboards[self.current_player], self.bitboards[3 - self.current_player]
        for i in range(self.width * self.height):
            r, c = i // self.width, i % self.width
            if (p1 >> i) & 1: square_state[0, r, c] = 1.0
            if (p2 >> i) & 1: square_state[1, r, c] = 1.0
        if self.last_move != -1:
            square_state[2, self.last_move // self.width, self.last_move % self.width] = 1.0
        if self.current_player == 1: square_state[3, :, :] = 1.0
        return square_state

MCTS = mcts_fast.MCTS