import numpy as np
import copy

try:
    import mcts_fast
except ImportError:
    print("错误：未找到 mcts_fast 模块，请先编译 C++ 文件。")

class BitBoard:
    def __init__(self, width=15, height=15, n_in_row=5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.init_board()

    def init_board(self):
        # 矩阵模式：0-空，1-黑子，2-白子
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.current_player = 1
        self.move_count = 0
        self.last_move = -1

    def copy(self):
        """深度克隆供 C++ MCTS 模拟使用"""
        new_board = BitBoard(self.width, self.height, self.n_in_row)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        new_board.last_move = self.last_move
        return new_board

    @property
    def availables(self):
        """获取当前所有空位索引，返回 List[int]"""
        return np.where(self.board.flatten() == 0)[0].tolist()

    def do_move(self, move):
        r, c = move // self.width, move % self.width
        self.board[r, c] = self.current_player
        self.last_move = move
        self.current_player = 3 - self.current_player
        self.move_count += 1

    def game_end(self):
        """矩阵模式下的五连珠检测"""
        for r in range(self.height):
            for c in range(self.width):
                if self.board[r, c] == 0: continue
                p = self.board[r, c]
                # 检查 横、纵、斜下、斜上 四个方向
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count = 1
                    for i in range(1, 5):
                        nr, nc = r + dr*i, c + dc*i
                        if 0 <= nr < self.height and 0 <= nc < self.width and self.board[nr, nc] == p:
                            count += 1
                        else: break
                    if count >= 5: return True, p
        if self.move_count == self.width * self.height: return True, -1
        return False, -1

    def current_state(self):
        """构造 4x15x15 特征图"""
        state = np.zeros((4, self.height, self.width))
        state[0] = (self.board == self.current_player).astype(float)
        state[1] = (self.board == (3 - self.current_player)).astype(float)
        if self.last_move != -1:
            state[2, self.last_move // self.width, self.last_move % self.width] = 1.0
        if self.current_player == 1: state[3] = 1.0
        return state

# 接口重定向到 C++ 模块
MCTS = mcts_fast.MCTS