import torch
import numpy as np
import time
from model import Net, device
from mcts_pure import BitBoard, MCTS

def print_board(board_obj):
    # 打印顶部列坐标
    print("\n   " + "  ".join([f"{i}" for i in range(board_obj.width)]))
    print("  +" + "---" * board_obj.width + "+")
    for r in range(board_obj.height):
        # 打印左侧行坐标
        row_str = f"{r} |"
        for c in range(board_obj.width):
            idx = r * board_obj.width + c
            if (board_obj.bitboards[1] >> idx) & 1: row_str += " X " # 玩家
            elif (board_obj.bitboards[2] >> idx) & 1: row_str += " O " # AI
            else: row_str += " . "
        print(row_str + "|")
    print("  +" + "---" * board_obj.width + "+")

def run_game():
    width, height, n_in_row = 6, 6, 4
    board = BitBoard(width, height, n_in_row=n_in_row)
    
    # 注意：这里的 n_res_blocks 必须和训练时 model.py 里的设定完全一致
    # 如果报错 Missing key，请检查这里是否和你的训练代码匹配
    net = Net(width, height, n_res_blocks=10).to(device) 
    
    model_path = '/Users/shuaiyifan/Desktop/model_m1_latest.pth'
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载 10 层强化模型: {model_path}")
    except Exception as e:
        print(f"加载失败({e})，将使用随机权重对弈。")

    net.eval()

    def policy_fn(b):
        state = torch.FloatTensor(b.current_state()).unsqueeze(0).to(device)
        with torch.no_grad():
            log_p, v = net(state)
        probs = np.exp(log_p.cpu().numpy().flatten())
        availables = [i for i in range(width*height) if not ((b.bitboards[1]|b.bitboards[2]) >> i) & 1]
        return zip(availables, probs[availables]), v.item()

    ai_player = MCTS(policy_fn, n_playout=800) # 对弈时增加搜索量，更强

    print("游戏开始！请输入 '行,列' (如 0,3)。输入 'q' 退出。")

    while True:
        print_board(board)
        end, winner = board.game_end()
        if end:
            print(f"游戏结束！胜者: {'人类(X)' if winner==1 else 'AI(O)' if winner==2 else '平局'}")
            break

        if board.current_player == 1:
            while True:
                user_input = input("你的回合 (X) -> ")
                if user_input.lower() == 'q': return
                try:
                    r, c = map(int, user_input.split(','))
                    move = r * width + c
                    if 0 <= r < height and 0 <= c < width and not ((board.bitboards[1]|board.bitboards[2]) >> move) & 1:
                        break
                    else: print("无效位置，请重试。")
                except: print("输入错误！请使用格式: 行,列 (例如: 1,2)")
        else:
            print("AI 正在思考...")
            acts, p = ai_player.get_move_probs(board, temp=1e-3)
            move = acts[np.argmax(p)]
            print(f"AI 落子: {move//width}, {move%width}")

        board.do_move(move)
        if move in ai_player.root.children:
            ai_player.root = ai_player.root.children[move]
            ai_player.root.parent = None
        else: ai_player.root = Node(None, 1.0)

if __name__ == "__main__":
    run_game()