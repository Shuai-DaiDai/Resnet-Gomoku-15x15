#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <map>
#include <cmath>
#include <vector>
#include <numeric>

namespace py = pybind11;

// 15x15 五子棋位运算胜负判断
bool check_win(const __int128& b, int w) {
    __int128 v = b & (b >> 1); v &= (v >> 2); if (v & (v >> 1)) return true; // 横
    v = b & (b >> w); v &= (v >> (2 * w)); if (v & (v >> w)) return true;    // 纵
    v = b & (b >> (w + 1)); v &= (v >> (2 * (w + 1))); if (v & (v >> (w + 1))) return true; // 斜下
    v = b & (b >> (w - 1)); v &= (v >> (2 * (w - 1))); if (v & (v >> (w - 1))) return true; // 斜上
    return false;
}

// C++ 版本的树节点
struct Node {
    double P;
    double Q = 0, U = 0, W = 0;
    int N = 0;
    std::map<int, Node*> children;
    Node* parent;
    Node(Node* p, double prob) : P(prob), parent(p) {}
};

// C++ 版本的 MCTS 逻辑
class MCTS {
public:
    MCTS(py::function policy_fn, int n_playout) : policy_fn(policy_fn), n_playout(n_playout) {}
    // 搜索逻辑的核心实现...
private:
    py::function policy_fn;
    int n_playout;
};

PYBIND11_MODULE(mcts_fast, m) {
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<py::function, int>())
        .def("get_move_probs", &MCTS::get_move_probs);
}