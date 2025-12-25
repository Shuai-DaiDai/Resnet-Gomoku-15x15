#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <map>
#include <cmath>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// 节点结构
struct Node {
    float P;
    float Q = 0, U = 0, W = 0;
    int N = 0;
    std::map<int, Node*> children;
    Node* parent;
    Node(Node* p, float prob) : P(prob), parent(p) {}
    ~Node() { for (auto& pair : children) delete pair.second; }
};

class MCTS {
public:
    MCTS(py::function policy_fn, int n_playout) : policy_fn(policy_fn), n_playout(n_playout) {
        root = new Node(nullptr, 1.0);
    }
    ~MCTS() { delete root; }

    // --- 报错点修复：确保函数在类内声明并实现 ---
    py::tuple get_move_probs(py::object board, float temp) {
        // 15x15 搜索逻辑实现
        // 此处为简化示例，确保接口对接成功
        std::vector<int> acts;
        std::vector<float> probs;
        return py::make_tuple(acts, probs);
    }

private:
    Node* root;
    py::function policy_fn;
    int n_playout;
};

// 绑定模块
PYBIND11_MODULE(mcts_fast, m) {
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<py::function, int>())
        .def("get_move_probs", &MCTS::get_move_probs); // 这样就不会报 "not a member" 了
}