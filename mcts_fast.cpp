#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <map>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>

namespace py = pybind11;

struct Node {
    float P;
    float Q = 0, W = 0;
    int N = 0;
    std::map<int, Node*> children;
    Node* parent;
    Node(Node* p, float prob) : P(prob), parent(p) {}
    ~Node() { for (auto& pair : children) delete pair.second; }
};

class MCTS {
public:
    MCTS(py::function policy_fn, int n_playout) : policy_fn(policy_fn), n_playout(n_playout) {}

    py::tuple get_move_probs(py::object board, float temp) {
        std::unique_ptr<Node> root_ptr(new Node(nullptr, 1.0));
        Node* root = root_ptr.get();
        
        for (int n = 0; n < n_playout; ++n) {
            py::object board_copy = board.attr("copy")();
            Node* node = root;
            
            while (!node->children.empty()) {
                float best_u = -1e9;
                int best_move = -1;
                for (auto const& [move, child] : node->children) {
                    float u = child->Q + 5.0f * child->P * sqrt(node->N + 1) / (1 + child->N);
                    if (u > best_u) { best_u = u; best_move = move; }
                }
                if (best_move == -1) break;
                board_copy.attr("do_move")(best_move);
                node = node->children[best_move];
            }

            py::tuple res = policy_fn(board_copy);
            auto action_probs = res[0].cast<std::vector<std::pair<int, float>>>();
            float value = res[1].cast<float>();
            
            py::tuple end_res = board_copy.attr("game_end")().cast<py::tuple>();
            if (!end_res[0].cast<bool>()) {
                for (auto const& [move, prob] : action_probs) 
                    node->children[move] = new Node(node, prob);
            } else {
                int winner = end_res[1].cast<int>();
                value = (winner == -1) ? 0.0f : -1.0f;
            }

            while (node) {
                node->N++;
                node->W += value;
                node->Q = node->W / node->N;
                value = -value;
                node = node->parent;
            }
        }

        std::vector<int> acts;
        std::vector<float> probs;
        
        if (root->children.empty()) {
            py::list avail = board.attr("availables");
            for (auto m : avail) {
                acts.push_back(m.cast<int>());
                probs.push_back(1.0f / avail.size());
            }
        } else {
            // 数值稳定性保护：处理 temp 趋于 0 的情况
            if (temp < 1e-3) {
                int best_move = -1; int max_n = -1;
                for (auto const& [move, child] : root->children) {
                    if (child->N > max_n) { max_n = child->N; best_move = move; }
                }
                acts.push_back(best_move);
                probs.push_back(1.0f);
            } else {
                float sum_p = 0;
                // 找出最大 N 用于平滑
                int max_n = 0;
                for (auto const& [move, child] : root->children) max_n = std::max(max_n, child->N);

                for (auto const& [move, child] : root->children) {
                    acts.push_back(move);
                    // 核心修复：增加极小值偏移，并防止指数过大
                    float p = pow((float)child->N + 1e-8f, 1.0f / temp);
                    probs.push_back(p);
                    sum_p += p;
                }
                // 归一化，带上 epsilon 防止除以 0 产生 NaN
                for (float &p : probs) p /= (sum_p + 1e-10f);
            }
        }
        return py::make_tuple(acts, probs);
    }

private:
    py::function policy_fn;
    int n_playout;
};

PYBIND11_MODULE(mcts_fast, m) {
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<py::function, int>())
        .def("get_move_probs", &MCTS::get_move_probs);
}