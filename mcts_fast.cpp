#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <map>
#include <cmath>
#include <vector>
#include <algorithm>

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
        Node* root = new Node(nullptr, 1.0);
        
        for (int n = 0; n < n_playout; ++n) {
            // 关键：确保 Python 的 BitBoard 类有 copy() 方法
            py::object board_copy = board.attr("copy")();
            Node* node = root;
            
            while (!node->children.empty()) {
                float best_u = -1e9;
                int best_move = -1;
                for (auto const& [move, child] : node->children) {
                    float u = child->Q + 5.0f * child->P * sqrt(node->N + 1) / (1 + child->N);
                    if (u > best_u) { best_u = u; best_move = move; }
                }
                board_copy.attr("do_move")(best_move);
                node = node->children[best_move];
            }

            py::tuple res = policy_fn(board_copy);
            auto action_probs = res[0].cast<std::vector<std::pair<int, float>>>();
            float value = res[1].cast<float>();
            
            bool end = board_copy.attr("game_end")()[0].cast<bool>();
            if (!end) {
                for (auto const& [move, prob] : action_probs) 
                    node->children[move] = new Node(node, prob);
            } else {
                int winner = board_copy.attr("game_end")().cast<py::tuple>()[1].cast<int>();
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
        
        // 安全保障：如果 root 没有孩子，强制从棋盘获取可用动作
        if (root->children.empty()) {
            py::list availables = board.attr("availables");
            for (auto item : availables) {
                acts.push_back(item.cast<int>());
                probs.push_back(1.0f / availables.size());
            }
        } else {
            float sum_n = 0;
            for (auto const& [move, child] : root->children) {
                acts.push_back(move);
                float p = pow(child->N, 1.0/temp);
                probs.push_back(p);
                sum_n += p;
            }
            for (float &p : probs) p /= sum_n;
        }

        delete root;
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