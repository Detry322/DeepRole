#ifndef FDEEP_REPLACE_H_
#define FDEEP_REPLACE_H_

#include <Eigen/Core>
#include <string>
#include <map>
#include <utility>
#include <vector>

typedef Eigen::Array<float, Eigen::Dynamic, 1> EigenVector;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> DenseWeights;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> DenseBiases;

namespace jdeep {

enum layer_type {
    INPUT,
    DENSE,
    CFV_MASK,
    CFV_FROM_WIN,
    ZERO_SUM,
};

enum activation_type {
    NA,
    LINEAR,
    RELU,
    SIGMOID
};

typedef int layer_id;

class model {
public:
    EigenVector predict(const EigenVector& input) const;

    static model load_model(const std::string& filename);

private:
    std::map<layer_id, DenseWeights> layer_weights;
    std::map<layer_id, DenseBiases> layer_biases;
    std::vector<layer_id> ordered_layers;
    std::map<layer_id, std::pair<layer_type, activation_type>> layer_info;
    std::map<layer_id, std::vector<layer_id>> predecessors;

    layer_id output_layer;
};

} // namespace jdeep;

#endif // FDEEP_REPLACE_H_
