#include "./fdeep_replace.h"

#include "json.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wreturn-std-move"
#include <fdeep/import_model.hpp>
#pragma GCC diagnostic pop

#include <cassert>
#include <fstream>
#include <iostream>

#include "./lookup_tables.h"

namespace jdeep {

static EigenVector calculate_dense_layer(
    const EigenVector& input,
    const DenseWeights& weights,
    const DenseBiases& biases,
    const activation_type activation
) {
    EigenVector result = (input.matrix().transpose() * weights + biases.transpose()).transpose().array();

    switch (activation) {
    case RELU:
        result = result.max(0.0);
        break;
    case SIGMOID:
        result = (1.0 / (1.0 + (-result).exp()));
        break;
    default: break;
    }
    return result;
}

static EigenVector calculate_cfv_mask_and_adjust_layer(const EigenVector& inp, const EigenVector& cfvs) {
    EigenVector result(NUM_PLAYERS * NUM_VIEWPOINTS);
    result.setZero();

    bool mask[NUM_PLAYERS * NUM_VIEWPOINTS] = {0};
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        if (inp(i) > 0.0) {
            for (int player = 0; player < NUM_PLAYERS; player++) {
                mask[NUM_VIEWPOINTS * player + ASSIGNMENT_TO_VIEWPOINT[i][player]] = true;
            }
        }   
    }

    int num_left = 0;
    for (int i = 0; i < NUM_PLAYERS * NUM_VIEWPOINTS; i++) {
        num_left += (int) (mask[i]);
    }

    float masked_sum = 0.0;
    for (int i = 0; i < NUM_PLAYERS * NUM_VIEWPOINTS; i++) {
        if (mask[i]) {
            masked_sum += cfvs(i);
        }
    }

    float subtract_amount = masked_sum / num_left;
    for (int i = 0; i < NUM_PLAYERS * NUM_VIEWPOINTS; i++) {
        if (mask[i]) {
            result(i) = cfvs(i) - subtract_amount;
        }
    }

    return result;
};

static EigenVector calculate_cfv_from_win_layer(const EigenVector& input_probs, const EigenVector& win_probs) {
    EigenVector result(NUM_PLAYERS * NUM_VIEWPOINTS);
    result.setZero();

    for (int assignment = 0; assignment < NUM_ASSIGNMENTS; assignment++) {
        float good_expected_payoff = 2 * GOOD_WIN_PAYOFF * win_probs(assignment) - GOOD_WIN_PAYOFF;
        float bad_expected_payoff = good_expected_payoff * EVIL_LOSE_PAYOFF;
        for (int player = 0; player < NUM_PLAYERS; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[assignment][player];
            float payoff = (viewpoint < NUM_GOOD_VIEWPOINTS) ? good_expected_payoff : bad_expected_payoff;
            result(NUM_VIEWPOINTS*player + viewpoint) += input_probs(NUM_PLAYERS + assignment) * payoff;
        }
    }

    return result;
};

static EigenVector calculate_zero_sum_layer(const EigenVector& input) {
    EigenVector result = input;
    result -= input.sum() / (NUM_VIEWPOINTS * NUM_PLAYERS);
    return result;
}


EigenVector model::predict(const EigenVector& input) const {
    std::map<layer_id, EigenVector> layer_results;
    layer_results[this->ordered_layers.front()] = input;

    for (const layer_id id : this->ordered_layers) {
        const layer_type type = this->layer_info.at(id).first;
        if (type == INPUT) continue;

        const activation_type activation = this->layer_info.at(id).second;
        const std::vector<layer_id>& parents = this->predecessors.at(id);

        switch (type) {
        case INPUT: break;
        case DENSE: {
            assert(parents.size() == 1);
            const EigenVector& input = layer_results[parents.front()];
            layer_results[id] = calculate_dense_layer(input, this->layer_weights.at(id), this->layer_biases.at(id), activation);
        } break;
        case CFV_MASK: {
            assert(parents.size() == 2);
            const EigenVector& input_1 = layer_results[parents.front()];
            const EigenVector& input_2 = layer_results[parents.back()];
            layer_results[id] = calculate_cfv_mask_and_adjust_layer(input_1, input_2);
        } break;
        case CFV_FROM_WIN: {
            assert(parents.size() == 2);
            const EigenVector& input_1 = layer_results[parents.front()];
            const EigenVector& input_2 = layer_results[parents.back()];
            layer_results[id] = calculate_cfv_from_win_layer(input_1, input_2);
        } break;
        case ZERO_SUM: {
            assert(parents.size() == 1);
            const EigenVector& input = layer_results[parents.front()];
            layer_results[id] = calculate_zero_sum_layer(input); 
        } break;
        }
    }

    return layer_results[this->output_layer];
};

// private:
//     std::map<layer_id, DenseWeights> layer_weights;
//     std::map<layer_id, DenseBiases> layer_biases;
//     std::vector<layer_id> ordered_layers;
//     std::map<layer_id, std::pair<layer_type, activation_type>> layer_info;
//     std::map<layer_id, std::vector<layer_id>> predecessors;
// };

DenseWeights load_weights(const nlohmann::json& json_data, const int output_size) {
    const std::vector<float> floats = fdeep::internal::decode_floats(json_data);
    assert(floats.size() % output_size == 0);
    const int input_size = floats.size() / output_size;

    DenseWeights result(input_size, output_size);

    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            result(i, j) = floats[i*output_size + j];
        }
    }

    return result;
}

DenseBiases load_biases(const nlohmann::json& json_data, const int output_size) {
    const std::vector<float> floats = fdeep::internal::decode_floats(json_data);

    DenseBiases result(output_size);

    for (int i = 0; i < output_size; i++) {
        result(i) = floats[i];
    }

    return result;
}

void run_test(const nlohmann::json& test_case, const model& test_model) {
    const int input_size = test_case["inputs"][0]["shape"][4];
    const int output_size = test_case["outputs"][0]["shape"][4];

    EigenVector input_vec = load_biases(test_case["inputs"][0]["values"], input_size).array();
    EigenVector output_vec = load_biases(test_case["outputs"][0]["values"], output_size).array();

    EigenVector result = test_model.predict(input_vec);

    if (result.size() != output_size) {
        std::cerr << "Result size: (" << result.rows() << ", " << result.cols() << "). Size: " << result.size() << std::endl;
        std::cerr << "Output size: " << output_size << std::endl;
        throw std::length_error("Test result is not the correct shape");
    }

    for (int i = 0; i < output_size; i++) {
        if (std::abs(result(i) - output_vec(i)) > 0.0001) {
            std::cerr << std::setprecision(16);
            std::cerr << "Result:" << std::endl;
            std::cerr << result.transpose().leftCols(6) << std::endl;
            std::cerr << "Expected:" << std::endl;
            std::cerr << output_vec.transpose().leftCols(6) << std::endl;
            throw std::domain_error("results don't match");
        }
    }
}

model model::load_model(const std::string& filename) {
    std::ifstream in_stream(filename);
    assert(in_stream.good());

    model result;

    std::cerr << "Loading model: " << filename << std::endl;
    nlohmann::json json_data;
    in_stream >> json_data;

    layer_id current_id = 0;

    std::map<std::string, layer_id> name_to_id;
    for (const nlohmann::json& layer : json_data["architecture"]["config"]["layers"]) {
        const std::string layer_class_name = layer["class_name"];
        const std::string layer_name = layer["name"];
        name_to_id[layer_name] = current_id;

        if (layer_class_name == "InputLayer") {
            result.layer_info[current_id] = std::make_pair(INPUT, NA);
        } else if (layer_class_name == "Dense") {
            activation_type activation = NA;
            if (layer["config"]["activation"] == "linear") {
                activation = LINEAR;
            } else if (layer["config"]["activation"] == "relu") {
                activation = RELU;
            } else if (layer["config"]["activation"] == "sigmoid") {
                activation = SIGMOID;
            } else {
                assert(false);
            }
            result.layer_info[current_id] = std::make_pair(DENSE, activation);

            const int output_size = layer["config"]["units"];
            result.layer_weights[current_id] = load_weights(json_data["trainable_params"][layer_name]["weights"], output_size);
            result.layer_biases[current_id] = load_biases(json_data["trainable_params"][layer_name]["bias"], output_size);
        } else if (layer_class_name == "CFVMaskAndAdjustLayer") {
            result.layer_info[current_id] = std::make_pair(CFV_MASK, NA);
        } else if (layer_class_name == "CFVFromWinProbsLayer") {
            result.layer_info[current_id] = std::make_pair(CFV_FROM_WIN, NA);
        } else if (layer_class_name == "ZeroSumLayer") {
            result.layer_info[current_id] = std::make_pair(ZERO_SUM, NA);
        }

        for (const nlohmann::json& inbound_nodes : layer["inbound_nodes"]) {
            for (const nlohmann::json& inbound_layer : inbound_nodes) {
                const std::string layer_name = inbound_layer[0];
                assert(name_to_id.count(layer_name) > 0);
                result.predecessors[current_id].push_back(name_to_id[layer_name]);
            }
        }

        result.ordered_layers.push_back(current_id);
        current_id++;
    }

    result.output_layer = name_to_id.at(json_data["architecture"]["config"]["output_layers"][0][0]);

    int test_num = 1;
    for (const nlohmann::json& test_case : json_data["tests"]) {
        std::cerr << "Running test " << test_num << "..." << std::endl;
        run_test(test_case, result);
    }

    return result;
}

} // namespace jdeep;
