#include "./nn.h"

#include <map>
#include <sstream>
#include <iostream>

#include "eigen_types.h"
#include "lookup_tables.h"
#include "game_constants.h"

std::map<std::tuple<int, int, int>, std::shared_ptr<Model>> model_cache;

static std::string get_model_filename(const std::string& search_dir, const int num_succeeds, const int num_fails, const int propose_count) {
    std::stringstream sstream;
    sstream << ((search_dir.empty()) ? "" : (search_dir + "/")) << num_succeeds << "_" << num_fails << "_" << propose_count << ".json";
    return sstream.str();
}


std::shared_ptr<Model> load_model(const std::string& search_dir, const int num_succeeds, const int num_fails, const int propose_count) {
    auto cache_key = std::make_tuple(num_succeeds, num_fails, propose_count);
    if (model_cache.count(cache_key) != 0) {
        return model_cache[cache_key];
    }

    auto model_filename = get_model_filename(search_dir, num_succeeds, num_fails, propose_count);
    auto model = jdeep::model::load_model(model_filename);
    auto model_ptr = std::make_shared<Model>(num_succeeds, num_fails, propose_count, std::move(model));

    model_cache[cache_key] = model_ptr;

    return model_ptr;
}

void Model::predict(const int proposer, const AssignmentProbs& input_probs, ViewpointVector* output_values) {
    EigenVector input_tensor(65);
    input_tensor.setZero();
    input_tensor(proposer) = 1.0;
    input_tensor.bottomRows<60>() = input_probs.cast<float>();

    EigenVector result = this->model.predict(input_tensor);

    for (int player = 0; player < NUM_PLAYERS; player++) {
        output_values[player] = result.block<NUM_VIEWPOINTS, 1>(NUM_VIEWPOINTS*player, 0).cast<double>();
    }
}


void print_loaded_models(const std::string& search_dir) {
    if (model_cache.size() == 0) {
        std::cerr << "No models loaded." << std::endl;
        return;
    }
    for (const auto& pair : model_cache) {
        const auto& tuple = pair.first;
        std::cerr << get_model_filename(search_dir, std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple)) << std::endl;
    }
}
