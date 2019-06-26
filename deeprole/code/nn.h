#ifndef NN_H_
#define NN_H_

#include <memory>

#include "./fdeep_replace.h"

#include "eigen_types.h"

struct Model {
    int num_succeeds;
    int num_fails;
    int propose_count;
    jdeep::model model;

    Model(int num_succeeds, int num_fails, int propose_count, jdeep::model model) :
        num_succeeds(num_succeeds),
        num_fails(num_fails),
        propose_count(propose_count),
        model(std::move(model)) {}

    void predict(const int proposer, const AssignmentProbs& input_probs, ViewpointVector* output_values);
};

std::shared_ptr<Model> load_model(const std::string& search_dir, const int num_succeeds, const int num_fails, const int propose_count);

void print_loaded_models(const std::string& search_dir);

#endif // NN_H_
