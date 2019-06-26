#ifndef SERIALIZATION_H_
#define SERIALIZATION_H_

#include <iostream>

#include "./lookahead.h"

void json_deserialize_starting_reach_probs(std::istream& in_stream, AssignmentProbs* starting_reach_probs);
void json_serialize_lookahead(const LookaheadNode* root, const AssignmentProbs& starting_reach_probs, std::ostream& out_stream);

template <typename Derived>
inline std::vector<std::vector<double>> eigen_to_double_vector(const Eigen::ArrayBase<Derived>& array) {
    std::vector<std::vector<double>> result;

    for (int i = 0; i < array.rows(); i++) {
        std::vector<double> row;
        for (int j = 0; j < array.cols(); j++) {
            row.push_back(array(i, j));
        }
        result.push_back(row);
    }

    return result;
}

template <typename Derived>
inline std::vector<double> eigen_to_single_vector(const Eigen::ArrayBase<Derived>& array) {
    std::vector<double> result;

    for (int i = 0; i < array.rows(); i++) {
        result.push_back(array(i));
    }

    return result;
}

#endif // SERIALIZATION_H_
