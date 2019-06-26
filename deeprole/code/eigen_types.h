#ifndef EIGEN_TYPES_H_
#define EIGEN_TYPES_H_

#include <Eigen/Core>
#include "game_constants.h"

typedef Eigen::Array<double, NUM_VIEWPOINTS, 1> ViewpointVector;
typedef Eigen::Array<double, NUM_VIEWPOINTS, NUM_PROPOSAL_OPTIONS> ProposeData;
typedef Eigen::Array<double, NUM_VIEWPOINTS, 2> VoteData;
typedef Eigen::Array<double, NUM_VIEWPOINTS, 2> MissionData;
typedef Eigen::Array<double, NUM_VIEWPOINTS, NUM_PLAYERS> MerlinData;
typedef Eigen::Array<double, NUM_ASSIGNMENTS, 1> AssignmentProbs;

#endif // EIGEN_TYPES_H_
