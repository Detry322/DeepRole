#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include "json.h"
#include "optionparser.h"
#include "lookahead.h"
#include "cfr_plus.h"
#include "util.h"
#include "nn.h"
#include "serialization.h"

using namespace std;

enum optionIndex { 
    UNKNOWN,
    HELP,
    NUM_DATAPOINTS,
    NUM_ITERATIONS,
    NUM_WAIT_ITERS,
    MODEL_SEARCH_DIR,
    OUT_DIR,
    FILE_SUFFIX,
    TEST_MODE,
    NUM_SUCCEEDS,
    NUM_FAILS,
    PROPOSE_COUNT,
    DEPTH,
    PLAY_MODE,
    PROPOSER,
    NN_TEST,
};

const option::Descriptor usage[] = {
    { UNKNOWN,           0,   "",              "",        option::Arg::None,       "USAGE: deeprole [options]\n\nOptions:"},
    { HELP,              0,   "h",         "help",        option::Arg::None,       "  \t-h, --help  \tPrint usage and exit." },
    { NUM_DATAPOINTS,    0,   "n",  "ndatapoints",    option::Arg::Optional,       "  \t-n<num>, --ndatapoints=<num>  \tNumber of datapoints to generate (10000 default)" },
    { NUM_ITERATIONS,    0,   "i",   "iterations",    option::Arg::Optional,       "  \t-i<num>, --iterations=<num>  \tNum of iterations to run for (default: 4000)"},
    { NUM_WAIT_ITERS,    0,   "w",       "witers",    option::Arg::Optional,       "  \t-w<num>, --witers=<num>  \tNum of iterations to ignore (default: 1000)"},
    { MODEL_SEARCH_DIR,  0,   "m",     "modeldir",    option::Arg::Optional,       "  \t-m<directory>, --modeldir=<directory>  \tWhere to search for models (default: 'models')"},
    { OUT_DIR,           0,   "o",          "out",    option::Arg::Optional,       "  \t-o<directory>, --out=<directory>  \tThe output directory to write to (default: .)"},
    { FILE_SUFFIX,       0,   "x",       "suffix",    option::Arg::Optional,       "  \t-x<text>, --suffix=<suffix>  \tCode to append to every filename (default: <random hex chars>)"},
    { TEST_MODE,         0,   "t",         "test",    option::Arg::Optional,       "  \t-t, --test  \tRun single test" },
    { NUM_SUCCEEDS,    0,   "s",       "succeeds",    option::Arg::Optional,       "  \t-s<num>, --succeeds=<num>  \tThe number of succeeds in the game (2 default)" },
    { NUM_FAILS,       0,   "f",          "fails",    option::Arg::Optional,       "  \t-f<num>, --fails=<num>  \tThe number of fails in the game (2 default)" },
    { PROPOSE_COUNT,   0,   "p",  "propose_count",    option::Arg::Optional,       "  \t-p<num>, --propose_count=<num>  \tThe proposal round (4 default)" },
    { DEPTH,           0,   "d",          "depth",    option::Arg::Optional,       "  \t-d<num>, --depth=<num>  \tThe depth to do CFR at (1 default)" },
    { PLAY_MODE,       0,   "l",           "play",    option::Arg::Optional,       "  \t-l, --play  \tRun in play mode. Read a belief from stdin, output data to stdout." },
    { PROPOSER,        0,   "r",       "proposer",    option::Arg::Optional,       "  \t-r, --proposer=<num>  \tUse a specific proposer for play mode." },
    { NN_TEST,         0,   "u",        "nn-test",    option::Arg::Optional,       "  \t-u, --nn-test  \tRun neural net test" },
    { 0, 0, 0, 0, 0, 0 }
};

std::string random_string(std::string::size_type length)
{
    static auto& chrs = "0123456789abcdef";
    static std::uniform_int_distribution<std::string::size_type> pick(0, sizeof(chrs) - 2);

    std::string s;

    s.reserve(length);

    while(length--)
        s += chrs[pick(rng)];

    return s;
}

void print_lookahead_information(const int depth, const int num_succeeds, const int num_fails, const int propose_count, const std::string& model_search_dir) {
    auto lookahead = create_avalon_lookahead(num_succeeds, num_fails, 0, propose_count, depth, model_search_dir);
    cerr << "                PROPOSE: " << count_lookahead_type(lookahead.get(), PROPOSE) << endl;
    cerr << "                   VOTE: " << count_lookahead_type(lookahead.get(), VOTE) << endl;
    cerr << "                MISSION: " << count_lookahead_type(lookahead.get(), MISSION) << endl;
    cerr << "        TERMINAL_MERLIN: " << count_lookahead_type(lookahead.get(), TERMINAL_MERLIN) << endl;
    cerr << "  TERMINAL_NO_CONSENSUS: " << count_lookahead_type(lookahead.get(), TERMINAL_NO_CONSENSUS) << endl;
    cerr << "TERMINAL_TOO_MANY_FAILS: " << count_lookahead_type(lookahead.get(), TERMINAL_TOO_MANY_FAILS) << endl;
    cerr << "    TERMINAL_PROPOSE_NN: " << count_lookahead_type(lookahead.get(), TERMINAL_PROPOSE_NN) << endl;
    cerr << "                  Total: " << count_lookahead(lookahead.get()) << endl;
}

void generate_datapoints(
    const int num_datapoints,
    const int depth,
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    const int iterations,
    const int wait_iterations,
    const std::string model_search_dir,
    const std::string output_dir,
    const std::string filename_suffix
) {
    std::string base_filename = (
        "d" + std::to_string(depth) + "_" +
        "s" + std::to_string(num_succeeds) + "_" +
        "f" + std::to_string(num_fails) + "_" +
        "p" + std::to_string(propose_count) + "_" +
        "i" + std::to_string(iterations) + "_" +
        "w" + std::to_string(wait_iterations) + "_" +
        ((filename_suffix.empty()) ? random_string(16) : filename_suffix) +
        ".csv"
    );
    const std::string filepath = ((output_dir.empty()) ? "" : (output_dir + "/")) + base_filename;

    cerr << "=========== DEEPROLE DATAPOINT GENERATOR =========" << endl;
    cerr << "           # Datapoints: " << num_datapoints << endl;
    cerr << "           # Iterations: " << iterations << endl;
    cerr << "           # Wait iters: " << wait_iterations << endl;
    cerr << "------------------ Game settings -------------------" << endl;
    cerr << "                  Depth: " << depth << endl;
    cerr << "                  Round: " << (num_succeeds + num_fails) << endl;
    cerr << "               Succeeds: " << num_succeeds << endl;
    cerr << "                  Fails: " << num_fails << endl;
    cerr << "              Propose #: " << propose_count << endl;
    cerr << "------------------ Sanity checks -------------------" << endl;
    print_lookahead_information(depth, num_succeeds, num_fails, propose_count, model_search_dir);
    cerr << "------------------ Loaded Models -------------------" << endl;
    print_loaded_models(model_search_dir);
    cerr << "------------------ Administration ------------------" << endl;
    cerr << " Output directory: " << "'" << output_dir << "'" << endl;
    cerr << " Writing to: " << filepath << endl;
    cerr << "====================================================" << endl;

    const int status_interval = max(1, min(100, num_datapoints/10));

    std::fstream fs;
    fs.open(filepath, std::fstream::out | std::fstream::app);

    for (int i = 0; i < num_datapoints; i++) {
        if (i % status_interval == 0) {
            cerr << i << "/" << num_datapoints << endl;
        }

        Initialization init;
        prepare_initialization(depth, num_succeeds, num_fails, propose_count, &init);
        run_initialization_with_cfr(iterations, wait_iterations, model_search_dir, &init);

        fs << init.Stringify() << endl << flush;
    }
    fs.close();
}

void test(
    const int depth,
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    const int proposer,
    const int iterations,
    const int wait_iterations,
    const std::string model_search_dir
) {
    auto lookahead = create_avalon_lookahead(
        num_succeeds,
        num_fails,
        proposer,
        propose_count,
        depth,
        model_search_dir
    );

    Initialization init;
    prepare_initialization(depth, num_succeeds, num_fails, propose_count, &init);
    run_initialization_with_cfr(iterations, wait_iterations, model_search_dir, &init);
    cout << init.solution_values[0].transpose() << endl;
}

void play_mode(
    const int depth,
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    const int proposer,
    const int iterations,
    const int wait_iterations,
    const std::string model_search_dir
) {
    cerr << "~.~.~.~.~.~.~.~. DEEPROLE PLAY MODE .~.~.~.~.~.~.~.~" << endl;
    cerr << "           # Iterations: " << iterations << endl;
    cerr << "           # Wait iters: " << wait_iterations << endl;
    cerr << "------------------ Game settings -------------------" << endl;
    cerr << "                  Depth: " << depth << endl;
    cerr << "                  Round: " << (num_succeeds + num_fails) << endl;
    cerr << "               Succeeds: " << num_succeeds << endl;
    cerr << "                  Fails: " << num_fails << endl;
    cerr << "              Propose #: " << propose_count << endl;
    cerr << "------------------ Sanity checks -------------------" << endl;
    print_lookahead_information(depth, num_succeeds, num_fails, propose_count, model_search_dir);
    cerr << "------------------ Loaded Models -------------------" << endl;
    print_loaded_models(model_search_dir);

    auto lookahead = create_avalon_lookahead(
        num_succeeds,
        num_fails,
        proposer,
        propose_count,
        depth,
        model_search_dir
    );

    AssignmentProbs starting_probs;
    json_deserialize_starting_reach_probs(std::cin, &starting_probs);

    ViewpointVector _dummy_values[NUM_PLAYERS];
    cfr_get_values(lookahead.get(), iterations, wait_iterations, starting_probs, true, _dummy_values);
    calculate_cumulative_strategy(lookahead.get());
    json_serialize_lookahead(lookahead.get(), starting_probs, std::cout);
}

void nn_test_mode(
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    const int proposer,
    const std::string model_search_dir
) {
    auto model = load_model(model_search_dir, num_succeeds, num_fails, propose_count);

    AssignmentProbs starting_probs;
    json_deserialize_starting_reach_probs(std::cin, &starting_probs);

    ViewpointVector results[NUM_PLAYERS];
    model->predict(proposer, starting_probs, results);

    nlohmann::json json;

    for (int i = 0; i < NUM_PLAYERS; i++) {
        json.push_back(eigen_to_single_vector(results[i]));
    }

    std::cout << std::setprecision(17) << std::setw(2) << json << std::endl;
}

int main(int argc, char* argv[]) {
    argc -= (argc > 0); argv += (argc > 0); // skip program name argv[0] if present
    
    option::Stats stats(usage, argc, argv);
    std::vector<option::Option> options(stats.options_max);
    std::vector<option::Option> buffer(stats.buffer_max);
    option::Parser parse(usage, argc, argv, &options[0], &buffer[0]);
    if (parse.error())
        return 1;

    if (options[HELP]) {
        option::printUsage(std::cout, usage);
        return 0;
    }

    std::string s_num_datapoints;
    std::string s_num_iterations;
    std::string s_num_wait_iters;
    std::string model_search_dir = "models";
    std::string out_dir = "deeprole_output";
    std::string file_suffix;
    std::string s_num_succeeds;
    std::string s_num_fails;
    std::string s_propose_count;
    std::string s_depth;
    std::string s_proposer;
    if (options[NUM_DATAPOINTS]) s_num_datapoints = std::string(options[NUM_DATAPOINTS].last()->arg);
    if (options[NUM_ITERATIONS]) s_num_iterations = std::string(options[NUM_ITERATIONS].last()->arg);
    if (options[NUM_WAIT_ITERS]) s_num_wait_iters = std::string(options[NUM_WAIT_ITERS].last()->arg);
    if (options[MODEL_SEARCH_DIR]) model_search_dir = std::string(options[MODEL_SEARCH_DIR].last()->arg);
    if (options[OUT_DIR]) out_dir = std::string(options[OUT_DIR].last()->arg);
    if (options[FILE_SUFFIX]) file_suffix = std::string(options[FILE_SUFFIX].last()->arg);
    if (options[NUM_SUCCEEDS]) s_num_succeeds = std::string(options[NUM_SUCCEEDS].last()->arg);
    if (options[NUM_FAILS]) s_num_fails = std::string(options[NUM_FAILS].last()->arg);
    if (options[PROPOSE_COUNT]) s_propose_count = std::string(options[PROPOSE_COUNT].last()->arg);
    if (options[DEPTH]) s_depth = std::string(options[DEPTH].last()->arg);
    if (options[PROPOSER]) s_proposer = std::string(options[PROPOSER].last()->arg);

    int num_datapoints = (s_num_datapoints.empty()) ? 10000 : std::stoi(s_num_datapoints);
    int num_iterations = (s_num_iterations.empty()) ? 3000 : std::stoi(s_num_iterations);
    int num_wait_iters = (s_num_wait_iters.empty()) ? 1000 : std::stoi(s_num_wait_iters);
    int num_succeeds = (s_num_succeeds.empty()) ? 2 : std::stoi(s_num_succeeds);
    int num_fails = (s_num_fails.empty()) ? 2 : std::stoi(s_num_fails);
    int propose_count = (s_propose_count.empty()) ? 4 : std::stoi(s_propose_count);
    int depth = (s_depth.empty()) ? 1 : std::stoi(s_depth);
    int proposer = (s_proposer.empty()) ? -1 : std::stoi(s_proposer);

    if (options[TEST_MODE]) {
        if (proposer < 0 || proposer >= NUM_PLAYERS) {
            proposer = 0;
        }
        test(
            depth,
            num_succeeds,
            num_fails,
            propose_count,
            proposer,
            num_iterations,
            num_wait_iters,
            model_search_dir
        );
        return 0;
    }

    if (options[PLAY_MODE] || options[NN_TEST]) {
        if (proposer < 0 || proposer >= NUM_PLAYERS) {
            std::cerr << "You must pass a valid proposer" << std::endl;
            return 1;
        }

        if (options[PLAY_MODE]) {
            play_mode(
                depth,
                num_succeeds,
                num_fails,
                propose_count,
                proposer,
                num_iterations,
                num_wait_iters,
                model_search_dir
            );            
        } else {
            nn_test_mode(num_succeeds, num_fails, propose_count, proposer, model_search_dir);
        }

        return 0;
    }

    seed_rng();
    generate_datapoints(
        num_datapoints,
        depth,
        num_succeeds,
        num_fails,
        propose_count,
        num_iterations,
        num_wait_iters,
        model_search_dir,
        out_dir,
        file_suffix
    );
    return 0;
}
