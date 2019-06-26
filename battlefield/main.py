from battlefield.bots import (
    # RandomBot, RandomBotUV,
    # SimpleBot,
    # ObserveBot, ExamineAgreementBot,
    # ISMCTSBot, MOISMCTSBot,
    # HumanBot, HumanLikeBot,
    # NNBot, NNBotWithObservePropose,
    # SimpleStatsBot,
    # SingleMCTSPlayoutBot, SingleMCTSHeuristicBot, SingleMCTSBaseOpponentBot,
    # CFRBot,
    # LearningBot,
    # ObserveBeaterBot,
    Deeprole,
    Deeprole_3_1,
    Deeprole_10_5,
    Deeprole_30_15,
    Deeprole_100_50,
    Deeprole_300_150,
    Deeprole_100_0,
    Deeprole_100_25,
    Deeprole_100_75,
    Deeprole_100_100,
    Deeprole_NoZeroing,
    Deeprole_OldNNs,
    Deeprole_NoZeroUnconstrained,
    Deeprole_ZeroingUnconstrained,
    Deeprole_ZeroingWinProbs,
)
from battlefield.tournament import (
    run_simple_tournament,
    run_large_tournament,
    run_all_combos_parallel,
    run_all_combos_simple,
    run_all_combos,
    print_tournament_statistics,
    check_config,
    run_learning_tournament,
    run_single_threaded_tournament,
    run_and_print_game
)
# from battlefield.compare_to_human import compute_human_statistics, print_human_statistics
# from battlefield.predict_roles import predict_evil_over_human_data, predict_evil_using_voting, grid_search
# from battlefield.determine_reachable_states import determine_reachable
# from battlefield.subgame import calculate_subgame_ll, test_calculate
# from battlefield.predict_merlin import get_merlin_prediction
# from battlefield.merlin import recover_all_deeprole_thoughts
import multiprocessing
import pandas as pd
import random
import sys
import gzip
import cPickle as pickle
from collections import defaultdict

TOURNAMENT_CONFIG = [
    {
        'bot': Deeprole,
        'role': 'merlin'
    },
    {
        'bot': Deeprole,
        'role': 'minion'
    },
    {
        'bot': Deeprole_NoZeroing,
        'role': 'servant'
    },
    {
        'bot': Deeprole_OldNNs,
        'role': 'servant'
    },
    {
        'bot': Deeprole,
        'role': 'assassin'
    }
]

def tournament():
    check_config(TOURNAMENT_CONFIG)
    print "hidden", tuple(x['role'] for x in TOURNAMENT_CONFIG)
    tournament_results = run_single_threaded_tournament(TOURNAMENT_CONFIG, num_games=2000, granularity=200)
    print_tournament_statistics(tournament_results)


def join_and_write(results):
    result_by_bot = defaultdict(lambda: [])
    for result in results:
        bot_name = result.loc[0].bot
        result_by_bot[bot_name].append(result)

    for bot, results in result_by_bot.items():
        print "Saving", bot
        with gzip.open('human/{}.msg.gz'.format(bot), 'w') as f:
            data = pd.concat(results)
            data.reset_index(drop=True, inplace=True)
            data.to_msgpack(f)


def parallel_human_compare():
    pool = multiprocessing.Pool()
    try:
        results = []
        for bot in [CFRBot(4000000)]:
            for tremble in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
                results.append(pool.apply_async(compute_human_statistics, (bot, tremble, False, 5)))

        print "Waiting for results"
        sys.stdout.flush()
        results = [ result.get() for result in results ]
        print "Concatenating and writing results"
        sys.stdout.flush()
        join_and_write(results)
    except KeyboardInterrupt:
        print 'terminating early'
        pool.terminate()


STRING_TO_BOT = {
    # "RandomBot": RandomBot,
    # "RandomBotUV": RandomBotUV,
    # "SimpleBot": SimpleBot,
    # "ObserveBot": ObserveBot,
    # "SimpleStatsBot": SimpleStatsBot,
    # "CFRBot_3000000": CFRBot(3000000),
    # "CFRBot_6000000": CFRBot(6000000),
    # "CFRBot_10000000": CFRBot(10000000),
    # "ISMCTSBot": ISMCTSBot,
    # "MOISMCTSBot": MOISMCTSBot,
    "Deeprole": Deeprole,
    "Deeprole_3_1": Deeprole_3_1,
    "Deeprole_10_5": Deeprole_10_5,
    "Deeprole_30_15": Deeprole_30_15,
    "Deeprole_100_50": Deeprole_100_50,
    "Deeprole_300_150": Deeprole_300_150,
    "Deeprole_100_0": Deeprole_100_0,
    "Deeprole_100_25": Deeprole_100_25,
    "Deeprole_100_75": Deeprole_100_75,
    "Deeprole_100_100": Deeprole_100_100,
    "Deeprole_OldNNs": Deeprole_OldNNs,
    "Deeprole_NoZeroing": Deeprole_NoZeroing,
    "Deeprole_NoZeroUnconstrained": Deeprole_NoZeroUnconstrained,
    "Deeprole_ZeroingUnconstrained": Deeprole_ZeroingUnconstrained,
    "Deeprole_ZeroingWinProbs": Deeprole_ZeroingWinProbs,
}

def human_compare():
    assert len(sys.argv) == 4, "Need 3 arguments to collect human stats"
    bot = STRING_TO_BOT[sys.argv[1]]
    min_game_id = int(sys.argv[2])
    max_game_id = int(sys.argv[3])
    stats = compute_human_statistics(
        bot,
        verbose=True,
        num_players=5,
        max_num_players=None,
        min_game_id=min_game_id,
        max_game_id=max_game_id,
        roles=set(['servant', 'merlin', 'minion', 'assassin'])
    )
    with gzip.open('human/{}-{}-{}.msg.gz'.format(bot.__name__, min_game_id, max_game_id), 'w') as f:
        stats.to_msgpack(f)


def predict_roles(bot, tremble):
    bot = STRING_TO_BOT[bot]
    tremble = float(tremble)
    dataframe, particles = predict_evil_over_human_data(bot, tremble)
    print "Writing dataframe..."
    with gzip.open('predict_roles/{}_{}_df.msg.gz'.format(bot.__name__, tremble), 'w') as f:
        dataframe.to_msgpack(f)

    print "Writing particles..."
    with gzip.open('predict_roles/{}_{}_particles.pkl.gz'.format(bot.__name__, tremble), 'w') as f:
        pickle.dump(particles, f)


def predict_merlin():
    assert len(sys.argv) == 4, "Need 3 arguments to collect human stats"
    bot = STRING_TO_BOT[sys.argv[1]]
    min_game_id = int(sys.argv[2])
    max_game_id = int(sys.argv[3])

    result = get_merlin_prediction(bot, num_players=5, min_game_id=min_game_id, max_game_id=max_game_id)

    with gzip.open('merlin/{}-{}-{}.msg.gz'.format(bot.__name__, min_game_id, max_game_id), 'w') as f:
        result.to_msgpack(f)


def benchmark_performance_simple():
    games_per_matching = int(sys.argv[1])
    bot_names = sys.argv[2:]
    bot_classes = [ STRING_TO_BOT[name] for name in bot_names ]
    roles = ['merlin', 'servant', 'assassin', 'minion', 'servant']
    run_all_combos_simple(bot_classes, roles, games_per_matching=games_per_matching)


def benchmark_performance():
    print "Launching..."
    games_per_matching = int(sys.argv[1])
    bot_names = sys.argv[2:]
    bot_classes = [ STRING_TO_BOT[name] for name in bot_names ]
    roles = ['merlin', 'servant', 'assassin', 'minion', 'servant']
    run_all_combos(bot_classes, roles, games_per_matching=games_per_matching, parallelization=40)




if __name__ == "__main__":
    # bots = [ ObserveBeaterBot, ObserveBot, ObserveBot, ObserveBot, ObserveBot ]
    # run_learning_tournament(bots, winrate_track=0)
    # grid_search()
    # predict_evil_using_voting()
    # tournament()
    # roles = ['merlin', 'assassin', 'servant', 'servant', 'minion']
    # random.shuffle(roles)
    # run_and_print_game(TOURNAMENT_CONFIG)
    # print_tournament_statistics(
    #     run_single_threaded_tournament(TOURNAMENT_CONFIG, num_games=100, granularity=1)
    # )
    # import sys
    # modulenames = set(sys.modules) & set(globals())
    # print [sys.modules[name] for name in modulenames]
    benchmark_performance()
    # recover_all_deeprole_thoughts()
    # predict_merlin()
    # human_compare()
    # determine_reachable(RandomBot, set(['merlin', 'minion', 'assassin', 'servant']), 5)
    # test_calculate()
    # df, _ = predict_evil_over_human_data(HumanLikeBot, 0.01)
    # import IPython; IPython.embed()
