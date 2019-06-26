import subprocess
import json
import os
import numpy as np

from battlefield.bots.deeprole.lookup_tables import ASSIGNMENT_TO_VIEWPOINT

DEEPROLE_BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'deeprole')
DEEPROLE_BINARY_BASE = os.path.join(DEEPROLE_BASE_DIR, 'code')


deeprole_cache = {}

def actually_run_deeprole_on_node(node, iterations, wait_iterations, no_zero, nn_folder, binary):
    command = [
        os.path.join(DEEPROLE_BINARY_BASE, binary),
        '--play',
        '--proposer={}'.format(node['proposer']),
        '--succeeds={}'.format(node['succeeds']),
        '--fails={}'.format(node['fails']),
        '--propose_count={}'.format(node['propose_count']),
        '--depth=1',
        '--iterations={}'.format(iterations),
        '--witers={}'.format(wait_iterations),
        '--modeldir={}'.format(nn_folder)
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=DEEPROLE_BASE_DIR
    )

    if no_zero:
        belief = node['nozero_belief']
    else:
        belief = node['new_belief']

    stdout, _ = process.communicate(input=str(belief) + "\n")
    result = json.loads(stdout)
    return result


def run_deeprole_on_node(node, iterations, wait_iterations, no_zero=False, nn_folder='deeprole_models', binary='deeprole'):
    global deeprole_cache

    if len(deeprole_cache) > 250:
        deeprole_cache = {}

    cache_key = (
        node['proposer'],
        node['succeeds'],
        node['fails'],
        node['propose_count'],
        tuple(node['new_belief']),
        iterations,
        wait_iterations,
        no_zero,
        nn_folder,
        binary
    )
    
    if cache_key in deeprole_cache:
        return deeprole_cache[cache_key]

    result = actually_run_deeprole_on_node(node, iterations, wait_iterations, no_zero, nn_folder, binary)
    deeprole_cache[cache_key] = result
    return result
