import tensorflow as tf

from export_to_fdeep.convert_model import model_to_fdeep_json
from train import CFVMaskAndAdjustLayer, CFVFromWinProbsLayer, ZeroSumLayer, loss, check_sum

import sys
import json

def main():
    if len(sys.argv) != 3:
        print "Usage: python convert.py <input model> <output model>"
        exit(1)
    _, inp, out = sys.argv

    model = tf.keras.models.load_model(inp, custom_objects={
        'CFVMaskAndAdustLayer': CFVMaskAndAdjustLayer,
        'loss': loss,
        'CFVFromWinProbsLayer': CFVFromWinProbsLayer,
        'ZeroSumLayer': ZeroSumLayer,
        'check_sum': check_sum
    })
    fdeep_json = model_to_fdeep_json(model)
    with open(out, 'w') as f:
        json.dump(fdeep_json, f, allow_nan=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
