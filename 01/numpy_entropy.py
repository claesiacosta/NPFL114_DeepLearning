#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    c = 0
    distribution = dict()
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            c += 1
            if (not line in distribution):
                distribution[line] = {"data": 0, "model": 0}
            distribution[line]["data"] +=1
                
    for key in distribution:
        distribution[key]["data"] /=c

    # TODO: Load model distribution, each line `string \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            key, value = line.split("\t")

            if (key in distribution):
                distribution[key]["model"] = float(value)

    # TODO: Create a NumPy array containing the model distribution.
    data_distribution = []
    model_distribution = []
    for item in distribution.values():
        data_distribution.append(item["data"])
        model_distribution.append(item["model"])

    data_np_dis = np.array(data_distribution)
    model_np_dis = np.array(model_distribution)

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -(data_np_dis * np.log(data_np_dis)).sum()

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    crossentropy = -(data_np_dis * np.log(model_np_dis)).sum()

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
