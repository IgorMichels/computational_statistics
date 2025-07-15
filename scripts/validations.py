import json

import numpy as np

file_template = "../data/example_{}/metrics.json"

for i in range(1, 5):
    file = file_template.format(i)
    with open(file, encoding="utf-8") as f:
        metrics = json.load(f)

    print(file)
    for model in metrics:
        print(f"  Model: {model}")
        model_metrics = metrics[model]
        for k, v in model_metrics.items():
            if "rhat" in k:
                print(f"    {k}: {[round(x, 4) for x in v]}")

        for k, v in model_metrics.items():
            if "ess" in k:
                print(f"    {k}: {[round(x, 4) for x in v]}")

        if "acceptance_rates" in model_metrics:
            avg_acceptance_rate = np.mean(model_metrics["acceptance_rates"])
            print(f"    average acceptance rate: {round(avg_acceptance_rate, 4)}")

        if "runtimes" in model_metrics:
            print(f"    runtimes: {[round(x, 4) for x in model_metrics['runtimes']]}")

        if "mean_runtime" in model_metrics:
            print(f"    mean runtime: {round(model_metrics['mean_runtime'], 4)}s")

        if "std_runtime" in model_metrics:
            print(f"    std runtime: {round(model_metrics['std_runtime'], 4)}s")

    print()
