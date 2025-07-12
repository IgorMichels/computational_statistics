# pylint: disable=too-many-locals

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

DEFAULT_N_ITER = 100_000
DEFAULT_BURN = 20_000
DEFAULT_SEED = 0

DEFAULT_CHAINS = 4
DEFAULT_PLACEBO = False
DEFAULT_VERBOSE = False

DEFAULT_M0 = "0.0"
DEFAULT_S0_2 = "4.0"
DEFAULT_ALPHA0 = "2.0"
DEFAULT_BETA0 = "1.0"

DEFAULT_N_TEMPS = 10
DEFAULT_MAX_TEMP = 10.0
DEFAULT_N_GIBBS_PER_TEMP = 1

datasets = [
    {"name": "example_1", "K": 2},
    {"name": "example_2", "K": 4},
    {"name": "example_3", "K": 4},
    {"name": "example_4", "K": 5},
]


def run_command(cmd):
    """
    Execute a shell command and display its execution status and duration.

    This function runs a shell command using subprocess, measures its execution time,
    and prints a formatted status message indicating success or failure along with
    the duration and command executed.

    Args:
        cmd (str): The shell command to execute

    Returns:
        bool: True if the command executed successfully (return code 0), False otherwise
    """
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, check=False)
    duration = time.time() - start_time

    status = "✅" if result.returncode == 0 else "❌"
    print(f"{status} [{duration:6.2f}s] {cmd}")

    return result.returncode == 0


def generate_commands(dataset):
    """
    Generate commands to run sampling tests on a specific dataset.

    This function creates a list of commands to execute:
    1. Gibbs sampler
    2. Tempered transitions
    3. Comparison tool

    Args:
        dataset (dict): Dictionary containing dataset information, including:
            - name (str): Dataset name
            - K (int): Number of components
            - Optional parameters like n_iter, burn, seed, chains, etc.

    Returns:
        list: List of strings containing the commands to be executed
    """
    n_iter = dataset.get("n_iter", DEFAULT_N_ITER)
    burn = dataset.get("burn", DEFAULT_BURN)
    seed = dataset.get("seed", DEFAULT_SEED)
    chains = dataset.get("chains", DEFAULT_CHAINS)
    placebo = dataset.get("placebo", DEFAULT_PLACEBO)
    verbose = dataset.get("verbose", DEFAULT_VERBOSE)
    m0 = dataset.get("m0", DEFAULT_M0)
    s0_2 = dataset.get("s0_2", DEFAULT_S0_2)
    alpha0 = dataset.get("alpha0", DEFAULT_ALPHA0)
    beta0 = dataset.get("beta0", DEFAULT_BETA0)
    n_temps = dataset.get("n_temps", DEFAULT_N_TEMPS)
    max_temp = dataset.get("max_temp", DEFAULT_MAX_TEMP)
    n_gibbs_per_temp = dataset.get("n_gibbs_per_temp", DEFAULT_N_GIBBS_PER_TEMP)

    data_name = dataset["name"]
    K_val = dataset["K"]

    dataset_args = f"--data {data_name} --K {K_val}"
    base_args = (
        f"--n_iter {n_iter} --burn {burn} --seed {seed}"
        f" --m0 {m0} --s0_2 {s0_2} --alpha0 {alpha0} --beta0 {beta0}"
    )

    if placebo:
        base_args += " --placebo"

    if verbose:
        base_args += " --verbose"

    tempered_args = (
        f"--n_temps {n_temps} --max_temp {max_temp} "
        f"--n_gibbs_per_temp {n_gibbs_per_temp}"
    )

    return [
        f"python gibbs_sampler.py {dataset_args} {base_args} --chains {chains}",
        f"python tempered_transitions.py {dataset_args} {base_args} "
        f"{tempered_args} --chains {chains}",
        f"python comparison_tool.py all {dataset_args} {base_args} {tempered_args}",
    ]


if __name__ == "__main__":
    start_time = time.time()

    if not run_command("python generate_data.py"):
        print("❌ Data generation failed")
        sys.exit(1)

    all_commands = []
    for dataset in datasets:
        all_commands.extend(generate_commands(dataset))

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_command, all_commands))

    failed_count = sum(1 for success in results if not success)
    total_time = time.time() - start_time

    print("\n📊 Summary:")
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f}min)")
    print(f"✅ Successes: {len(results) - failed_count}/{len(results)}")
    print(f"❌ Failures: {failed_count}/{len(results)}")
