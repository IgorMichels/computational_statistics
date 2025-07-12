import numpy as np


def parse_prior_args(arg_str, K, param_name):
    """Parse prior arguments that can be either scalar or vector.

    Args:
        arg_str: String containing comma-separated values or single value
        K: Number of mixture components
        param_name: Name of the parameter for error messages

    Returns:
        Scalar value or numpy array of size K

    Raises:
        ValueError: If the number of values doesn't match K
    """
    if arg_str is None:
        return None

    values = [float(x) for x in arg_str.split(",")]
    if len(values) == 1:
        return values[0]
    if len(values) == K:
        return np.array(values)
    raise ValueError(
        f"{param_name} must be either scalar or have K={K} values, got {len(values)}"
    )


def add_common_args(subparser):
    """Add common arguments to an argument parser.

    Args:
        subparser: ArgumentParser subparser to add arguments to
    """
    subparser.add_argument(
        "--data", type=str, default="example_1", help="Data directory"
    )
    subparser.add_argument(
        "--K", type=int, default=4, help="Number of mixture components"
    )
    subparser.add_argument(
        "--placebo", action="store_true", help="Use placebo on relabeling"
    )
    subparser.add_argument(
        "--verbose", action="store_true", help="Indicates if verbose output is desired"
    )


def add_sampling_args(subparser, n_iter=10000, burn=2000):
    """Add sampling-related arguments to an argument parser.

    Args:
        subparser: ArgumentParser subparser to add arguments to
        n_iter: Default number of iterations
        burn: Default burn-in period
    """
    subparser.add_argument(
        "--n_iter", type=int, default=n_iter, help="Number of iterations"
    )
    subparser.add_argument("--burn", type=int, default=burn, help="Burn-in period")
    subparser.add_argument("--seed", type=int, default=0, help="Random seed")


def add_prior_args(subparser):
    """Add prior parameter arguments to an argument parser.

    Args:
        subparser: ArgumentParser subparser to add arguments to
    """
    subparser.add_argument(
        "--m0",
        type=str,
        default="0.0",
        help="Prior means for mu (scalar or comma-separated values)",
    )
    subparser.add_argument(
        "--s0_2",
        type=str,
        default="4.0",
        help="Prior variances for mu (scalar or comma-separated values)",
    )
    subparser.add_argument(
        "--alpha0",
        type=str,
        default="2.0",
        help="Prior shape parameters for sigma2 (scalar or comma-separated values)",
    )
    subparser.add_argument(
        "--beta0",
        type=str,
        default="1.0",
        help="Prior scale parameters for sigma2 (scalar or comma-separated values)",
    )


def add_chain_args(subparser):
    """Add chain-related arguments to an argument parser.

    Args:
        subparser: ArgumentParser subparser to add arguments to
    """
    subparser.add_argument("--chains", type=int, default=4, help="Number of chains")


def add_tempered_args(subparser, max_temp=10.0, n_gibbs_per_temp=1):
    """Add tempered transitions specific arguments to an argument parser.

    Args:
        subparser: ArgumentParser subparser to add arguments to
        max_temp: Default maximum temperature
        n_gibbs_per_temp: Default number of Gibbs steps per temperature
    """
    subparser.add_argument(
        "--n_temps", type=int, default=10, help="Number of temperatures"
    )
    subparser.add_argument(
        "--max_temp", type=float, default=max_temp, help="Maximum temperature"
    )
    subparser.add_argument(
        "--n_gibbs_per_temp",
        type=int,
        default=n_gibbs_per_temp,
        help="Gibbs steps per temperature",
    )


def add_gibbs_args(subparser):
    """Add all arguments for Gibbs sampler script.

    Args:
        subparser: ArgumentParser subparser to add arguments to
    """
    add_common_args(subparser)
    add_sampling_args(subparser, n_iter=10000, burn=2000)
    add_chain_args(subparser)
    add_prior_args(subparser)


def add_tempered_transitions_args(subparser):
    """Add all arguments for tempered transitions script.

    Args:
        subparser: ArgumentParser subparser to add arguments to
    """
    add_common_args(subparser)
    add_sampling_args(subparser, n_iter=10000, burn=2000)
    add_chain_args(subparser)
    add_prior_args(subparser)
    add_tempered_args(subparser, max_temp=10.0, n_gibbs_per_temp=1)


def add_comparison_args(subparser):
    """Add all arguments for comparison tool script.

    Args:
        subparser: ArgumentParser subparser to add arguments to
    """
    add_common_args(subparser)
    add_sampling_args(subparser, n_iter=10000, burn=2000)
    add_prior_args(subparser)
    add_tempered_args(subparser, max_temp=10.0, n_gibbs_per_temp=1)


def parse_all_prior_args(args):
    """Parse all prior arguments from command line args.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (m0, s0_2, alpha0, beta0) parsed prior parameters
    """
    m0 = parse_prior_args(args.m0, args.K, "m0")
    s0_2 = parse_prior_args(args.s0_2, args.K, "s0_2")
    alpha0 = parse_prior_args(args.alpha0, args.K, "alpha0")
    beta0 = parse_prior_args(args.beta0, args.K, "beta0")

    return m0, s0_2, alpha0, beta0


def print_summary_header(method_name, chains=None):
    """Print a formatted summary header.

    Args:
        method_name: Name of the sampling method
        chains: Number of chains (optional)
    """
    if chains:
        print(f"\n=== {method_name.upper()} SUMMARY ===")
    else:
        print(f"\n=== {method_name.upper()} SUMMARY ===")


def print_parameter_summary(param_symbol, metrics):
    """Print formatted parameter summary statistics.

    Args:
        param_symbol: Symbol for the parameter (e.g., "μ", "σ²")
        metrics: Tuple of (mean_vals, ci_lower, ci_upper, rhat_vals, ess_vals)
    """
    mean_vals, ci_lower, ci_upper, rhat_vals, ess_vals = metrics
    print(f"Posterior mean {param_symbol}      :", np.round(mean_vals, 4))
    print(f"95% CI lower {param_symbol}        :", np.round(ci_lower, 4))
    print(f"95% CI upper {param_symbol}        :", np.round(ci_upper, 4))
    print(f"R‑hat ({param_symbol})             :", np.round(rhat_vals, 3))
    print(f"ESS  ({param_symbol})              :", np.round(ess_vals, 1))


def print_runtime_summary(
    times, acceptance_rates=None, n_temps=None, max_temp=None, n_gibbs_per_temp=None
):
    """Print runtime and acceptance rate summary.

    Args:
        times: List of runtime values
        acceptance_rates: List of acceptance rates (optional)
        n_temps: Number of temperatures (optional)
        max_temp: Maximum temperature (optional)
        n_gibbs_per_temp: Number of Gibbs steps per temperature (optional)
    """
    print(f"\nMean runtime / chain: {np.mean(times):.2f}s")

    if acceptance_rates is not None:
        print(f"Average acceptance rate: {np.mean(acceptance_rates):.2%}")

    if n_temps is not None and max_temp is not None and n_gibbs_per_temp is not None:
        print(f"Parameters: {n_temps} temperatures, max_temp={max_temp}")
        print(f"            {n_gibbs_per_temp} Gibbs steps per temperature")


def create_output_message(data_name):
    """Create standardized output message for diagnostic files.

    Args:
        data_name: Name of the dataset

    Returns:
        Formatted output message string
    """
    return f"\nDiagnostic PNGs saved in ../figures/{data_name}"
