# pylint: disable=too-many-locals
# pylint: disable=too-many-statements

import time
from pathlib import Path

import numpy as np
from metrics import credible_interval, ess_1d
from samplers import run_chain


def main():
    """
    Executa comparação entre Gibbs Sampler e Tempered Transitions.
    Salva os resultados das cadeias e métricas no diretório data.
    """
    # Carregar dados
    y = np.load("../data/data.npy")

    # Parâmetros globais
    K = 4
    n_iter = 5000
    burn = 1000
    seed = 0

    # Parâmetros do Tempered Transitions
    n_temps = 10
    max_temp = 10
    n_gibbs_per_temp = 1

    print("=== COMPARAÇÃO: GIBBS SAMPLER vs TEMPERED TRANSITIONS ===\n")

    # Executar Gibbs Sampler (n_temps=1, max_temp=1, n_gibbs_per_temp=1)
    print("1. Executando Gibbs Sampler...")
    start_time = time.time()
    gibbs_samples, _, gibbs_acc_rate = run_chain(
        y, K, n_iter, burn, seed, n_temps=1, max_temp=1, n_gibbs_per_temp=1
    )
    gibbs_time = time.time() - start_time
    print(f"   Tempo: {gibbs_time:.2f}s")
    print(f"   Taxa de aceitação: {gibbs_acc_rate:.2%}")
    print(f"   Média das médias: {gibbs_samples.mean(axis=0)}")

    # Executar Tempered Transitions
    print("\n2. Executando Tempered Transitions...")
    start_time = time.time()
    tempered_samples, _, tempered_acc_rate = run_chain(
        y,
        K,
        n_iter,
        burn,
        seed,
        n_temps=n_temps,
        max_temp=max_temp,
        n_gibbs_per_temp=n_gibbs_per_temp,
        keep_last=True,
    )
    tempered_time = time.time() - start_time
    print(f"   Tempo: {tempered_time:.2f}s")
    print(f"   Taxa de aceitação: {tempered_acc_rate:.2%}")
    print(f"   Média das médias: {tempered_samples.mean(axis=0)}")

    # Calcular métricas
    print("\n3. Calculando métricas...")
    gibbs_ci = np.array([credible_interval(gibbs_samples[:, k]) for k in range(K)])
    tempered_ci = np.array(
        [credible_interval(tempered_samples[:, k]) for k in range(K)]
    )

    gibbs_ess = np.array([ess_1d(gibbs_samples[:, k]) for k in range(K)])
    tempered_ess = np.array([ess_1d(tempered_samples[:, k]) for k in range(K)])

    # Criar diretório de resultados se não existir
    results_dir = Path("../data/comparison_results")
    results_dir.mkdir(exist_ok=True)

    # Salvar amostras
    print("\n4. Salvando resultados...")
    np.save(results_dir / "gibbs_samples.npy", gibbs_samples)
    np.save(results_dir / "tempered_samples.npy", tempered_samples)

    # Salvar métricas em um arquivo estruturado
    results = {
        "gibbs": {
            "samples": gibbs_samples,
            "time": gibbs_time,
            "means": gibbs_samples.mean(axis=0),
            "stds": gibbs_samples.std(axis=0),
            "credible_intervals": gibbs_ci,
            "ess": gibbs_ess,
        },
        "tempered": {
            "samples": tempered_samples,
            "time": tempered_time,
            "means": tempered_samples.mean(axis=0),
            "stds": tempered_samples.std(axis=0),
            "credible_intervals": tempered_ci,
            "ess": tempered_ess,
        },
        "parameters": {
            "K": K,
            "n_iter": n_iter,
            "burn": burn,
            "seed": seed,
            "gibbs_params": {"n_temps": 1, "max_temp": 1, "n_gibbs_per_temp": 1},
            "tempered_params": {
                "n_temps": n_temps,
                "max_temp": max_temp,
                "n_gibbs_per_temp": n_gibbs_per_temp,
                "keep_last": True,
            },
        },
    }

    # Salvar como arquivo .npz (mais eficiente para múltiplos arrays)
    np.savez_compressed(
        results_dir / "comparison_results.npz",
        **{key: value for key, value in results.items() if key != "parameters"},
        **results["parameters"],
    )

    # Imprimir resumo comparativo
    print("\n=== RESUMO COMPARATIVO ===")
    print("Gibbs Sampler:")
    print(f"  - Tempo de execução: {gibbs_time:.2f}s")
    print(f"  - Taxa de aceitação: {gibbs_acc_rate:.3f}")
    print(f"  - Médias estimadas: {np.round(gibbs_samples.mean(axis=0), 4)}")
    print(f"  - Desvios padrão: {np.round(gibbs_samples.std(axis=0), 4)}")
    print(f"  - IC 95% inferior: {np.round(gibbs_ci[:, 0], 4)}")
    print(f"  - IC 95% superior: {np.round(gibbs_ci[:, 1], 4)}")
    print(f"  - ESS: {np.round(gibbs_ess, 1)}")

    print("\nTempered Transitions:")
    print(f"  - Tempo de execução: {tempered_time:.2f}s")
    print(f"  - Taxa de aceitação: {tempered_acc_rate:.3f}")
    print(f"  - Médias estimadas: {np.round(tempered_samples.mean(axis=0), 4)}")
    print(f"  - Desvios padrão: {np.round(tempered_samples.std(axis=0), 4)}")
    print(f"  - IC 95% inferior: {np.round(tempered_ci[:, 0], 4)}")
    print(f"  - IC 95% superior: {np.round(tempered_ci[:, 1], 4)}")
    print(f"  - ESS: {np.round(tempered_ess, 1)}")

    print(f"\nRazão de eficiência (Gibbs/Tempered): {gibbs_time / tempered_time:.2f}")
    print(
        f"Razão ESS média (Tempered/Gibbs): {np.mean(tempered_ess) / np.mean(gibbs_ess):.2f}"
    )

    print(f"\nResultados salvos em: {results_dir}")
    print("  - gibbs_samples.npy")
    print("  - tempered_samples.npy")
    print("  - comparison_results.npz")

    return results


if __name__ == "__main__":
    main()
