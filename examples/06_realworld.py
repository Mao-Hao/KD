"""Example 06 - Real-world PDE discovery showcase (FD + autograd).

End-to-end pipeline on real (non-synthetic) PDE benchmark data, in
BOTH derivative modes:

  - finite_diff: fast, works well on clean grid data
  - autograd: slower (trains a neural-network surrogate first),
                   more robust on noisy/sparse data

Each (dataset, mode) pair gets its own output directory:
  examples/out/realworld/<dataset>/fd/report.html
  examples/out/realworld/<dataset>/nn/report.html

Datasets covered (all bundled with kd2 — no internet required):
  - chafee-infante: u_t = u_xx - u + u^3
  - kdv: u_t = -u * u_x - 0.0025 * u_xxx
  - pde-divide: u_t = -u_x / x + 0.25 * u_xx
  - pde-compound: u_t = u * u_xx + u_x^2

Runtime: ~12-15 minutes total on a modern CPU (FD ~30s/dataset,
autograd ~3 min/dataset). Not part of the smoke suite — run it
manually when you want to inspect figures.

For comparison: the SGA-PDE paper trains the NN surrogate for 100,000
epochs. We use 15,000 here as a budget compromise; bump to 50,000 +
if autograd results look noisy.

Run: python examples/06_realworld.py
      open examples/out/realworld/chafee_infante/fd/report.html
"""

from pathlib import Path

import kd2

OUT_BASE = Path(__file__).parent / "out" / "realworld"

# Datasets to run. All four are bundled benchmarks from the SGA-PDE paper.
DATASETS = [
    kd2.load_chafee_infante,
    kd2.load_kdv,
    kd2.load_pde_divide,
    kd2.load_pde_compound,
]

# Per-mode SGA settings. Real benchmarks need a wider search than synthetic
# Burgers; KdV in particular has small (~0.0025) coefficients that need a
# generous AIC budget to survive STRidge pruning.
GENERATIONS = 80
POPULATION = 20
DEPTH = 4
WIDTH = 5
AUTOGRAD_TRAIN_EPOCHS = 15_000 # Paper uses 100k; bump if autograd looks noisy.

results: list[
    tuple[str, str, str, str, float]
] = [] # (dataset, mode, truth, found, aic)

for loader in DATASETS:
    dataset = loader()
    print()
    print(f"=== {dataset.name} ===")
    print(f"Ground truth: {dataset.ground_truth}")
    kd2.preview(dataset)

    for mode_label, derivatives in [("fd", "finite_diff"), ("nn", "autograd")]:
        print(f"\n--- {dataset.name} / {mode_label} ---")
        model = kd2.Model(
            algorithm="sga",
            generations=GENERATIONS,
            population=POPULATION,
            depth=DEPTH,
            width=WIDTH,
            derivatives=derivatives,
            autograd_train_epochs=AUTOGRAD_TRAIN_EPOCHS,
            seed=0,
            verbose=False,
        )
        model.fit(dataset)

        print(f"Discovered: {model.best_expr_}")
        print(f"Best AIC: {model.best_score_:.4f}")

        out_dir = OUT_BASE / dataset.name.replace("-", "_") / mode_label
        viz = kd2.VizEngine(output_dir=out_dir)
        report = viz.render_all(
            model.result_,
            algorithm=model.algorithm_,
            dataset=dataset,
        )
        print(f"Report: {report.report}")
        print(f"Figures: {len(report.figures)} files")

        results.append(
            (
                dataset.name,
                mode_label,
                dataset.ground_truth or "",
                model.best_expr_,
                model.best_score_,
            )
        )

# Summary table grouped by dataset, side-by-side FD vs NN.
print()
print("=" * 80)
print("SUMMARY (lower AIC = better fit)")
print("=" * 80)
current = ""
for name, mode, truth, found, aic in results:
    if name != current:
        print()
        print(name)
        print(f" Ground truth: {truth}")
        current = name
    print(f" [{mode:>2}] AIC={aic:>9.4f} | {found}")
print()
print(f"All reports saved under: {OUT_BASE}")
