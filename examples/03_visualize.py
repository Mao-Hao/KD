"""Example 03 - Visualize an SGA fit with the HTML report.

After ``model.fit(...)``, kd2 can render a multi-figure HTML report
showing convergence curves, term importance, residuals, and more.

Uses the built-in Chafee-Infante benchmark (real PDE data bundled
with kd2). Output goes to ``examples/out/chafee_infante/``.

Run: python examples/03_visualize.py
      open examples/out/chafee_infante/report.html
"""

from pathlib import Path

import kd2

# 1. Load a real benchmark dataset bundled with kd2.
dataset = kd2.load_chafee_infante()
print(f"Loaded: {dataset.name}, ground truth: {dataset.ground_truth}")
kd2.preview(dataset)

# 2. Fit. Chafee-Infante has a richer term library than Burgers, so we
# bump the SGA budget above the defaults.
model = kd2.Model(
    algorithm="sga",
    generations=60,
    population=20,
    seed=0,
    verbose=False,
)
model.fit(dataset)

print()
print(f"Discovered: {model.best_expr_}")
print(f"Best AIC: {model.best_score_:.4f}")

# 3. Render the HTML report. We pass the fitted SGA plugin
# (``model.algorithm_``) so plugin-specific figures are included
# alongside the universal ones.
out_dir = Path(__file__).parent / "out" / "chafee_infante"
viz = kd2.VizEngine(output_dir=out_dir)
report = viz.render_all(
    model.result_,
    algorithm=model.algorithm_,
    dataset=dataset,
)

print(f"Report: {report.report}")
print(f"Figures: {len(report.figures)} files")
