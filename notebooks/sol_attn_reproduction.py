# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.15",
#   "plotly==6.9.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Sol-Attn: an executable claim-by-claim reproduction

    Long video generators spend much of their time comparing tokens. Sol-Attn proposes doing exact work only on promising blocks and approximating the rest from block averages.

    **Verdict: partially reproduced.** On four-seed 32K-token tests, proxy correction removed **70.8–94.9%** of the error caused by matched exact-only sparsity while the fused path averaged **1.29–1.32×** dense speed. Kernel memory stayed near dense rather than improving, and a released SANA tensor probe showed that sparsity calibration is distribution dependent.
    """)
    return


@app.cell
def _():
    headline = [
        {"family": "Random", "target": "15%", "error_reduction": 67.73, "speedup": 1.203},
        {"family": "Random", "target": "10%", "error_reduction": 73.95, "speedup": 1.431},
        {"family": "Smooth", "target": "15%", "error_reduction": 94.71, "speedup": 1.218},
        {"family": "Smooth", "target": "10%", "error_reduction": 95.03, "speedup": 1.428},
        {"family": "Temporal", "target": "15%", "error_reduction": 92.32, "speedup": 1.156},
        {"family": "Temporal", "target": "10%", "error_reduction": 92.71, "speedup": 1.424},
    ]
    scaling = [
        {"length_k": 16, "target": "15%", "speedup": 0.832},
        {"length_k": 32, "target": "15%", "speedup": 1.222},
        {"length_k": 64, "target": "15%", "speedup": 1.014},
        {"length_k": 128, "target": "15%", "speedup": 1.113},
        {"length_k": 16, "target": "10%", "speedup": 0.921},
        {"length_k": 32, "target": "10%", "speedup": 1.437},
        {"length_k": 64, "target": "10%", "speedup": 1.188},
        {"length_k": 128, "target": "10%", "speedup": 1.341},
    ]
    robustness = {
        "Random": 70.88,
        "Smooth": 94.87,
        "Temporal": 92.52,
        "Heavy-tail control": 1.28,
    }
    sana_density = {
        "Layer 0": 9.55,
        "Layer 2": 0.00,
        "Layer 6": 1.56,
        "Layer 12": 3.12,
    }
    return headline, robustness, sana_density, scaling


@app.cell
def _(headline):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    primary = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Error reduction vs exact-only",
            "End-to-end speedup over dense",
        ),
    )
    colors = {"15%": "#3E7CB1", "10%": "#73BFB8"}
    for _target in ("15%", "10%"):
        _selected = [row for row in headline if row["target"] == _target]
        primary.add_bar(
            x=[row["family"] for row in _selected],
            y=[row["error_reduction"] for row in _selected],
            name=f"{_target} exact blocks",
            marker_color=colors[_target],
            legendgroup=_target,
            row=1,
            col=1,
        )
        primary.add_bar(
            x=[row["family"] for row in _selected],
            y=[row["speedup"] for row in _selected],
            name=f"{_target} exact blocks",
            marker_color=colors[_target],
            legendgroup=_target,
            showlegend=False,
            row=1,
            col=2,
        )
    primary.add_hline(y=1, line_dash="dash", line_color="#555", row=1, col=2)
    primary.update_yaxes(title_text="L2 error reduction (%)", range=[0, 105], row=1, col=1)
    primary.update_yaxes(title_text="Speedup (×)", range=[0, 1.6], row=1, col=2)
    primary.update_layout(
        title="32K-token Blackwell result · four-seed means",
        barmode="group",
        height=430,
        template="plotly_white",
    )
    primary
    return (go,)


@app.cell
def _(mo):
    mo.md("""
    ## Reconstructing the mechanism

    1. Split queries, keys, and values into 64-token blocks.
    2. Score each block pair with the dot product of its mean query and mean key.
    3. For each query block, select scores above `mean + β × standard deviation`.
    4. Compute selected blocks exactly. For each omitted block, reuse its proxy score and summed value vector as a zeroth-order softmax contribution.

    The key trick is that the threshold's mean and variance come from pooled key moments. The streaming kernel therefore does not write the quadratic proxy-score map. At 128K tokens, the measured routing state was **668.7× smaller** than that diagnostic map, while kernel incremental memory was **1.0004× dense**—routing memory improved, total kernel memory did not.
    """)
    return


@app.cell
def _(mo):
    family = mo.ui.dropdown(
        options=["Random", "Smooth", "Temporal", "Heavy-tail control"],
        value="Random",
        label="Inspect a tensor family",
    )
    family
    return (family,)


@app.cell
def _(family, headline, mo, robustness):
    chosen = family.value
    if chosen == "Heavy-tail control":
        detail = (
            f"**{chosen}:** correction reduced matched exact-only error by only "
            f"**{robustness[chosen]:.2f}%**. This negative control shows that block "
            "means must summarize the omitted signal."
        )
    else:
        _rows = [row for row in headline if row["family"] == chosen]
        detail = (
            f"**{chosen}:** mean error reduction was **{robustness[chosen]:.2f}%**. "
            f"At 15%/10% exact blocks, end-to-end speedups were "
            f"**{_rows[0]['speedup']:.2f}× / {_rows[1]['speedup']:.2f}×**."
        )
    mo.callout(mo.md(detail), kind="info")
    return


@app.cell
def _(go, scaling):
    scale_figure = go.Figure()
    for _target, _color in (("15%", "#3E7CB1"), ("10%", "#19A7A0")):
        _rows = [row for row in scaling if row["target"] == _target]
        scale_figure.add_scatter(
            x=[row["length_k"] for row in _rows],
            y=[row["speedup"] for row in _rows],
            mode="lines+markers",
            name=_target,
            line={"color": _color},
        )
    scale_figure.add_hline(y=1, line_dash="dash", line_color="#555")
    scale_figure.update_layout(
        title="C32 corrected-kernel scaling",
        xaxis_title="Sequence length (K tokens)",
        yaxis_title="End-to-end speedup over dense",
        template="plotly_white",
        height=380,
    )
    scale_figure
    return


@app.cell
def _(mo, sana_density):
    _rows = "\n".join(
        f"| {layer} | {density:.2f}% |" for layer, density in sana_density.items()
    )
    mo.md(
        f"""
        ## Released SANA probe and limits

        We captured QKV tensors from four layers of the public
        `Efficient-Large-Model/Sana_600M_512px_diffusers` checkpoint. Its released
        attention is ReLU-linear rather than the softmax attention assumed here, so
        this is a distribution diagnostic—not video-model validation.

        | Probe | Mean selected density across nominal 10/15% targets |
        |---|---:|
        {_rows}

        The isolated 4K kernel was 1.48–1.58× dense, but routing selected 0–11%
        depending on the layer. Two deeper layers benefited strongly from correction,
        layer 0 did not, and one layer selected no exact blocks.

        **Compute:** Kubernetes; NVIDIA RTX PRO 6000 Blackwell; peak 16 GPUs
        concurrently allocated; 0.43 wall-hours from first launch through the last
        successful terminal run. The experiment does not establish VBench quality,
        full-video inference speed, or the paper's H100/RTX 5090 numbers.
        """
    )
    return


if __name__ == "__main__":
    app.run()
