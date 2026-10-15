# Stationary Strings Theorem V: Arrest of the Navier–Stokes Vortex Singularity by Elastic String Tension

> **Türkçe:** [README_TR.md](./README_TR.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.YYYYYYY.svg)](https://doi.org/10.5281/zenodo.YYYYYYY)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Status:** Unpublished preprint (v3, October 15, 2026). Not peer-reviewed.
> All numerical results are exactly reproducible from the code in this repo (`python duran_sicimler_sim/solver_engine.py` — T1–T12).

A **model study** adding one-dimensional elastic string (Oldroyd-B / Polyakov filament) tension dynamics
to the September 2026 Navier–Stokes inner-core equations ([1], Sections 3.1 and 4.1),
with high-precision numerical verification (`SciPy Radau`, `DOP853`, `brentq`, $10^{-14}$ tolerance).

**Key findings (details in `duran_sicimler_sim/SONUC_RAPORU.md`):**

| Finding | Result |
|---|---|
| Explicit-model singularity arrest | $s_{\mathrm{kritik}} \in [16.12,\, 58.93]$, finite $u_{\max} \in [5.9\times10^{2},\, 1.0\times10^{12}]$ (16 parameter pairs) |
| Fully self-consistent model | No finite-$s$ zero crossing; asymptotic saturation $\Pi_{\mathrm{eff}} \approx 0.181$–$0.187$ at $s=40$ |
| Theorem 2: $X \to 0$ axis regularization | $\ell_\theta=\sqrt{X/X_*}\,\tilde{\ell}$ cancels the $1/X$ pole; $\Pi_{\mathrm{reg}}(0)=0.952236996205853$ (Test T6) |
| Theorem 3: balance-subtracted global model (T7) | Cutoff-free convergence on $[0,\infty)$; cross-check $2.20\times10^{-14}$ |
| Proposition 4.2 transport operators (T8) | Tangential stress residual $\|T_0\|_\infty$: $0.5447 \to 0.4666$ (14.34% reduction) |
| Full $(X,s)$ PDE + FENE-P (T9) | fig5; Table 8 ODE–PDE agreement |
| Theorem 4 rational curvature (T10) | PDE wake convergence |
| Theorem 5 two-field PDE + Lyapunov (T11) | Pointwise power cancellation exact zero; quenching + positivity |
| Lagrange spectral transfer (T12) | $k_{\max}==n$ 4/4, 12/12 pass |
| Acoustic event horizon | $q_{\mathrm{ufuk}} \in [9.8\times10^{-4},\, 1.2\times10^{-2}]$ |
| Topological invariant $\mathcal{I}_n = n c_s$ | $\max\|\mathcal{I}_n - n c_s\| \le 7.11\times10^{-15}$ (machine precision) |

## Repository layout

```
.
├── paper/
│   ├── duran_sicimler_teoremi_V.tex      # Paper source (Turkish)
│   ├── duran_sicimler_teoremi_V.pdf     # Compiled paper, TR
│   ├── duran_sicimler_teoremi_V_EN.tex  # Paper source (English)
│   └── duran_sicimler_teoremi_V_EN.pdf  # Compiled paper, EN
├── duran_sicimler_sim/
│   ├── solver_engine.py               # Numerical solver (Radau/brentq/DOP853)
│   ├── generate_figures.py            # Generates publication figures
│   ├── SONUC_RAPORU.md                # Numerical findings + LaTeX tables
│   └── fig1_fig6.{pdf,png}            # Generated figures
├── CITATION.cff
└── LICENSE
```

## Reproduction

```bash
pip install numpy scipy matplotlib
python duran_sicimler_sim/solver_engine.py      # verification tests (T1–T12)
python duran_sicimler_sim/generate_figures.py   # figures (PDF + 300 DPI PNG)
```

Compiling the paper (MiKTeX / TeX Live):

```bash
cd paper
pdflatex -interaction=nonstopmode duran_sicimler_teoremi_V.tex
pdflatex -interaction=nonstopmode duran_sicimler_teoremi_V.tex
```

The build log contains no `!` errors and no `Overfull` warnings.

## Citation

```bibtex
@misc{meral2026duranV,
  author       = {Fahri Meral},
  title        = {Stationary Strings Theorem V: Arrest of the September 2026 Navier-Stokes
                   Vortex Singularity by Elastic String Tension, $X \to 0$ Axis
                   Regularization and Self-Consistent Saturation Proof},
  year         = {2026},
  month        = oct,
  doi          = {10.5281/zenodo.YYYYYYY},
  url          = {https://github.com/fahrim/stationary-strings-v}
}
```

See also `CITATION.cff`. (The DOI will be filled in once the first Zenodo release is published.)

## License

- **Code** (`duran_sicimler_sim/`): [MIT](LICENSE)
- **Paper text and figures** (`paper/`, `*.pdf`, `*.png`, `SONUC_RAPORU.md`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — shareable and adaptable with attribution.

## Note

This repo is a **priority and transparency record**: v1 is archived as-is, corrections arrive as new releases (v2, v3…); the v1 record is never deleted. Content is not peer-reviewed; findings are numerical results under the stated model assumptions ($F$ reference profile, $X_*=1.0$, $X \ge 0.05$).
