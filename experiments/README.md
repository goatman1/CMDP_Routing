# Reproduce the synthetic EAFR–OMD results

Use Python 3.8 with the versions in requirements.txt (the recorded run used Python 3.8.2).

From the package root:

    python -m pip install -r experiments/requirements.txt
    python experiments/run_experiments.py
    python experiments/run_timing_sweep.py
    python experiments/verify_occupancies.py

run_experiments.py generates the three tests, their exact-occupancy metrics, all raw run arrays (including method-name arrays), the vector figure, matched-seed contrasts, and numerical verification records. When the repository-level `.vendor` directory is present, Python 3.8 adds it to the import path automatically. The recorded run takes about 30 seconds; timing is hardware dependent. verify_occupancies.py separately checks causal occupancy reconstruction, feasibility, and opportunity inequalities on small CMDPs with stochastic transitions.

No external datasets or trained predictors are needed. Full loss and resource vectors are synthetic and available to every method after each episode. Fixed-penalty replanning uses mu=1 without test-set tuning. The display method SA–EAFR–OMD is the compressed every-start shared-controller master over three persistent outputs: Reference, EAFR–OMD, and Rolling. Its aggregate controller-rate weights start at zero, receive mass `pi=1/(3*N*(J+1))` every episode, and satisfy `W.sum()==n/N` after every update. The script also verifies exact agreement with explicitly stored every-start specialists. The older fresh-copy EAFR–OMD master is retained as a separate recovery baseline. The change to exact forecasts is not provided to either master.

Loss, regret, downside and violation use exact probability occupancies. Means and standard errors summarize variation over exogenous random streams, not sampled-trajectory noise. The experiments illustrate mechanisms; they do not establish asymptotic rates or physical routing validity.

The main manuscript and its experiment appendix specify the complete generators, comparator sets, feedback rules, and score-update timing. Results/verification.json records dependency versions, seed ranges, run time, and solver checks. Results/occupancy_audit.json records the additional stochastic-flow audit. Output directory names in the actual package are lowercase `results/`.

The recovery run checks the shared master's exact-suffix bound

    L*log(3*N*(J+1))/log(3/2)

in every seed and records its observed maximum and minimum slack in `results/verification.json`. Both masters have signed recovery guarantees; do not apply the base EAFR–OMD positive-downside theorem to them. The public source repository, https://github.com/goatman1/CMDP_Routing, uses the MIT License. Third-party numerical packages retain their licenses.

## Paired timing sweep

`run_timing_sweep.py` evaluates N = 64, 128, 256, 512, 1,024, 2,048, and 4,096 using seeds 0 through 9. Within each seed, every method uses the same fair-bit stream, and horizons use nested prefixes. Hedge and SA–EAFR–OMD are recomputed with their correct horizon-dependent rates and priors. Other controllers use the same prefix of the longest run. This sweep is more expensive than the original three tests because it uses 40,960 two-stage episodes with numerical optimization.

The four fixed patterns begin from the common forecast-blind reference each episode. “Ignore” skips a stage update; “trust” performs an unpenalized feasible replan. On this specific instance, ignore/ignore equals the reference, ignore/trust is the zero-loss selective policy, and trust/ignore and trust/trust equal rolling. A pattern selected at episode start can still use the later broadcast. These ablations do not establish regret against every possible pattern.

The output `results/timing_sweep_raw.npz` includes method labels, bits, occupancies, scores, losses, realized comparator losses, resource use, and master weights. Summary CSV/JSON files report cumulative and per-episode loss, signed static regret, downside, and positive expected-resource violation. Paired confidence intervals are descriptive and are not adjusted for multiple comparisons. The figure is saved as `figures/timing_sweep.pdf` and `.png`.

`results/timing_sweep_verification.json` records solver and occupancy checks, the exact late-replan identity, the early directional-score bound, LP verification of the hindsight comparator, reproduction of the original 256-episode results, and hashes confirming that the original mechanism outputs were preserved. The experiment specializes to known deterministic resources and does not test the unknown-resource bound. It does not claim a reproduced Roy–Das FAG-K baseline.
