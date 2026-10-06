# Reproduce the synthetic EAFR–OMD results

Use Python 3.8 with the versions in requirements.txt (the recorded run used Python 3.8.2).

From the package root:

    python -m pip install -r experiments/requirements.txt
    python experiments/run_experiments.py
    python experiments/verify_occupancies.py

run_experiments.py generates the three tests, their exact-occupancy metrics, all raw run arrays (including method-name arrays), the vector figure, matched-seed contrasts, and numerical verification records. When the repository-level `.vendor` directory is present, Python 3.8 adds it to the import path automatically. The recorded run takes about 30 seconds; timing is hardware dependent. verify_occupancies.py separately checks causal occupancy reconstruction, feasibility, and opportunity inequalities on small CMDPs with stochastic transitions.

No external datasets or trained predictors are needed. Full loss and resource vectors are synthetic and available to every method after each episode. Fixed-penalty replanning uses mu=1 without test-set tuning. The display method SA–EAFR–OMD is the compressed every-start shared-controller master over three persistent outputs: Reference, EAFR–OMD, and Rolling. Its aggregate controller-rate weights start at zero, receive mass `pi=1/(3*N*(J+1))` every episode, and satisfy `W.sum()==n/N` after every update. The script also verifies exact agreement with explicitly stored every-start specialists. The older fresh-copy EAFR–OMD master is retained as a separate recovery baseline. The change to exact forecasts is not provided to either master.

Loss, regret, downside and violation use exact probability occupancies. Means and standard errors summarize variation over exogenous random streams, not sampled-trajectory noise. The experiments illustrate mechanisms; they do not establish asymptotic rates or physical routing validity.

The main manuscript and its experiment appendix specify the complete generators, comparator sets, feedback rules, and score-update timing. Results/verification.json records dependency versions, seed ranges, run time, and solver checks. Results/occupancy_audit.json records the additional stochastic-flow audit. Output directory names in the actual package are lowercase `results/`.

The recovery run checks the shared master's exact-suffix bound

    L*log(3*N*(J+1))/log(3/2)

in every seed and records its observed maximum and minimum slack in `results/verification.json`. Both masters have signed recovery guarantees; do not apply the base EAFR–OMD positive-downside theorem to them. The source code has no separately assigned release license. Third-party numerical packages retain their licenses.
