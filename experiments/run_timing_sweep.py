"""Paired horizon sweep for the known-resource, two-stage timing instance.

The original generator and main figure are preserved. Non-horizon-tuned
controllers are run once on each longest stream; Hedge and SA-EAFR-OMD are
recomputed with the specified horizon at every sweep point. All metrics use
exact occupancy probabilities, not sampled trajectories.

Run from the package root: python experiments/run_timing_sweep.py
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import run_experiments as base
import numpy as np
from scipy.optimize import linprog
from scipy.stats import t
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'results'
HORIZONS = (64, 128, 256, 512, 1024, 2048, 4096)
METHOD_IDS = [
    'reference', 'departure_optimistic', 'rolling', 'fixed_penalty',
    'episode_hedge', 'eafr_omd', 'sa_eafr_omd',
    'ignore_ignore', 'ignore_trust', 'trust_ignore', 'trust_trust',
]
METHOD_LABELS = [
    'Forecast-blind OMD (h=0)', 'Departure-only optimistic OMD (h=M0)',
    'Rolling replanning', 'Fixed-penalty replanning (mu=1)',
    'Episode Hedge (reference/rolling)', 'EAFR-OMD', 'SA-EAFR-OMD',
    'Ignore early / ignore late', 'Ignore early / trust late',
    'Trust early / ignore late', 'Trust early / trust late',
]
AUDIT = {
    'controller_trace_episodes': 0, 'evaluated_horizon_episodes': 0,
    'closed_form_pattern_lp_checks': 0, 'comparator_lp_checks': 0,
    'max_occupancy_feasibility_residual': 0.,
    'max_executed_occupancy_residual': 0.,
    'max_loss_reconstruction_residual': 0.,
    'max_eafr_late_formula_residual': 0.,
    'max_eafr_resource_equality_residual': 0.,
    'max_reference_spend_or_null_mass': 0.,
    'max_eafr_early_score': 0., 'max_eafr_late_score': 0.,
    'max_score_increment_excess': 0.,
    'max_maximum_horizon_reproduction_residual': 0.,
    'max_pattern_equivalence_residual': 0.,
    'max_closed_form_pattern_lp_residual': 0.,
    'max_comparator_lp_residual': 0.,
    'legacy_256_runs_reproduced': 0,
    'max_legacy_256_loss_residual': 0.,
}


def check_max(key, value, tolerance=2e-7):
    value = float(value)
    AUDIT[key] = max(AUDIT.get(key, 0.), value)
    assert value <= tolerance, (key, value, tolerance)


def check_occupancies(q):
    q = np.asarray(q)
    residual = max(float(np.abs(q @ base.EQ.T - 1.).max()),
                   float(np.maximum(q @ base.G - 1., 0.).max()),
                   float(np.maximum(-q, 0.).max()))
    check_max('max_occupancy_feasibility_residual', residual)


def costs_from_bits(bits):
    c = np.zeros((len(bits), 5))
    c[:, 2] = bits
    c[:, 3] = 1 - bits
    c[:, 4] = 1.
    return c


def trust_late(q, bits):
    """Unique late LP solution with the first-stage occupancy held fixed."""
    out = np.zeros_like(q)
    out[:, :2] = q[:, :2]
    x = q[:, 0]
    out[:, 2] = (1. - x) * (1. - bits)
    out[:, 3] = (1. - x) * bits
    out[:, 4] = x
    return out


def pattern_occupancies(trace):
    """A fixed pattern skips or performs each unpenalized stage replan.

    Each episode starts at the common forecast-blind reference. The pattern is
    fixed across episodes, but that reference still learns from past losses.
    Selecting this pattern before the episode does not prevent its late action
    from reading the late broadcast when its second bit is 'trust'.
    """
    reference = trace['occupancies'][:, 0]
    n = len(reference)
    patterns = np.zeros((n, 4, 5))
    patterns[:, 0] = reference
    patterns[:, 1] = trust_late(reference, trace['bits'])
    # The early objective is 1-x; its unique feasible minimizer spends all
    # resource and is forced to use null later, whether or not late is trusted.
    patterns[:, 2, 0] = patterns[:, 3, 0] = 1.
    patterns[:, 2, 4] = patterns[:, 3, 4] = 1.
    check_occupancies(patterns)
    check_max('max_pattern_equivalence_residual',
              np.abs(patterns[:, 3] - trace['occupancies'][:, 2]).max())
    # Validate the closed forms against the existing LP solver on fixed,
    # evenly spaced episodes, rather than add redundant solves to the sweep.
    early = base.replan(reference[0], np.array([0., 1., 0., 0., 0.]), 0., 0)
    check_max('max_closed_form_pattern_lp_residual', np.abs(early-patterns[0, 2]).max())
    AUDIT['closed_form_pattern_lp_checks'] += 1
    for k in np.unique(np.linspace(0, n-1, min(16, n), dtype=int)):
        c = costs_from_bits(trace['bits'][k:k+1])[0]
        q = base.replan(reference[k], c, 0., 1)
        check_max('max_closed_form_pattern_lp_residual', np.abs(q-patterns[k, 1]).max())
        AUDIT['closed_form_pattern_lp_checks'] += 1
    return patterns


def audit_trace(trace):
    q = trace['occupancies']
    c = costs_from_bits(trace['bits'])
    early = trace['early_eafr']
    eafr = q[:, 5]
    n = len(q)
    check_occupancies(q)
    check_occupancies(early)
    check_max('max_loss_reconstruction_residual',
              np.abs(np.einsum('nmk,nk->nm', q, c)-trace['loss'][:, :6]).max())
    check_max('max_eafr_late_formula_residual',
              np.abs(eafr-trust_late(early, trace['bits'])).max())
    check_max('max_eafr_resource_equality_residual', np.abs(eafr@base.G-1.).max())
    check_max('max_reference_spend_or_null_mass', np.abs(q[:, 0, [0, 4]]).max())
    # One state at each layer makes policy propagation exact: retain the early
    # block, normalize the final late block, and propagate unit state mass.
    executed = np.c_[early[:, :2], eafr[:, 2:]/eafr[:, 2:].sum(1)[:, None]]
    check_max('max_executed_occupancy_residual', np.abs(executed-eafr).max())
    score = trace['stage_score_history']
    increments = np.diff(np.vstack([np.zeros((1, 2)), score]), axis=0)
    check_max('max_score_increment_excess', max(0., float(increments[:, 0].max())-.25))
    check_max('max_eafr_late_score', np.abs(score[:, 1]).max())
    AUDIT['max_eafr_early_score'] = max(AUDIT['max_eafr_early_score'], float(score[:, 0].max()))
    assert np.all(score[:, 0] <= np.arange(1, n+1)/4. + 2e-7)
    early_damage = np.maximum(np.einsum('nk,nk->n', c, early-q[:, 0]), 0.)
    assert early_damage.sum() <= 8.*math.sqrt(score[-1, 0]) + 1e-5
    assert early_damage.sum() <= 4.*math.sqrt(n) + 1e-5
    AUDIT['controller_trace_episodes'] += n


def evaluate_horizon(trace, patterns, n):
    c = costs_from_bits(trace['bits'][:n])
    q = np.zeros((n, len(METHOD_IDS), 5))
    q[:, :6] = trace['occupancies'][:n]
    q[:, 7:] = patterns[:n]
    hedge = np.full(2, .5)
    hedge_eta = math.sqrt(8.*math.log(2)/n)/2.
    master = base.SharedControllerMaster(n, 2.)
    hedge_weights = np.zeros((n, 2))
    master_weights = np.zeros((n, 3))
    for k in range(n):
        hedge_weights[k] = hedge
        q[k, 4] = hedge @ q[k, [0, 2]]
        # These are exactly the weights used by combine(), before feedback.
        alpha = ((master.W + master.pi)*master.rates).sum(1)
        alpha /= alpha.sum()
        master_weights[k] = alpha
        q[k, 6] = alpha @ q[k, [0, 5, 2]]
        loss = q[k] @ c[k]
        mixed_loss = master.combine(loss[[0, 5, 2]])
        check_max('max_loss_reconstruction_residual', abs(mixed_loss-loss[6]))
        hedge *= np.exp(-hedge_eta*loss[[0, 2]])
        hedge /= hedge.sum()
    check_occupancies(q)
    loss = np.einsum('nmk,nk->nm', q, c)
    resource = q @ base.G
    damage = np.maximum(loss-loss[:, :1], 0.)
    violation = np.maximum(resource-1., 0.)
    comparator = float(min(trace['bits'][:n].sum(), n-trace['bits'][:n].sum()))
    lp = linprog(c.sum(0), A_ub=base.G[None, :], b_ub=[1.],
                 A_eq=base.EQ, b_eq=np.ones(2), bounds=[(0., 1.)]*5, method='highs')
    assert lp.success, lp.message
    check_max('max_comparator_lp_residual', abs(comparator-lp.fun))
    AUDIT['comparator_lp_checks'] += 1
    check_max('max_pattern_equivalence_residual',
              max(float(np.abs(loss[:, 0]-loss[:, 7]).max()),
                  float(np.abs(loss[:, 2]-loss[:, 9]).max()),
                  float(np.abs(loss[:, 2]-loss[:, 10]).max()),
                  float(np.abs(loss[:, 8]).max())))
    if n == len(trace['loss']):
        check_max('max_maximum_horizon_reproduction_residual',
                  np.abs(loss[:, :7]-trace['loss']).max(), 2e-10)
    AUDIT['evaluated_horizon_episodes'] += n
    return {'loss': loss, 'damage': damage, 'violation': violation,
            'resource': resource, 'regret': loss.sum(0)-comparator,
            'comparator': comparator, 'hedge_weights': hedge_weights,
            'master_weights': master_weights}


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)/math.sqrt(len(values)))


def write_table(stem, rows):
    (OUT/(stem+'.json')).write_text(json.dumps(rows, indent=2), encoding='utf8')
    with (OUT/(stem+'.csv')).open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summaries(runs_by_horizon):
    rows = []
    paired = []
    for n, runs in runs_by_horizon.items():
        totals = np.array([r['loss'].sum(0) for r in runs])
        regrets = np.array([r['regret'] for r in runs])
        damage = np.array([r['damage'].sum(0) for r in runs])
        violations = np.array([r['violation'].sum(0) for r in runs])
        for i, (method_id, label) in enumerate(zip(METHOD_IDS, METHOD_LABELS)):
            row = {'episodes': n, 'runs': len(runs), 'method_id': method_id, 'method': label}
            for key, values in [('cumulative_loss', totals[:, i]),
                                ('average_loss', totals[:, i]/n),
                                ('static_regret', regrets[:, i]),
                                ('average_static_regret', regrets[:, i]/n),
                                ('forecast_downside', damage[:, i]),
                                ('positive_expected_resource_violation', violations[:, i])]:
                row[key+'_mean'], row[key+'_se'] = mean_se(values)
            rows.append(row)
        # Paired descriptive intervals, with one stream shared across methods.
        # Subtracting the same realized comparator makes regret differences
        # exactly equal to loss differences within each seed.
        for j in [0, 1, 2, 3, 4, 6, 8]:
            difference = totals[:, 5]-totals[:, j]
            mean, se = mean_se(difference)
            radius = float(t.ppf(.975, len(runs)-1))*se
            assert np.max(np.abs(difference-(regrets[:, 5]-regrets[:, j]))) < 1e-8
            paired.append({'episodes': n, 'paired_runs': len(runs),
                           'contrast': 'EAFR-OMD minus '+METHOD_LABELS[j],
                           'cumulative_loss_difference_mean': mean,
                           'cumulative_loss_difference_se': se,
                           'ci95_low': mean-radius, 'ci95_high': mean+radius,
                           'average_loss_difference_mean': mean/n,
                           'average_loss_ci95_low': (mean-radius)/n,
                           'average_loss_ci95_high': (mean+radius)/n,
                           'static_regret_difference_mean': mean})
    return rows, paired


def plot_sweep(rows, horizons, stem):
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.8))
    shown = [0, 1, 2, 3, 4, 5, 6, 8]
    labels = ['Reference / I-I', r'Departure ($h=M^0$)',
              'Rolling / T-I / T-T', 'Fixed penalty', 'Episode Hedge',
              'EAFR-OMD', 'SA-EAFR-OMD', 'I-T (selective)']
    colors = ['#777777', '#CC79A7', '#D55E00', '#56B4E9',
              '#E69F00', '#0072B2', '#000000', '#009E73']
    styles = ['-', '--', '-', ':', '-.', '-', '--', ':']
    for i, label, color, style in zip(shown, labels, colors, styles):
        data = [next(r for r in rows if r['episodes']==n and r['method_id']==METHOD_IDS[i])
                for n in horizons]
        for axis, key in zip(axes, ['cumulative_loss', 'average_loss', 'static_regret']):
            mean = np.array([r[key+'_mean'] for r in data])
            se = np.array([r[key+'_se'] for r in data])
            axis.plot(horizons, mean, label=label, color=color, linestyle=style,
                      linewidth=1.4 if i in [5, 6, 8] else 1.1)
            axis.fill_between(horizons, mean-se, mean+se, color=color, alpha=.1, linewidth=0.)
    for axis, title in zip(axes, ['(a) Cumulative loss', '(b) Average loss', '(c) Static regret']):
        axis.set_xscale('log', base=2)
        ticks = horizons[::2]
        if horizons[-1] not in ticks:
            ticks = ticks + [horizons[-1]]
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(n) for n in ticks])
        axis.set(xlabel='Episodes N', title=title)
        axis.grid(alpha=.15, linewidth=.5)
    axes[0].set_ylabel('Exact occupancy loss')
    axes[1].set_ylabel('Loss / N')
    axes[2].set_ylabel('Loss - best fixed policy')
    axes[1].set_ylim(-.035, 1.04)
    axes[2].axhline(0., color='#777777', linewidth=.5, zorder=0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, 1.01),
               ncol=4, frameon=False, columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0., 0., 1., .82), w_pad=1.2)
    for suffix in ['pdf', 'png']:
        fig.savefig(ROOT/'figures'/(stem+'.'+suffix), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizons', type=int, nargs='+', default=list(HORIZONS))
    parser.add_argument('--seeds', type=int, default=10, help='Use paired seeds 0,...,seeds-1.')
    parser.add_argument('--output-stem', default='timing_sweep')
    args = parser.parse_args()
    horizons = sorted(set(args.horizons))
    assert args.seeds >= 2 and horizons[0] >= 2
    assert all(ch.isalnum() or ch=='_' for ch in args.output_stem)
    started = time.perf_counter()
    numerical_checks = base.checks()
    runs_by_horizon = {n: [] for n in horizons}
    traces = []
    patterns_by_seed = []
    legacy_file = OUT/'raw_runs.npz'
    legacy = None
    if legacy_file.exists():
        with np.load(legacy_file) as data:
            legacy = data['separation_loss'].copy()
    preserved = [OUT/'raw_runs.npz', OUT/'summary.json', OUT/'verification.json',
                 ROOT/'figures'/'mechanisms.pdf', ROOT/'figures'/'mechanisms.png']
    before_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in preserved if p.exists()}
    for seed in range(args.seeds):
        trace = base.separation(seed, max(horizons), record_details=True)
        audit_trace(trace)
        patterns = pattern_occupancies(trace)
        traces.append(trace)
        patterns_by_seed.append(patterns)
        for n in horizons:
            result = evaluate_horizon(trace, patterns, n)
            runs_by_horizon[n].append(result)
            if n==256 and legacy is not None and seed<len(legacy):
                check_max('max_legacy_256_loss_residual',
                          np.abs(result['loss'][:, :7]-legacy[seed]).max(), 2e-9)
                AUDIT['legacy_256_runs_reproduced'] += 1
        print('timing sweep seed', seed, 'done; maximum N =', max(horizons), flush=True)
    rows, paired = summaries(runs_by_horizon)
    write_table(args.output_stem+'_summary', rows)
    write_table(args.output_stem+'_paired', paired)
    arrays = {
        'seeds': np.arange(args.seeds), 'horizons': np.array(horizons),
        'method_ids': np.asarray(METHOD_IDS), 'method_labels': np.asarray(METHOD_LABELS),
        'bits': np.array([r['bits'] for r in traces]),
        'longest_trace_original_method_names': np.asarray(base.METHODS),
        'longest_trace_original_occupancies': np.array([r['occupancies'] for r in traces]),
        'eafr_early_occupancies': np.array([r['early_eafr'] for r in traces]),
        'eafr_stage_score_history': np.array([r['stage_score_history'] for r in traces]),
        'eafr_local_available_opportunity': np.array([r['opportunity'] for r in traces]),
        'eafr_local_captured_opportunity': np.array([r['captured'] for r in traces]),
        'fixed_pattern_names': np.asarray(METHOD_LABELS[7:]),
        'fixed_pattern_occupancies': np.array(patterns_by_seed),
    }
    for n, runs in runs_by_horizon.items():
        for key in ['loss', 'damage', 'violation', 'resource', 'regret',
                    'comparator', 'hedge_weights', 'master_weights']:
            arrays[key+'_N'+str(n)] = np.array([r[key] for r in runs])
    np.savez_compressed(OUT/(args.output_stem+'_raw.npz'), **arrays)
    plot_sweep(rows, horizons, args.output_stem)
    after_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in preserved if p.exists()}
    assert before_hashes == after_hashes
    assert AUDIT['controller_trace_episodes'] == args.seeds*max(horizons)
    assert AUDIT['evaluated_horizon_episodes'] == args.seeds*sum(horizons)
    assert base.STATS['shared_master_mass_checks'] == args.seeds*(max(horizons)+sum(horizons))
    report = {
        'all_checks_passed': True, 'horizons': horizons,
        'seeds': list(range(args.seeds)),
        'pairing': 'Common fair-bit stream prefix across methods and horizons within each seed.',
        'resource_model': 'Known deterministic g=(1,0,1,1,0), B=1; K_n=Q_B.',
        'initialization': [0., 1., .5, .5, 0.],
        'static_comparator': 'min(sum(Z_n), N-sum(Z_n)); independently checked by LP at each horizon.',
        'horizon_tuned_methods': ['episode_hedge', 'sa_eafr_omd'],
        'other_methods_use_identical_paired_prefixes': True,
        'pattern_definition': 'Start each episode at the common forecast-blind reference; ignore skips a stage update, trust performs an unpenalized prefix-feasible replan.',
        'pattern_equivalences': {'ignore_ignore': 'reference', 'ignore_trust': 'zero-loss selective causal policy (up to solver residual)', 'trust_ignore': 'rolling', 'trust_trust': 'rolling'},
        'information_scope': 'A rule selected before the episode can still use late broadcasts; only policies whose late action law cannot use the current late information are late-blind.',
        'uncertainty': 'One sample standard error across exogenous seeds; paired Student-t 95% intervals are descriptive and unadjusted for multiplicity.',
        'numerical_checks': numerical_checks, 'occupancy_and_timing_audit': AUDIT,
        'solver': base.STATS, 'original_outputs_preserved_sha256': after_hashes,
        'python': sys.version, 'platform': platform.platform(),
        'processor': platform.processor(), 'numpy': np.__version__,
        'scipy': __import__('scipy').__version__,
        'matplotlib': __import__('matplotlib').__version__,
        'seconds': time.perf_counter()-started,
        'limitations': [
            'Finite synthetic sweep; not an empirical proof of an asymptotic rate.',
            'Known-resource specialization; does not test unknown-resource learning.',
            'Exact occupancy losses and expected-budget feasibility; no pathwise safety claim.',
            'Fixed-pattern ablations are not a general regret guarantee against all stage patterns.',
            'No Roy-Das FAG-K baseline is claimed or implemented without verified algorithm equations.',
        ],
    }
    (OUT/(args.output_stem+'_verification.json')).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
