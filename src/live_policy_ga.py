"""V3 fixed-capacity operating-policy GA. Reuses current-state look-ahead."""
from dataclasses import asdict, dataclass
import time

import numpy as np

from live_hospital_state import OperatingPolicy
from live_canonical import canonical_hash, HASH_SCHEMA
from live_policy_diagnosis import diagnose_live_result, diagnosis_is_current
from live_policy_control import GENES, POLICY_BOUNDS
from live_lookahead import simulate_lookahead_from_current_state, target_values
from live_pressure import LiveTargets, RESOURCES


@dataclass(frozen=True)
class LiveGAOptions:
    population: int = 10
    generations: int = 6
    search_replications: int = 3
    patience: int = 3
    seed: int = 42

    def validate(self):
        for name, low, high in (("population", 4, 40), ("generations", 1, 30),
                               ("search_replications", 1, 10), ("patience", 2, 30)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Live GA {name} must be an integer from {low} to {high}.")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Live GA seed must be a nonnegative 32-bit integer.")


def policy_key(policy):
    return tuple(getattr(policy, gene) for gene in GENES)


def random_policy(rng):
    return OperatingPolicy(**{gene: round(int(rng.integers(round(high / step) + 1)) * step, 2)
        for gene, (_, high, step) in POLICY_BOUNDS.items()})


def crossover_mutate(first, second, rng):
    values = {}
    for gene, (_, high, step) in POLICY_BOUNDS.items():
        value = getattr(first if rng.random() < .5 else second, gene)
        if rng.random() < .25:
            value += int(rng.integers(-5, 6)) * step
        values[gene] = round(np.clip(round(value / step) * step, 0., high), 2)
    policy = OperatingPolicy(**values)
    policy.validate()
    return policy


def live_fitness(evaluation, policy, targets, horizon, baseline=None):
    """Transparent objective; target/robustness terms dominate restriction savings."""
    metrics, runs = evaluation["aggregate"], evaluation["runs"]
    mw = max(targets.mean_wait_target_minutes, 1.)
    hw = max(targets.high_risk_wait_target_minutes, 1.)
    util = max(targets.max_utilization_target, .01)
    failed = np.mean([sum(not passed for passed in row["condition_breakdown"].values()) for row in runs])
    violation = np.mean([max(0., row["mean_wait"] - targets.mean_wait_target_minutes) / mw
        + 2 * max(0., row["high_risk_mean_wait"] - targets.high_risk_wait_target_minutes) / hw
        + sum(max(0., row["resource_pressure"][name]["time_weighted_utilization"] - targets.max_utilization_target) / util
              + max(0., row["resource_pressure"][name]["longest_above_target_streak"] - getattr(targets, "sustained_overload_grace_minutes", 30.)) / max(getattr(targets, "sustained_overload_grace_minutes", 30.), 1.)
              + max(0., row["resource_pressure"][name]["longest_full_saturation_streak"] - getattr(targets, "critical_saturation_grace_minutes", 15.)) / max(getattr(targets, "critical_saturation_grace_minutes", 15.), 1.)
              for name in RESOURCES) for row in runs])
    reserve = sum(getattr(policy, gene) / POLICY_BOUNDS[gene][1] for gene in GENES if "reserve" in gene)
    complexity = sum(getattr(policy, gene) / POLICY_BOUNDS[gene][1] for gene in GENES if "reserve" not in gene)
    nonrecovery = float(np.mean([not row["condition_breakdown"]["recovery_pass"] for row in runs]))
    recovery = metrics["recovery_time_minutes"]
    disruption = 0.
    if baseline is not None and evaluation["robustness"] <= baseline["robustness"]:
        old = baseline["aggregate"]
        disruption = (500 * max(0., metrics["mean_wait"] - old["mean_wait"]) / mw
            + 10 * max(0., metrics["queue_peak"] - old["queue_peak"])
            + 100 * max(0., old["completed_patients"] - metrics["completed_patients"]))
    components = dict(robustness_reward=100 * evaluation["robustness"],
        failed_condition_penalty=2000 * float(failed), violation_penalty=5000 * float(violation),
        high_risk_wait_penalty=80 * metrics["high_risk_mean_wait"] / hw,
        mean_wait_penalty=40 * metrics["mean_wait"] / mw,
        p95_penalty=5 * metrics["p95_wait"] / mw, queue_penalty=2 * metrics["queue_peak"],
        pressure_duration_penalty=10 * sum(metrics["resource_pressure"][name]["minutes_above_target"] +
            2 * metrics["resource_pressure"][name]["minutes_at_full_saturation"] for name in RESOURCES) / horizon,
        queue_growth_penalty=50 * max(0., metrics["queue_at_horizon_end"] - provenance_queue(evaluation) - 2),
        nonrecovery_penalty=200 * nonrecovery, recovery_penalty=20 * recovery / horizon if recovery is not None else 0.,
        reserve_penalty=30 * reserve, complexity_penalty=2 * complexity, disruption_penalty=disruption)
    score = components["robustness_reward"] - sum(v for k, v in components.items() if k != "robustness_reward")
    return float(score), components


def provenance_queue(evaluation):
    return evaluation["provenance"]["current_queue"]


def optimization_provenance(hospital, event, horizon, replications, targets, options):
    state_hash = hospital.state_hash()
    return dict(version="3.4-sustained-pressure-policy-ga", live_state_hash=state_hash,
        state_hash_schema=HASH_SCHEMA,
        live_state_context=dict(sim_time_minutes=hospital.sim_time_minutes, active_patient_count=len(hospital.active_patients),
            queue=len(hospital._waiting), base_arrival_rate=hospital.arrival_rate,
            effective_arrival_rate=hospital.effective_arrival_rate,
            active_events=[{**e.specification(), "status": e.status} for e in hospital.stress_events.values()
                if e.status in ("SCHEDULED", "ACTIVE")]),
        sim_time_minutes=hospital.sim_time_minutes, physical_capacity=asdict(hospital.capacity),
        effective_capacity=hospital.resource_summary(), current_queue=len(hospital._waiting),
        current_policy=asdict(hospital.current_policy), proposed_event=event.specification() if event else None,
        profile_sha256=hospital.profile_sha256, horizon_minutes=horizon,
        verification_replications=replications, targets=target_values(targets), ga_options=asdict(options),
        gene_bounds={k: list(v) for k, v in POLICY_BOUNDS.items()},
        seed_strategy="experiment_seed=SeedSequence([live_seed,ga_seed,3303]).generate_state(1)[0]; lookahead seeds=SeedSequence([experiment_seed,3002,index]); stable optimization seed shared across all candidates/generations/full verification; state hash is for stale protection only",
        live_seed=hospital.seed, policy_assumption="Candidate policy held throughout look-ahead; live temporary recovery is a separate human-selected application mode.")


def optimization_is_current(result, hospital, event, horizon, replications, targets, options):
    return bool(result) and result.get("provenance") == optimization_provenance(
        hospital, event, horizon, replications, targets, options) and diagnosis_is_current(result)


def optimize_live_policy(hospital, event=None, horizon_minutes=120, verification_replications=5,
                         targets=None, options=None, progress=None):
    started = time.perf_counter()
    targets, options = targets or LiveTargets(), options or LiveGAOptions(seed=hospital.seed)
    options.validate()
    provenance = optimization_provenance(hospital, event, horizon_minutes, verification_replications, targets, options)
    checkpoint = hospital.checkpoint()
    # Stable future experiment; state hashes are used only for stale protection.
    experiment_seed = int(np.random.SeedSequence([hospital.seed, options.seed, 3303]).generate_state(1)[0])
    rng = np.random.default_rng(options.seed)
    cache, evaluations, history = {}, [], []
    baseline = simulate_lookahead_from_current_state(checkpoint, event, horizon_minutes,
        options.search_replications, targets, experiment_seed, policy_override=checkpoint.current_policy)

    def evaluate(policy, generation):
        key = policy_key(policy)
        if key not in cache:
            prediction = baseline if policy == checkpoint.current_policy else simulate_lookahead_from_current_state(
                checkpoint, event, horizon_minutes, options.search_replications, targets, experiment_seed, policy_override=policy)
            assert prediction["provenance"]["physical_capacity"] == provenance["physical_capacity"]
            fitness, components = live_fitness(prediction, policy, targets, horizon_minutes, baseline)
            cache[key] = dict(policy=asdict(policy), evaluation=prediction, fitness=fitness, components=components,
                              first_generation=generation)
            evaluations.append(cache[key])
        return cache[key]["fitness"]

    anchors = [checkpoint.current_policy, OperatingPolicy(),
               OperatingPolicy(high_risk_priority_weight=5., doctor_reserve_percentage=.25,
                               nurse_reserve_percentage=.25)]
    population = anchors + [random_policy(rng) for _ in range(options.population - len(anchors))]
    generation_zero = [asdict(p) for p in population]
    best, best_fitness, stagnant = population[0], -float("inf"), 0
    early_stopped = False
    for generation in range(options.generations):
        ranked = sorted(population, key=lambda p: evaluate(p, generation), reverse=True)
        score = evaluate(ranked[0], generation)
        if score > best_fitness + 1e-6:
            best, best_fitness, stagnant = ranked[0], score, 0
        else:
            stagnant += 1
        history.append(dict(generation=generation, best_fitness=best_fitness,
            best_policy=asdict(best), candidates=[dict(policy=asdict(p), fitness=evaluate(p, generation)) for p in population]))
        if progress:
            progress(generation + 1, options.generations, best_fitness)
        if stagnant >= options.patience:
            early_stopped = True
            break
        def select():
            return max([ranked[int(rng.integers(len(ranked)))] for _ in range(3)], key=lambda p: evaluate(p, generation))
        population = [best] + [crossover_mutate(select(), select(), rng) for _ in range(options.population - 1)]
    current_verified = baseline if verification_replications == options.search_replications else simulate_lookahead_from_current_state(checkpoint, event, horizon_minutes,
        verification_replications, targets, experiment_seed, policy_override=checkpoint.current_policy)
    # Full verification of every unique tested policy guarantees selection of the
    # best verified candidate found, including useful below-threshold policies.
    winner = asdict(best)
    verified_cache = {policy_key(checkpoint.current_policy): current_verified}
    verified_scores = []
    for observation in evaluations:
        candidate = OperatingPolicy(**observation["policy"])
        key = policy_key(candidate)
        if key not in verified_cache:
            verified_cache[key] = observation["evaluation"] if verification_replications == options.search_replications else simulate_lookahead_from_current_state(checkpoint, event, horizon_minutes,
                verification_replications, targets, experiment_seed, policy_override=candidate)
        prediction = verified_cache[key]
        score, _ = live_fitness(prediction, candidate, targets, horizon_minutes, current_verified)
        verified_scores.append((score, candidate, prediction))
    verified_score, best, verified = max(verified_scores, key=lambda item: item[0])
    best_fitness = evaluate(best, 0)
    retained_current = policy_key(best) == policy_key(checkpoint.current_policy)
    from live_policy_diagnosis import metric_changes, meaningful_improvement
    meaningful = meaningful_improvement(metric_changes(current_verified, verified))
    feasible = verified["robustness"] >= targets.robustness_threshold
    outcome = "VERIFIED SUCCESS" if feasible else "PARTIAL IMPROVEMENT" if meaningful else "NO MEANINGFUL IMPROVEMENT"
    status = "OPTIMIZED POLICY HANDLES EVENT" if feasible else "BEST AVAILABLE IMPROVEMENT ? STILL " + verified["verdict"] if meaningful else "NO MEANINGFUL IMPROVEMENT WITH CURRENT PHYSICAL CAPACITY"
    result = dict(current_policy=asdict(checkpoint.current_policy), optimized_policy=asdict(best),
        search_winner=winner, search_fitness=best_fitness, verified_fitness=verified_score, outcome=outcome,
        verified_candidate_count=len(verified_cache), current_verified=current_verified,
        optimized_verified=verified, adaptive_status=status,
        feasibility_message="Verified policy meets targets." if feasible else
            "The best verified tested policy remains below the configured robustness target. Human review may apply a partial improvement, reject it, or modify/retry. Finite heuristic search does not prove infeasibility.",
        generations_completed=len(history), early_stopped=early_stopped, generation_zero=generation_zero,
        history=history, evaluations=evaluations, unique_evaluations=len(evaluations),
        verification_retained_current=retained_current, experiment_seed=experiment_seed,
        provenance=provenance, runtime_seconds=time.perf_counter() - started)
    result["run_id"] = canonical_hash(provenance)[:20]
    result["diagnosis"] = diagnose_live_result(result)
    assert hospital.state_hash() == provenance["live_state_hash"], "Optimization mutated live state."
    return result
