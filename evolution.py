import os
from multiprocessing import get_context

import numpy as np

from brain import reinforce
from match import simulate


def _worker_count(cfg):
    if cfg.workers and cfg.workers > 0:
        return cfg.workers
    return max(1, os.cpu_count() or 1)


def _pool_context():
    try:
        return get_context("fork")
    except ValueError:
        return get_context()


def _eval_worker(task):
    index, individual, opponent, cfg, seed, individual_is_a = task
    rng = np.random.default_rng(seed)
    if individual_is_a:
        result = simulate(individual, opponent, cfg, rng, randomize=True)
        fit = result.fitness_a
        own_caps = result.scores[0]
    else:
        result = simulate(opponent, individual, cfg, rng, randomize=True)
        fit = result.fitness_b
        own_caps = result.scores[1]
    total_caps = result.scores[0] + result.scores[1]
    return (
        index,
        fit,
        result.end_reason,
        result.elapsed_time,
        result.contacts,
        total_caps,
        own_caps,
    )


def _pick_opponent(population, best_genome, bots, cfg, rng, hof=None):
    p = rng.random()
    acc = 0.0
    if bots and cfg.bot_prob > 0.0:
        acc += cfg.bot_prob
        if p < acc:
            return bots[int(rng.integers(0, len(bots)))]
    if hof and cfg.hof_opponent_prob > 0.0:
        acc += cfg.hof_opponent_prob
        if p < acc:
            return hof[int(rng.integers(0, len(hof)))]
    acc += cfg.best_opponent_prob
    if p < acc:
        return best_genome
    return population[int(rng.integers(0, len(population)))]


def _summarize_stats(
    total_games,
    capture_end_count,
    timeout_count,
    draw_count,
    total_time,
    total_contacts,
    total_caps,
    own_caps=0,
    wins=0,
    capture_wins=0,
):
    denom = max(total_games, 1)
    return {
        "games": total_games,
        "capture_end_rate": capture_end_count / denom,
        "timeout_rate": timeout_count / denom,
        "draw_rate": draw_count / denom,
        "avg_time": total_time / denom,
        "avg_contacts": total_contacts / denom,
        "avg_total_caps": total_caps / denom,
        "avg_own_caps": own_caps / denom,
        "win_rate": wins / denom,
        "capture_win_rate": capture_wins / denom,
    }


def evaluate(population, best_genome, bots, cfg, rng, hof=None):
    n = len(population)
    tasks = []

    for i in range(n):
        for _ in range(cfg.games_per_genome):
            opponent = _pick_opponent(population, best_genome, bots, cfg, rng, hof=hof)
            individual_is_a = rng.random() < 0.5
            seed = int(rng.integers(0, 2**31 - 1))
            tasks.append((i, population[i], opponent, cfg, seed, individual_is_a))

    total = np.zeros(n)
    cap_end = 0
    timeout_end = 0
    draw_end = 0
    total_time = 0.0
    total_contacts = 0
    total_caps = 0
    own_caps_sum = 0
    workers = min(_worker_count(cfg), len(tasks))

    if workers > 1:
        with _pool_context().Pool(workers) as pool:
            results = pool.imap_unordered(_eval_worker, tasks, chunksize=4)
            for index, fit, end_reason, elapsed, contacts, tot_c, own_c in results:
                total[index] += fit
                if end_reason == "capture":
                    cap_end += 1
                elif end_reason == "timeout":
                    timeout_end += 1
                else:
                    draw_end += 1
                total_time += elapsed
                total_contacts += contacts
                total_caps += tot_c
                own_caps_sum += own_c
    else:
        for task in tasks:
            index, fit, end_reason, elapsed, contacts, tot_c, own_c = _eval_worker(task)
            total[index] += fit
            if end_reason == "capture":
                cap_end += 1
            elif end_reason == "timeout":
                timeout_end += 1
            else:
                draw_end += 1
            total_time += elapsed
            total_contacts += contacts
            total_caps += tot_c
            own_caps_sum += own_c

    stats = _summarize_stats(
        len(tasks),
        cap_end,
        timeout_end,
        draw_end,
        total_time,
        total_contacts,
        total_caps,
        own_caps=own_caps_sum,
    )
    return total / cfg.games_per_genome, stats


def fine_tune(best_genome, population, bots, cfg, rng, hof=None):
    genome = best_genome.copy()
    trajectories = []

    for _ in range(cfg.rl_episodes):
        opponent = _pick_opponent(population, best_genome, bots, cfg, rng, hof=hof)
        if rng.random() < 0.5:
            result = simulate(
                genome, opponent, cfg, rng, stochastic=True, collect_for=0, randomize=True
            )
        else:
            result = simulate(
                opponent, genome, cfg, rng, stochastic=True, collect_for=1, randomize=True
            )
        if result.trajectory:
            trajectories.append(result.trajectory)

    return reinforce(genome, trajectories, cfg)


def benchmark_score(genome, opponents, cfg, rng):
    total = 0.0
    wins = 0
    capture_wins = 0
    cap_end = 0
    timeout_end = 0
    draw_end = 0
    total_time = 0.0
    total_contacts = 0
    total_caps = 0
    own_caps = 0
    count = 0

    for opponent in opponents:
        res_a = simulate(genome, opponent, cfg, rng)
        total += res_a.fitness_a
        won_a = res_a.winner == 0
        wins += won_a
        capture_wins += won_a and res_a.end_reason == "capture"
        cap_end += res_a.end_reason == "capture"
        timeout_end += res_a.end_reason == "timeout"
        draw_end += res_a.end_reason == "draw"
        total_time += res_a.elapsed_time
        total_contacts += res_a.contacts
        total_caps += res_a.scores[0] + res_a.scores[1]
        own_caps += res_a.scores[0]

        res_b = simulate(opponent, genome, cfg, rng)
        total += res_b.fitness_b
        won_b = res_b.winner == 1
        wins += won_b
        capture_wins += won_b and res_b.end_reason == "capture"
        cap_end += res_b.end_reason == "capture"
        timeout_end += res_b.end_reason == "timeout"
        draw_end += res_b.end_reason == "draw"
        total_time += res_b.elapsed_time
        total_contacts += res_b.contacts
        total_caps += res_b.scores[0] + res_b.scores[1]
        own_caps += res_b.scores[1]

        count += 2

    stats = _summarize_stats(
        count,
        cap_end,
        timeout_end,
        draw_end,
        total_time,
        total_contacts,
        total_caps,
        own_caps=own_caps,
        wins=wins,
        capture_wins=capture_wins,
    )
    return total / max(count, 1), stats["win_rate"], stats


def head_to_head(genome_x, genome_y, cfg, rng, games):
    wins = 0
    capture_wins = 0
    decisive = 0
    cap_end = 0
    timeout_end = 0
    draw_end = 0
    total_time = 0.0
    total_contacts = 0
    total_caps = 0
    own_caps = 0

    for g in range(games):
        if g % 2 == 0:
            res = simulate(genome_x, genome_y, cfg, rng, randomize=True)
            won = res.winner == 0
            own_caps += res.scores[0]
        else:
            res = simulate(genome_y, genome_x, cfg, rng, randomize=True)
            won = res.winner == 1
            own_caps += res.scores[1]

        if res.winner is not None:
            decisive += 1
            wins += won
            capture_wins += won and res.end_reason == "capture"

        cap_end += res.end_reason == "capture"
        timeout_end += res.end_reason == "timeout"
        draw_end += res.end_reason == "draw"
        total_time += res.elapsed_time
        total_contacts += res.contacts
        total_caps += res.scores[0] + res.scores[1]

    stats = _summarize_stats(
        games,
        cap_end,
        timeout_end,
        draw_end,
        total_time,
        total_contacts,
        total_caps,
        own_caps=own_caps,
        wins=wins,
        capture_wins=capture_wins,
    )
    if decisive < max(1, games // 2):
        return 0.5, stats
    return wins / decisive, stats
