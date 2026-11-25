import numpy as np


def tournament(population, fitness, cfg, rng):
    idx = rng.integers(0, len(population), cfg.tournament_k)
    best = int(idx[0])
    for i in idx[1:]:
        if fitness[i] > fitness[best]:
            best = int(i)
    return population[best]


def crossover(parent_a, parent_b, rng):
    mask = rng.random(parent_a.shape) < 0.5
    return np.where(mask, parent_a, parent_b)


def mutate(genome, cfg, rng):
    child = genome.copy()
    mask = rng.random(child.shape) < cfg.mutation_rate
    count = int(mask.sum())
    if count:
        child[mask] += rng.normal(0.0, cfg.mutation_scale, count)
    if rng.random() < 0.05:
        count = max(1, child.size // 20)
        idx = rng.integers(0, child.size, count)
        child[idx] = rng.normal(0.0, 1.0, count)
    return np.clip(child, -cfg.genome_clip, cfg.genome_clip)


def evolve(population, fitness, cfg, rng):
    order = np.argsort(-fitness)
    new_population = [population[int(i)].copy() for i in order[: cfg.elite]]

    while len(new_population) < cfg.population_size:
        parent_a = tournament(population, fitness, cfg, rng)
        parent_b = tournament(population, fitness, cfg, rng)
        child = crossover(parent_a, parent_b, rng)
        child = mutate(child, cfg, rng)
        new_population.append(child)

    return new_population
