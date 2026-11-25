import numpy as np


def unpack(genome, cfg):
    i = 0
    n1 = cfg.input_size * cfg.hidden_size
    W1 = genome[i:i + n1].reshape(cfg.input_size, cfg.hidden_size)
    i += n1
    b1 = genome[i:i + cfg.hidden_size]
    i += cfg.hidden_size
    n2 = cfg.hidden_size * cfg.output_size
    W2 = genome[i:i + n2].reshape(cfg.hidden_size, cfg.output_size)
    i += n2
    b2 = genome[i:i + cfg.output_size]
    return W1, b1, W2, b2


def pack(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])


def forward(genome, x, cfg):
    W1, b1, W2, b2 = unpack(genome, cfg)
    z1 = x @ W1 + b1
    h = np.tanh(z1)
    z2 = h @ W2 + b2
    mu = np.tanh(z2)
    return mu, h


def random_genome(cfg, rng):
    return rng.normal(0.0, cfg.init_std, cfg.genome_size)


def act(genome, x, cfg, rng, stochastic):
    mu, h = forward(genome, x, cfg)
    if stochastic:
        a = mu + rng.normal(0.0, cfg.rl_std, mu.shape)
        a = np.clip(a, -1.0, 1.0)
    else:
        a = mu
    return a, mu, h


def reinforce(genome, trajectories, cfg):
    if not trajectories:
        return genome.copy()

    W1, b1, W2, b2 = unpack(genome, cfg)
    gW1 = np.zeros_like(W1)
    gb1 = np.zeros_like(b1)
    gW2 = np.zeros_like(W2)
    gb2 = np.zeros_like(b2)

    prepared = []
    all_returns = []
    total_steps = 0
    for traj in trajectories:
        G = 0.0
        returns = np.empty(len(traj))
        for k in range(len(traj) - 1, -1, -1):
            G = traj[k]["r"] + cfg.rl_gamma * G
            returns[k] = G
        prepared.append((traj, returns))
        all_returns.append(returns)
        total_steps += len(traj)

    if total_steps == 0:
        return genome.copy()

    flat = np.concatenate(all_returns)
    baseline = float(np.mean(flat))
    scale = float(np.std(flat)) + 1e-8

    for traj, returns in prepared:
        adv = np.clip((returns - baseline) / scale, -5.0, 5.0)[:, None]
        X = np.array([step["x"] for step in traj])
        H = np.array([step["h"] for step in traj])
        MU = np.array([step["mu"] for step in traj])
        A = np.array([step["a"] for step in traj])

        dZ2 = ((A - MU) / (cfg.rl_std ** 2)) * (1.0 - MU ** 2) * adv
        gW2 += H.T @ dZ2
        gb2 += dZ2.sum(axis=0)

        dH = dZ2 @ W2.T
        dZ1 = dH * (1.0 - H ** 2)
        gW1 += X.T @ dZ1
        gb1 += dZ1.sum(axis=0)

    grad = pack(gW1, gb1, gW2, gb2) / max(total_steps, 1)
    norm = float(np.linalg.norm(grad))
    if norm > cfg.rl_grad_clip:
        grad *= cfg.rl_grad_clip / norm
    updated = genome + cfg.rl_lr * grad
    return np.clip(updated, -cfg.genome_clip, cfg.genome_clip)
