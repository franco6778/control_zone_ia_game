import numpy as np

from arena import RectArena, create_zones
from brain import act
from robot import Robot, pick_target_zone, sensors


class MatchResult:
    def __init__(
        self,
        fitness_a,
        fitness_b,
        winner,
        steps,
        trajectory,
        scores=(0, 0),
        end_reason="draw",
        contacts=0,
        elapsed_time=0.0,
    ):
        self.fitness_a = fitness_a
        self.fitness_b = fitness_b
        self.winner = winner
        self.steps = steps
        self.trajectory = trajectory
        self.scores = tuple(scores)
        self.end_reason = end_reason  # "capture", "timeout", "draw", "aborted"
        self.contacts = contacts
        self.elapsed_time = elapsed_time


def policy_action(controller, x, cfg, rng, stochastic):
    if isinstance(controller, np.ndarray):
        return act(controller, x, cfg, rng, stochastic)
    return controller.act(x, rng, stochastic)


def simulate(
    genome_a,
    genome_b,
    cfg,
    rng,
    stochastic=False,
    collect_for=None,
    hook=None,
    max_steps=None,
    randomize=False,
):
    arena = RectArena(cfg)
    zones = create_zones(arena, cfg, rng, randomize=randomize)
    cx, cy = arena.center

    if randomize:
        spawn_x = float(rng.uniform(0.30, 0.40) * arena.width)
        spawn_y = float(rng.uniform(-0.18, 0.18) * arena.height)
        robot_a = Robot(
            [cx - spawn_x, cy + spawn_y],
            float(rng.uniform(-0.4, 0.4)),
        )
        robot_b = Robot(
            [cx + spawn_x, cy - spawn_y],
            np.pi + float(rng.uniform(-0.4, 0.4)),
        )
    else:
        spawn_x = arena.width * 0.35
        robot_a = Robot([cx - spawn_x, cy], 0.0)
        robot_b = Robot([cx + spawn_x, cy], np.pi)

    fitness = [0.0, 0.0]
    scores = [0, 0]
    trajectory = [] if collect_for is not None else None

    if max_steps is None:
        max_steps = cfg.max_steps

    winner = None
    end_reason = "draw"
    steps = 0
    contacts = 0
    touching = False

    target_a = pick_target_zone(robot_a, zones)
    target_b = pick_target_zone(robot_b, zones)
    prev_dist_za = float(np.linalg.norm(target_a.pos - robot_a.pos))
    prev_dist_zb = float(np.linalg.norm(target_b.pos - robot_b.pos))
    prev_target_id_a = target_a.id
    prev_target_id_b = target_b.id

    for step in range(max_steps):
        steps = step + 1
        elapsed = steps * cfg.dt

        # Desbloquear zonas por tiempo
        for z in zones:
            if not z.unlocked and elapsed >= z.unlock_time:
                z.unlocked = True

        # Si no queda ninguna zona activa pero aun hay bloqueadas, desbloquear la siguiente
        if not any(z.active for z in zones):
            locked = [z for z in zones if not z.unlocked and z.owner is None]
            if locked:
                nxt = min(locked, key=lambda z: z.unlock_time)
                nxt.unlocked = True
                nxt.unlock_time = elapsed

        x_a = sensors(robot_a, robot_b, zones, 0, scores, arena, cfg)
        x_b = sensors(robot_b, robot_a, zones, 1, scores, arena, cfg)

        a_a, mu_a, h_a = policy_action(
            genome_a, x_a, cfg, rng, stochastic and collect_for == 0
        )
        a_b, mu_b, h_b = policy_action(
            genome_b, x_b, cfg, rng, stochastic and collect_for == 1
        )

        robot_a.drive(a_a[0], a_a[1], cfg)
        robot_b.drive(a_b[0], a_b[1], cfg)

        hit_a_impact = 0.0
        hit_b_impact = 0.0

        delta = robot_b.pos - robot_a.pos
        dist = float(np.linalg.norm(delta))
        min_dist = 2.0 * cfg.robot_radius
        if 1e-9 < dist < min_dist:
            normal = delta / dist
            overlap = min_dist - dist
            if not touching:
                contacts += 1
                v_a = robot_a.vel + robot_a.kick
                v_b = robot_b.vel + robot_b.kick
                hit_by_b = max(0.0, -float(np.dot(v_b, normal)))
                hit_by_a = max(0.0, float(np.dot(v_a, normal)))
                hit_a_impact = hit_by_a / cfg.max_speed
                hit_b_impact = hit_by_b / cfg.max_speed
                robot_a.kick -= normal * (cfg.knock_base + cfg.knock_gain * hit_by_b)
                robot_b.kick += normal * (cfg.knock_base + cfg.knock_gain * hit_by_a)
            touching = True
            robot_a.pos -= normal * (overlap * 0.5)
            robot_b.pos += normal * (overlap * 0.5)

            rel_vel = float(np.dot(robot_b.vel - robot_a.vel, normal))
            if rel_vel < 0.0:
                impulse = -(1.0 + cfg.restitution) * rel_vel * 0.5
                robot_a.vel -= normal * impulse
                robot_b.vel += normal * impulse
        else:
            touching = False

        arena.clamp_robot(robot_a)
        arena.clamp_robot(robot_b)

        reward_a = 0.0
        reward_b = 0.0

        # Actualizar progreso de las zonas activas
        a_in_any_active = False
        b_in_any_active = False
        for z in zones:
            if not z.active:
                z.contested = False
                continue
            a_in = z.contains(robot_a.pos)
            b_in = z.contains(robot_b.pos)
            a_in_any_active = a_in_any_active or a_in
            b_in_any_active = b_in_any_active or b_in

            if a_in and b_in:
                # Ambos dentro: zona disputada, nadie suma tiempo
                z.contested = True
                reward_a += cfg.contest_reward_per_sec * cfg.dt
                reward_b += cfg.contest_reward_per_sec * cfg.dt
            elif a_in:
                z.contested = False
                z.progress[0] += cfg.dt
                reward_a += cfg.hold_reward_per_sec * cfg.dt
                if z.progress[0] >= cfg.capture_time:
                    z.progress[0] = cfg.capture_time
                    z.owner = 0
                    scores[0] += 1
                    reward_a += cfg.zone_capture_reward
                    reward_b -= cfg.zone_lost_penalty
            elif b_in:
                z.contested = False
                z.progress[1] += cfg.dt
                reward_b += cfg.hold_reward_per_sec * cfg.dt
                if z.progress[1] >= cfg.capture_time:
                    z.progress[1] = cfg.capture_time
                    z.owner = 1
                    scores[1] += 1
                    reward_b += cfg.zone_capture_reward
                    reward_a -= cfg.zone_lost_penalty
            else:
                z.contested = False

        # Premio por empujar al rival (doble si el rival esta en una zona activa)
        if hit_a_impact > 0.0:
            mult = 2.0 if b_in_any_active else 1.0
            reward_a += cfg.contact_bonus * hit_a_impact * mult
        if hit_b_impact > 0.0:
            mult = 2.0 if a_in_any_active else 1.0
            reward_b += cfg.contact_bonus * hit_b_impact * mult

        # Shaping potencial por acercarse a la zona objetivo
        cur_target_a = pick_target_zone(robot_a, zones)
        cur_target_b = pick_target_zone(robot_b, zones)
        dist_za = float(np.linalg.norm(cur_target_a.pos - robot_a.pos))
        dist_zb = float(np.linalg.norm(cur_target_b.pos - robot_b.pos))

        if cur_target_a.id == prev_target_id_a and not cur_target_a.contains(robot_a.pos):
            reward_a += cfg.approach_weight * (prev_dist_za - dist_za) / arena.diag
        if cur_target_b.id == prev_target_id_b and not cur_target_b.contains(robot_b.pos):
            reward_b += cfg.approach_weight * (prev_dist_zb - dist_zb) / arena.diag

        prev_dist_za = dist_za
        prev_dist_zb = dist_zb
        prev_target_id_a = cur_target_a.id
        prev_target_id_b = cur_target_b.id

        fitness[0] += reward_a
        fitness[1] += reward_b

        if trajectory is not None:
            if collect_for == 0:
                trajectory.append(
                    {"x": x_a, "a": a_a, "mu": mu_a, "h": h_a, "r": reward_a}
                )
            else:
                trajectory.append(
                    {"x": x_b, "a": a_b, "mu": mu_b, "h": h_b, "r": reward_b}
                )

        done = False
        if scores[0] >= cfg.zones_to_win:
            winner = 0
            end_reason = "capture"
            done = True
        elif scores[1] >= cfg.zones_to_win:
            winner = 1
            end_reason = "capture"
            done = True
        elif all(z.owner is not None for z in zones):
            end_reason = "capture"
            done = True
            if scores[0] > scores[1]:
                winner = 0
            elif scores[1] > scores[0]:
                winner = 1

        if hook is not None:
            state = {
                "arena": arena,
                "zones": zones,
                "robots": [robot_a, robot_b],
                "scores": list(scores),
                "fitness": list(fitness),
                "step": steps,
                "max_steps": max_steps,
                "elapsed": elapsed,
                "max_time": cfg.max_time,
                "done": done,
                "winner": winner,
                "end_reason": end_reason,
            }
            if hook(state) is False:
                return MatchResult(
                    fitness[0],
                    fitness[1],
                    winner,
                    steps,
                    trajectory,
                    scores=scores,
                    end_reason="aborted",
                    contacts=contacts,
                    elapsed_time=elapsed,
                )

        if done:
            break

    elapsed = steps * cfg.dt

    if end_reason != "capture":
        if scores[0] > scores[1]:
            winner = 0
            end_reason = "timeout"
        elif scores[1] > scores[0]:
            winner = 1
            end_reason = "timeout"
        else:
            prog_a = sum(z.progress[0] for z in zones if z.owner is None)
            prog_b = sum(z.progress[1] for z in zones if z.owner is None)
            if abs(prog_a - prog_b) > 0.5:
                winner = 0 if prog_a > prog_b else 1
                end_reason = "timeout"
            else:
                end_reason = "draw"

    terminal = [0.0, 0.0]
    if end_reason == "capture" and winner is not None:
        remaining = max_steps - steps
        speed_bonus = cfg.win_speed_bonus * remaining / max_steps
        if winner == 0:
            terminal[0] = cfg.win_capture_reward + speed_bonus
            terminal[1] = -cfg.loss_capture_penalty
        else:
            terminal[1] = cfg.win_capture_reward + speed_bonus
            terminal[0] = -cfg.loss_capture_penalty
    elif end_reason == "timeout" and winner is not None:
        if winner == 0:
            terminal[0] = cfg.win_timeout_reward
            terminal[1] = -cfg.loss_timeout_penalty
        else:
            terminal[1] = cfg.win_timeout_reward
            terminal[0] = -cfg.loss_timeout_penalty
    else:
        terminal[0] = -cfg.draw_penalty
        terminal[1] = -cfg.draw_penalty

    fitness[0] += terminal[0]
    fitness[1] += terminal[1]
    if trajectory:
        trajectory[-1]["r"] += terminal[collect_for]

    return MatchResult(
        fitness[0],
        fitness[1],
        winner,
        steps,
        trajectory,
        scores=scores,
        end_reason=end_reason,
        contacts=contacts,
        elapsed_time=elapsed,
    )
