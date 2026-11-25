import numpy as np


class Robot:
    def __init__(self, pos, heading):
        self.pos = np.array(pos, dtype=float)
        self.heading = float(heading)
        self.vel = np.zeros(2)
        self.kick = np.zeros(2)

    def forward_vec(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    def right_vec(self):
        return np.array([-np.sin(self.heading), np.cos(self.heading)])

    def drive(self, left, right, cfg):
        forward = 0.5 * (left + right)
        turn = 0.5 * (right - left)

        self.heading += turn * cfg.turn_rate * cfg.dt

        direction = self.forward_vec()
        self.vel += direction * forward * cfg.accel * cfg.dt
        self.vel *= max(0.0, 1.0 - cfg.friction * cfg.dt)

        speed = float(np.linalg.norm(self.vel))
        if speed > cfg.max_speed:
            self.vel *= cfg.max_speed / speed

        self.kick *= max(0.0, 1.0 - cfg.kick_friction * cfg.dt)
        self.pos += (self.vel + self.kick) * cfg.dt


def rect_wall_distance(pos, direction, arena):
    dx = float(direction[0])
    dy = float(direction[1])

    if dx > 1e-9:
        tx = (arena.x_max - pos[0]) / dx
    elif dx < -1e-9:
        tx = (arena.x_min - pos[0]) / dx
    else:
        tx = float("inf")

    if dy > 1e-9:
        ty = (arena.y_max - pos[1]) / dy
    elif dy < -1e-9:
        ty = (arena.y_min - pos[1]) / dy
    else:
        ty = float("inf")

    return max(0.0, min(tx, ty))


def pick_target_zone(robot, zones):
    active = [z for z in zones if z.active]
    if active:
        return min(active, key=lambda z: float(np.linalg.norm(z.pos - robot.pos)))
    locked = [z for z in zones if not z.unlocked and z.owner is None]
    if locked:
        return min(locked, key=lambda z: z.unlock_time)
    return zones[0]


def sensors(robot, opponent, zones, robot_idx, scores, arena, cfg):
    opp_idx = 1 - robot_idx
    fwd = robot.forward_vec()
    right = robot.right_vec()

    # 1. Oponente relativo (4)
    rel_opp = opponent.pos - robot.pos
    dist_opp = float(np.linalg.norm(rel_opp))
    dir_opp = rel_opp / dist_opp if dist_opp > 1e-6 else np.zeros(2)

    target = pick_target_zone(robot, zones)
    opp_in_target = 1.0 if (target.active and target.contains(opponent.pos)) else 0.0

    # 2. Zona objetivo (6)
    rel_zone = target.pos - robot.pos
    dist_zone = float(np.linalg.norm(rel_zone))
    dir_zone = rel_zone / dist_zone if dist_zone > 1e-6 else np.zeros(2)
    self_in_target = 1.0 if (target.active and target.contains(robot.pos)) else 0.0
    self_prog = target.progress[robot_idx] / cfg.capture_time
    opp_prog = target.progress[opp_idx] / cfg.capture_time

    # 3. Velocidad propia (2)
    total_vel = robot.vel + robot.kick
    vel_fwd = float(np.dot(total_vel, fwd)) / cfg.max_speed
    vel_right = float(np.dot(total_vel, right)) / cfg.max_speed

    # 4. 4 rayos a las paredes rectangulares (frente, derecha, atras, izquierda) (4)
    max_dim = max(arena.width, arena.height)
    wall_rays = []
    for k in range(4):
        angle = robot.heading + k * (np.pi / 2.0)
        d = np.array([np.cos(angle), np.sin(angle)])
        t = rect_wall_distance(robot.pos, d, arena)
        clearance = max(t - cfg.robot_radius, 0.0)
        wall_rays.append(min(clearance / max_dim, 1.0))

    # 5. Marcador y zonas activas (2)
    score_diff = (scores[robot_idx] - scores[opp_idx]) / float(cfg.num_zones)
    active_ratio = sum(1 for z in zones if z.active) / float(cfg.num_zones)

    values = [
        float(np.dot(dir_opp, fwd)),
        float(np.dot(dir_opp, right)),
        min(dist_opp / arena.diag, 1.0),
        opp_in_target,
        float(np.dot(dir_zone, fwd)),
        float(np.dot(dir_zone, right)),
        min(dist_zone / arena.diag, 1.0),
        self_in_target,
        min(self_prog, 1.0),
        min(opp_prog, 1.0),
        vel_fwd,
        vel_right,
        *wall_rays,
        score_diff,
        active_ratio,
    ]
    return np.array(values, dtype=float)
