import numpy as np


class ControlZone:
    def __init__(self, zone_id, pos, radius, unlock_time):
        self.id = zone_id
        self.pos = np.array(pos, dtype=float)
        self.radius = float(radius)
        self.unlock_time = float(unlock_time)
        self.unlocked = unlock_time <= 0.0
        self.owner = None  # None, 0 (azul), 1 (rojo)
        self.progress = [0.0, 0.0]  # segundos acumulados por cada robot
        self.contested = False

    @property
    def active(self):
        return self.unlocked and self.owner is None

    def contains(self, pos):
        return float(np.linalg.norm(np.asarray(pos, dtype=float) - self.pos)) <= self.radius


class RectArena:
    def __init__(self, cfg):
        self.cfg = cfg
        self.x_min, self.x_max, self.y_min, self.y_max = cfg.arena_bounds
        self.center = np.array(cfg.center, dtype=float)
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.diag = cfg.arena_diag

    def clamp_robot(self, robot):
        r = self.cfg.robot_radius
        rest = self.cfg.wall_restitution

        if robot.pos[0] < self.x_min + r:
            robot.pos[0] = self.x_min + r
            if robot.vel[0] < 0.0:
                robot.vel[0] = -robot.vel[0] * rest
            if robot.kick[0] < 0.0:
                robot.kick[0] = -robot.kick[0] * rest
        elif robot.pos[0] > self.x_max - r:
            robot.pos[0] = self.x_max - r
            if robot.vel[0] > 0.0:
                robot.vel[0] = -robot.vel[0] * rest
            if robot.kick[0] > 0.0:
                robot.kick[0] = -robot.kick[0] * rest

        if robot.pos[1] < self.y_min + r:
            robot.pos[1] = self.y_min + r
            if robot.vel[1] < 0.0:
                robot.vel[1] = -robot.vel[1] * rest
            if robot.kick[1] < 0.0:
                robot.kick[1] = -robot.kick[1] * rest
        elif robot.pos[1] > self.y_max - r:
            robot.pos[1] = self.y_max - r
            if robot.vel[1] > 0.0:
                robot.vel[1] = -robot.vel[1] * rest
            if robot.kick[1] > 0.0:
                robot.kick[1] = -robot.kick[1] * rest


def create_zones(arena, cfg, rng, randomize=False):
    cx, cy = arena.center
    dx = arena.width * 0.28
    dy = arena.height * 0.28

    # 5 zonas: centro + 4 cuadrantes
    base_positions = [
        np.array([cx, cy], dtype=float),
        np.array([cx - dx, cy - dy], dtype=float),
        np.array([cx + dx, cy + dy], dtype=float),
        np.array([cx + dx, cy - dy], dtype=float),
        np.array([cx - dx, cy + dy], dtype=float),
    ]

    if randomize:
        # Zona 0 empieza cerca del centro (equidistante), las otras 4 en orden aleatorio con leve jitter
        rest = base_positions[1:]
        order = rng.permutation(len(rest))
        positions = [base_positions[0]]
        jitter_x = arena.width * 0.06
        jitter_y = arena.height * 0.06
        for idx in order:
            p = rest[int(idx)].copy()
            p[0] += float(rng.uniform(-jitter_x, jitter_x))
            p[1] += float(rng.uniform(-jitter_y, jitter_y))
            margin = cfg.zone_radius + cfg.robot_radius + 16.0
            p[0] = float(np.clip(p[0], arena.x_min + margin, arena.x_max - margin))
            p[1] = float(np.clip(p[1], arena.y_min + margin, arena.y_max - margin))
            positions.append(p)
    else:
        positions = base_positions

    zones = []
    for i in range(cfg.num_zones):
        unlock_t = i * cfg.unlock_interval
        zones.append(ControlZone(i, positions[i], cfg.zone_radius, unlock_t))
    return zones
