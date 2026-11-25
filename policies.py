import numpy as np


class ZoneRusher:
    """Corre a la zona activa y empuja al rival si esta disputandola."""

    def __init__(self, speed_base=0.65, steer_gain=1.4, ram_distance=0.22):
        self.speed_base = speed_base
        self.steer_gain = steer_gain
        self.ram_distance = ram_distance

    def act(self, x, rng, stochastic):
        opp_fwd, opp_right, opp_dist, opp_in_zone = x[0], x[1], x[2], x[3]
        zone_fwd, zone_right, zone_dist, self_in_zone = x[4], x[5], x[6], x[7]
        vel_fwd = x[10]

        # Si el rival esta en la zona o muy cerca, embestirlo para sacarlo
        if opp_in_zone > 0.5 or (self_in_zone > 0.5 and opp_dist < self.ram_distance):
            target_right = opp_right
            forward = float(np.clip(self.speed_base + 0.6 * opp_fwd, -0.3, 1.0))
        elif self_in_zone > 0.5:
            # Frenar dentro de la zona pequena y centrarse
            target_right = zone_right
            forward = float(np.clip(zone_dist * 6.0 * zone_fwd - 0.9 * vel_fwd, -1.0, 1.0))
        else:
            # Ir rapido a la zona activa y desacelerar al acercarse
            target_right = zone_right
            if zone_dist < 0.14:
                desired_speed = min(1.0, zone_dist * 8.0)
                forward = float(np.clip((desired_speed * zone_fwd - 0.6 * vel_fwd) * 1.5, -0.8, 1.0))
            else:
                forward = float(np.clip(self.speed_base + 0.5 * zone_fwd, -0.2, 1.0))

        steer = float(np.clip(self.steer_gain * target_right, -1.0, 1.0))
        left_out = float(np.clip(forward - steer, -1.0, 1.0))
        right_out = float(np.clip(forward + steer, -1.0, 1.0))
        action = np.array([left_out, right_out])
        return action, action, np.zeros(2)


class Brawler:
    """Busca primero chocar al rival cerca de la zona y luego tomar el punto."""

    def __init__(self, speed_base=0.75, steer_gain=1.6, chase_dist=0.35):
        self.speed_base = speed_base
        self.steer_gain = steer_gain
        self.chase_dist = chase_dist

    def act(self, x, rng, stochastic):
        opp_fwd, opp_right, opp_dist, opp_in_zone = x[0], x[1], x[2], x[3]
        zone_fwd, zone_right, zone_dist, self_in_zone = x[4], x[5], x[6], x[7]
        vel_fwd = x[10]

        if opp_in_zone > 0.5 or opp_dist < self.chase_dist:
            target_right = opp_right
            forward = float(np.clip(self.speed_base + 0.5 * opp_fwd, 0.0, 1.0))
        elif self_in_zone > 0.5:
            target_right = zone_right
            forward = float(np.clip(zone_dist * 5.0 * zone_fwd - 0.9 * vel_fwd, -1.0, 1.0))
        else:
            target_right = zone_right
            if zone_dist < 0.14:
                desired_speed = min(1.0, zone_dist * 8.0)
                forward = float(np.clip((desired_speed * zone_fwd - 0.6 * vel_fwd) * 1.5, -0.8, 1.0))
            else:
                forward = float(np.clip(self.speed_base + 0.4 * zone_fwd, 0.0, 1.0))

        steer = float(np.clip(self.steer_gain * target_right, -1.0, 1.0))
        left_out = float(np.clip(forward - steer, -1.0, 1.0))
        right_out = float(np.clip(forward + steer, -1.0, 1.0))
        action = np.array([left_out, right_out])
        return action, action, np.zeros(2)


def make_bots(n=8, seed=777):
    rng = np.random.default_rng(seed)
    bots = []
    for i in range(n):
        if i % 2 == 0:
            bots.append(
                ZoneRusher(
                    speed_base=float(rng.uniform(0.55, 0.85)),
                    steer_gain=float(rng.uniform(1.1, 1.8)),
                    ram_distance=float(rng.uniform(0.16, 0.28)),
                )
            )
        else:
            bots.append(
                Brawler(
                    speed_base=float(rng.uniform(0.60, 0.90)),
                    steer_gain=float(rng.uniform(1.2, 1.9)),
                    chase_dist=float(rng.uniform(0.25, 0.42)),
                )
            )
    return bots


class HumanController:
    def __init__(self):
        import pygame

        self.pygame = pygame

    def act(self, x, rng, stochastic):
        keys = self.pygame.key.get_pressed()
        forward = 0.0
        if keys[self.pygame.K_UP] or keys[self.pygame.K_w]:
            forward += 1.0
        if keys[self.pygame.K_DOWN] or keys[self.pygame.K_s]:
            forward -= 1.0

        turn = 0.0
        if keys[self.pygame.K_RIGHT] or keys[self.pygame.K_d]:
            turn += 1.0
        if keys[self.pygame.K_LEFT] or keys[self.pygame.K_a]:
            turn -= 1.0

        left = float(np.clip(forward - turn, -1.0, 1.0))
        right = float(np.clip(forward + turn, -1.0, 1.0))
        action = np.array([left, right])
        return action, action, np.zeros(2)
