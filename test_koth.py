import unittest

import numpy as np

from arena import RectArena, create_zones
from brain import random_genome, reinforce
from config import Config
from evolution import benchmark_score, evaluate, head_to_head
from match import simulate
from policies import Brawler, ZoneRusher, make_bots
from robot import Robot


class IdlePolicy:
    def act(self, x, rng, stochastic):
        a = np.zeros(2)
        return a, a, np.zeros(2)


class ControlZonesTests(unittest.TestCase):
    def test_five_zones_unlock_over_time(self):
        cfg = Config(num_zones=5, unlock_interval=12.0)
        arena = RectArena(cfg)
        zones = create_zones(arena, cfg, np.random.default_rng(0), randomize=False)
        self.assertEqual(len(zones), 5)
        self.assertTrue(zones[0].unlocked)
        self.assertTrue(all(not z.unlocked for z in zones[1:]))
        self.assertEqual([z.unlock_time for z in zones], [0.0, 12.0, 24.0, 36.0, 48.0])

    def test_capture_takes_ten_seconds_and_wins_at_three_zones(self):
        cfg = Config(capture_time=10.0, unlock_interval=12.0, zones_to_win=3, max_time=65.0)
        rng = np.random.default_rng(1)
        res = simulate(ZoneRusher(), IdlePolicy(), cfg, rng, randomize=False)
        self.assertEqual(res.winner, 0)
        self.assertEqual(res.end_reason, "capture")
        self.assertEqual(res.scores[0], 3)
        # 3 zonas * 10s cada una => al menos 30 segundos
        self.assertGreaterEqual(res.elapsed_time, 30.0)

    def test_contested_zone_pauses_capture(self):
        cfg = Config(capture_time=10.0, unlock_interval=100.0, max_time=15.0)
        arena = RectArena(cfg)
        zones = create_zones(arena, cfg, np.random.default_rng(2), randomize=False)
        # Dos ZoneRushers identicos en posiciones simetricas chocan y disputan la zona central
        res = simulate(ZoneRusher(), ZoneRusher(), cfg, np.random.default_rng(2), randomize=False)
        self.assertGreater(res.contacts, 0)
        # En 15s disputados ninguno deberia llegar facil a 3 zonas
        self.assertLess(res.scores[0], 3)
        self.assertLess(res.scores[1], 3)
        self.assertEqual(len(zones), 5)

    def test_walls_keep_robots_inside_rect_arena(self):
        cfg = Config()
        arena = RectArena(cfg)
        r = Robot([arena.x_min - 50.0, arena.y_max + 50.0], 0.0)
        r.vel = np.array([-100.0, 100.0])
        arena.clamp_robot(r)
        self.assertGreaterEqual(r.pos[0], arena.x_min + cfg.robot_radius)
        self.assertLessEqual(r.pos[1], arena.y_max - cfg.robot_radius)
        self.assertGreater(r.vel[0], 0.0)
        self.assertLess(r.vel[1], 0.0)

    def test_rl_trajectory_includes_terminal_reward(self):
        cfg = Config(capture_time=5.0, unlock_interval=6.0, zones_to_win=3, max_time=40.0)
        rng = np.random.default_rng(3)
        res = simulate(
            ZoneRusher(),
            IdlePolicy(),
            cfg,
            rng,
            stochastic=False,
            collect_for=0,
            randomize=False,
        )
        self.assertEqual(res.end_reason, "capture")
        self.assertEqual(res.winner, 0)
        self.assertAlmostEqual(sum(s["r"] for s in res.trajectory), res.fitness_a, places=5)
        self.assertGreater(res.trajectory[-1]["r"], cfg.win_capture_reward * 0.8)

    def test_evolution_and_reinforce_pipeline(self):
        cfg = Config(
            population_size=4,
            games_per_genome=4,
            max_time=20.0,
            capture_time=5.0,
            unlock_interval=6.0,
            workers=1,
        )
        rng = np.random.default_rng(4)
        pop = [random_genome(cfg, rng) for _ in range(cfg.population_size)]
        bots = make_bots(4, seed=10)

        fitness, stats = evaluate(pop, pop[0], bots, cfg, rng, hof=[pop[1]])
        self.assertEqual(fitness.shape, (cfg.population_size,))
        self.assertAlmostEqual(
            stats["capture_end_rate"] + stats["timeout_rate"] + stats["draw_rate"], 1.0, places=6
        )

        _, win, bstats = benchmark_score(pop[0], bots, cfg, rng)
        self.assertGreaterEqual(win, 0.0)
        self.assertLessEqual(win, 1.0)
        self.assertLessEqual(bstats["capture_win_rate"], win + 1e-9)

        h2h, _ = head_to_head(pop[0], pop[1], cfg, rng, games=4)
        self.assertGreaterEqual(h2h, 0.0)
        self.assertLessEqual(h2h, 1.0)

        res = simulate(pop[0], Brawler(), cfg, rng, stochastic=True, collect_for=0, randomize=True)
        updated = reinforce(pop[0], [res.trajectory], cfg)
        self.assertEqual(updated.shape, pop[0].shape)
        self.assertTrue(np.all(np.isfinite(updated)))


if __name__ == "__main__":
    unittest.main()
