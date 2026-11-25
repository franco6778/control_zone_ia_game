from dataclasses import dataclass
import math


@dataclass
class Config:
    # Ventana y mapa rectangular
    width: int = 1100
    height: int = 760
    arena_margin_x: float = 70.0
    arena_margin_y: float = 80.0
    robot_radius: float = 18.0

    # 5 zonas de control pequenas
    num_zones: int = 5
    zone_radius: float = 42.0
    capture_time: float = 10.0
    unlock_interval: float = 12.0
    zones_to_win: int = 3

    # Fisica con empuje fuerte para disputar zonas pequenas
    dt: float = 1.0 / 30.0
    max_speed: float = 260.0
    accel: float = 1050.0
    turn_rate: float = 4.5
    friction: float = 2.2
    restitution: float = 0.25
    wall_restitution: float = 0.3
    knock_base: float = 175.0
    knock_gain: float = 1.45
    kick_friction: float = 2.1

    # Red neuronal
    input_size: int = 18
    hidden_size: int = 24
    output_size: int = 2
    init_std: float = 0.5

    # Duracion de partida (120s a 30 Hz = 3600 pasos)
    max_time: float = 120.0

    # Evolucion y seleccion
    population_size: int = 20
    elite: int = 3
    games_per_genome: int = 16
    tournament_k: int = 3

    # Recompensas
    hold_reward_per_sec: float = 3.5
    contest_reward_per_sec: float = 0.6
    approach_weight: float = 10.0
    contact_bonus: float = 3.0
    zone_capture_reward: float = 60.0
    zone_lost_penalty: float = 35.0
    win_capture_reward: float = 120.0
    win_speed_bonus: float = 40.0
    loss_capture_penalty: float = 120.0
    win_timeout_reward: float = 30.0
    loss_timeout_penalty: float = 30.0
    draw_penalty: float = 25.0

    # Oponentes
    bot_prob: float = 0.25
    hof_opponent_prob: float = 0.20
    best_opponent_prob: float = 0.40
    hof_max_size: int = 8
    mutation_rate: float = 0.08
    mutation_scale: float = 0.25
    genome_clip: float = 6.0

    # RL (REINFORCE)
    rl_episodes: int = 12
    rl_lr: float = 0.05
    rl_gamma: float = 0.99
    rl_std: float = 0.30
    rl_grad_clip: float = 5.0

    # Sparring y control
    sparring_size: int = 8
    report_size: int = 8
    hof_games: int = 16
    hof_threshold: float = 0.6

    generations: int = 50
    seed: int = 0
    render: bool = True
    demo_speed: float = 2.0
    workers: int = 0

    @property
    def max_steps(self):
        return int(round(self.max_time / self.dt))

    @property
    def arena_bounds(self):
        return (
            self.arena_margin_x,
            self.width - self.arena_margin_x,
            self.arena_margin_y,
            self.height - self.arena_margin_y,
        )

    @property
    def arena_width(self):
        x0, x1, _, _ = self.arena_bounds
        return x1 - x0

    @property
    def arena_height(self):
        _, _, y0, y1 = self.arena_bounds
        return y1 - y0

    @property
    def arena_diag(self):
        return math.hypot(self.arena_width, self.arena_height)

    @property
    def center(self):
        x0, x1, y0, y1 = self.arena_bounds
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def genome_size(self):
        return (
            self.input_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.output_size
            + self.output_size
        )
