import argparse
import csv
import os

import numpy as np

from brain import random_genome
from config import Config
from evolution import benchmark_score, evaluate, fine_tune, head_to_head
from genetic import evolve
from match import simulate
from policies import make_bots


LOG_FIELDS = [
    "generation",
    "pop_best",
    "pop_mean",
    "eval_capture_end_pct",
    "eval_timeout_pct",
    "eval_draw_pct",
    "eval_avg_total_caps",
    "eval_avg_time",
    "eval_avg_contacts",
    "hof_win_pct",
    "hof_capture_end_pct",
    "hof_promoted",
    "rl_accepted",
    "rl_win_pct",
    "ctrl_win_pct",
    "ctrl_capture_win_pct",
    "ctrl_avg_own_caps",
    "ctrl_avg_time",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control Zones AI: 5 zonas progresivas (10s) con GA + NN + RL"
    )
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--bot-prob", type=float, default=None, help="probabilidad de sparring contra bots")
    parser.add_argument("--zone-radius", type=float, default=None, help="radio de cada zona de control")
    parser.add_argument("--capture-time", type=float, default=None, help="segundos necesarios para capturar una zona")
    parser.add_argument("--unlock-interval", type=float, default=None, help="segundos entre desbloqueos de zona")
    parser.add_argument("--max-time", type=float, default=None, help="duracion maxima de la partida en segundos")
    parser.add_argument("--log", default="training_log.csv", help="archivo CSV de metricas (vacio para desactivar)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignora el mejor genoma guardado")
    parser.add_argument("--watch", action="store_true", help="solo reproduce el mejor genoma")
    parser.add_argument("--play", action="store_true", help="juega tu contra el mejor genoma")
    parser.add_argument("--eval", action="store_true", help="evalua el mejor genoma guardado")
    parser.add_argument("--save", default="best_genome.npz")
    return parser.parse_args()


def make_config(args):
    cfg = Config()
    if args.generations is not None:
        cfg.generations = args.generations
    if args.population is not None:
        cfg.population_size = args.population
    if args.seed is not None:
        cfg.seed = args.seed
    if args.workers is not None:
        cfg.workers = args.workers
    if args.bot_prob is not None:
        cfg.bot_prob = args.bot_prob
    if args.zone_radius is not None:
        cfg.zone_radius = args.zone_radius
    if args.capture_time is not None:
        cfg.capture_time = args.capture_time
    if args.unlock_interval is not None:
        cfg.unlock_interval = args.unlock_interval
    if args.max_time is not None:
        cfg.max_time = args.max_time
    if args.headless:
        cfg.render = False
    return cfg


def load_best(path, cfg, report_set, rng):
    if not os.path.exists(path):
        return None, None
    try:
        genome = np.load(path)["genome"].copy()
    except Exception as exc:
        print(f"No pude leer {path}: {exc}")
        return None, None
    if genome.shape != (cfg.genome_size,):
        print(f"Forma incompatible en {path}: {genome.shape} != ({cfg.genome_size},)")
        return None, None
    fit, win, stats = benchmark_score(genome, report_set, cfg, rng)
    return genome, (win, fit, stats)


def run_eval(cfg, rng, path, report_set):
    loaded, info = load_best(path, cfg, report_set, rng)
    if loaded is None:
        print(f"No existe un genoma valido en {path}; entrena primero")
        return

    win, fit, stats = info
    _, self_stats = head_to_head(loaded, loaded, cfg, rng, cfg.hof_games)
    print(f"Evaluacion de {path}:")
    print(
        f"  Control bots ({stats['games']} partidas): "
        f"victorias {win * 100:.1f}% (por 3 zonas {stats['capture_win_rate'] * 100:.1f}%) | "
        f"zonas propias {stats['avg_own_caps']:.2f} | totales {stats['avg_total_caps']:.2f} | "
        f"tiempo {stats['avg_time']:.1f}s | contactos {stats['avg_contacts']:.1f} | fitness {fit:.2f}"
    )
    print(
        f"  Espejo ({self_stats['games']} partidas): "
        f"fin por captura {self_stats['capture_end_rate'] * 100:.1f}% | "
        f"timeout {self_stats['timeout_rate'] * 100:.1f}% | "
        f"zonas totales {self_stats['avg_total_caps']:.2f} | "
        f"tiempo {self_stats['avg_time']:.1f}s | contactos {self_stats['avg_contacts']:.1f}"
    )


def run_watch(cfg, rng, path):
    from render import Renderer

    if not os.path.exists(path):
        print(f"No existe {path}; entrena primero")
        return

    genome = np.load(path)["genome"]
    renderer = Renderer(cfg)
    match_num = 0
    try:
        while not renderer.quit_requested:
            match_num += 1
            renderer.hud_lines = [
                f"Demo {match_num} - Control Zones (5 zonas, {cfg.capture_time:.0f}s)",
                "Mejor genoma (azul) vs Mejor genoma (rojo)",
            ]
            simulate(genome, genome, cfg, rng, randomize=True, hook=renderer.draw)
    finally:
        renderer.close()


def run_play(cfg, rng, path):
    from policies import HumanController
    from render import Renderer

    if not os.path.exists(path):
        print(f"No existe {path}; entrena primero")
        return

    genome = np.load(path)["genome"]
    cfg.demo_speed = 1.0
    renderer = Renderer(cfg)
    renderer.mode = "play"
    human = HumanController()

    last_state = {}

    def hook(state):
        last_state.clear()
        last_state.update(state)
        return renderer.draw(state)

    match_num = 0
    try:
        while not renderer.quit_requested:
            match_num += 1
            renderer.hud_lines = [
                f"Partida {match_num}: TU (azul) vs IA (rojo)",
                f"Controla {cfg.zones_to_win} zonas ({cfg.capture_time:.0f}s c/u) para ganar",
            ]
            result = simulate(human, genome, cfg, rng, randomize=False, hook=hook)
            if renderer.quit_requested:
                break
            reason = "3 zonas" if result.end_reason == "capture" else "tiempo"
            if result.winner == 0:
                text = f"GANASTE! {result.scores[0]}-{result.scores[1]} ({reason})"
            elif result.winner == 1:
                text = f"Gano la IA {result.scores[0]}-{result.scores[1]} ({reason})"
            else:
                text = f"Empate {result.scores[0]}-{result.scores[1]}"
            renderer.flash(last_state, text, 1.8)
    finally:
        renderer.close()


def main():
    args = parse_args()
    cfg = make_config(args)
    rng = np.random.default_rng(cfg.seed)

    sparring_bots = make_bots(cfg.sparring_size, seed=cfg.seed + 777)
    report_set = make_bots(cfg.report_size, seed=cfg.seed + 1777)

    if args.watch:
        run_watch(cfg, rng, args.save)
        return

    if args.play:
        run_play(cfg, rng, args.save)
        return

    if args.eval:
        run_eval(cfg, rng, args.save, report_set)
        return

    population = [random_genome(cfg, rng) for _ in range(cfg.population_size)]
    best_genome = population[0].copy()
    hof = []

    if not args.fresh:
        loaded, info = load_best(args.save, cfg, report_set, rng)
        if loaded is not None:
            best_genome = loaded
            population[0] = loaded.copy()
            hof.append(loaded.copy())
            win, _, stats = info
            print(
                f"Mejor genoma previo cargado "
                f"({win * 100:.0f}% vic, {stats['avg_own_caps']:.2f} zonas/partida vs control)"
            )

    log_file = None
    log_writer = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        log_writer.writeheader()

    renderer = None
    if cfg.render:
        from render import Renderer

        renderer = Renderer(cfg)

    try:
        for generation in range(1, cfg.generations + 1):
            fitness, eval_stats = evaluate(
                population, best_genome, sparring_bots, cfg, rng, hof=hof
            )
            champion_idx = int(np.argmax(fitness))
            champion = population[champion_idx].copy()
            champion_fitness = float(fitness[champion_idx])

            _, bench_win, bench_stats = benchmark_score(champion, report_set, cfg, rng)

            tuned = fine_tune(champion, population, sparring_bots, cfg, rng, hof=hof)
            tuned_vs_champ, _ = head_to_head(tuned, champion, cfg, rng, cfg.hof_games)
            accepted = tuned_vs_champ > 0.5
            candidate = tuned if accepted else champion

            win_rate, hof_stats = head_to_head(candidate, best_genome, cfg, rng, cfg.hof_games)
            promoted = win_rate > cfg.hof_threshold
            if promoted:
                best_genome = candidate.copy()
                hof.append(best_genome.copy())
                if len(hof) > cfg.hof_max_size:
                    hof.pop(0)
                np.savez(args.save, genome=best_genome)
                hof_note = f"nuevo {win_rate * 100:.0f}%"
            else:
                hof_note = f"conserva {win_rate * 100:.0f}%"

            status = "ok" if accepted else "no"
            print(
                f"Gen {generation:03d} | pop {champion_fitness:7.2f} (med {fitness.mean():7.2f}) | "
                f"cap3 {eval_stats['capture_end_rate'] * 100:4.1f}% "
                f"zonas {eval_stats['avg_total_caps']:4.2f} "
                f"t {eval_stats['avg_time']:4.1f}s chq {eval_stats['avg_contacts']:4.1f} | "
                f"HOF {hof_note} | RL {status} {tuned_vs_champ * 100:3.0f}% | "
                f"ctrl {bench_win * 100:5.1f}% (zonas {bench_stats['avg_own_caps']:.2f})"
            )

            if log_writer is not None:
                log_writer.writerow(
                    {
                        "generation": generation,
                        "pop_best": round(champion_fitness, 4),
                        "pop_mean": round(float(fitness.mean()), 4),
                        "eval_capture_end_pct": round(eval_stats["capture_end_rate"] * 100.0, 2),
                        "eval_timeout_pct": round(eval_stats["timeout_rate"] * 100.0, 2),
                        "eval_draw_pct": round(eval_stats["draw_rate"] * 100.0, 2),
                        "eval_avg_total_caps": round(eval_stats["avg_total_caps"], 2),
                        "eval_avg_time": round(eval_stats["avg_time"], 2),
                        "eval_avg_contacts": round(eval_stats["avg_contacts"], 2),
                        "hof_win_pct": round(win_rate * 100.0, 2),
                        "hof_capture_end_pct": round(hof_stats["capture_end_rate"] * 100.0, 2),
                        "hof_promoted": int(promoted),
                        "rl_accepted": int(accepted),
                        "rl_win_pct": round(tuned_vs_champ * 100.0, 2),
                        "ctrl_win_pct": round(bench_win * 100.0, 2),
                        "ctrl_capture_win_pct": round(bench_stats["capture_win_rate"] * 100.0, 2),
                        "ctrl_avg_own_caps": round(bench_stats["avg_own_caps"], 2),
                        "ctrl_avg_time": round(bench_stats["avg_time"], 2),
                    }
                )
                log_file.flush()

            if renderer is not None:
                renderer.hud_lines = [
                    f"Generacion {generation}/{cfg.generations}",
                    f"Fin por 3 zonas: {eval_stats['capture_end_rate'] * 100:.0f}% | Zonas/part: {eval_stats['avg_total_caps']:.2f}",
                    f"Control bots: {bench_win * 100:.0f}% (zonas {bench_stats['avg_own_caps']:.2f})",
                ]
                simulate(best_genome, best_genome, cfg, rng, randomize=True, hook=renderer.draw)
                if renderer.quit_requested:
                    break

            new_population = evolve(population, fitness, cfg, rng)
            if accepted:
                new_population[-1] = tuned
            population = new_population
    finally:
        if log_file is not None:
            log_file.close()
        if renderer is not None:
            renderer.close()

    np.savez(args.save, genome=best_genome)
    print(f"Mejor genoma guardado en {args.save}")
    if args.log:
        print(f"Metricas guardadas en {args.log}")


if __name__ == "__main__":
    main()
