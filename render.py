import math
import numpy as np
import pygame


class Renderer:
    def __init__(self, cfg):
        pygame.init()
        self.cfg = cfg
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("Control Zones AI - 5 Zonas (10s)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.small = pygame.font.SysFont(None, 18)
        self.big = pygame.font.SysFont(None, 32)
        self.speed = cfg.demo_speed
        self.paused = False
        self.quit_requested = False
        self.mode = "watch"
        self.hud_lines = []

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif self.mode != "play" and event.key == pygame.K_UP:
                    self.speed = min(self.speed * 1.5, 16.0)
                elif self.mode != "play" and event.key == pygame.K_DOWN:
                    self.speed = max(self.speed / 1.5, 0.25)

    def draw(self, state):
        self._handle_events()
        if self.quit_requested:
            return False

        while self.paused and not self.quit_requested:
            self.clock.tick(30)
            self._handle_events()

        if self.quit_requested:
            return False

        self._render(state)
        self.clock.tick(max(1, int((1.0 / self.cfg.dt) * self.speed)))
        return True

    def flash(self, state, text, seconds):
        end = pygame.time.get_ticks() + int(seconds * 1000)
        overlay = pygame.Surface((self.cfg.width, self.cfg.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))

        while pygame.time.get_ticks() < end and not self.quit_requested:
            self._handle_events()
            if self.quit_requested:
                return
            if state:
                self._render(state)
            else:
                self.screen.fill((18, 20, 28))
            self.screen.blit(overlay, (0, 0))
            message = self.big.render(text, True, (255, 255, 255))
            rect = message.get_rect(center=(self.cfg.width // 2, self.cfg.height // 2))
            self.screen.blit(message, rect)
            pygame.display.flip()
            self.clock.tick(30)

    def _render(self, state):
        arena = state["arena"]
        self.screen.fill((16, 18, 26))

        # Mapa rectangular
        rect = pygame.Rect(
            int(arena.x_min),
            int(arena.y_min),
            int(arena.width),
            int(arena.height),
        )
        pygame.draw.rect(self.screen, (28, 32, 44), rect, border_radius=8)

        # Grilla sutil
        step = 60
        for gx in range(int(arena.x_min) + step, int(arena.x_max), step):
            pygame.draw.line(self.screen, (36, 40, 54), (gx, int(arena.y_min)), (gx, int(arena.y_max)), 1)
        for gy in range(int(arena.y_min) + step, int(arena.y_max), step):
            pygame.draw.line(self.screen, (36, 40, 54), (int(arena.x_min), gy), (int(arena.x_max), gy), 1)

        pygame.draw.rect(self.screen, (100, 110, 145), rect, 3, border_radius=8)

        # Zonas de control
        elapsed = state["elapsed"]
        for zone in state["zones"]:
            self._draw_zone(zone, elapsed)

        # Robots
        colors = [(80, 165, 255), (255, 95, 95)]
        for robot, color in zip(state["robots"], colors):
            self._draw_robot(robot, color)

        # Marcador superior centrado
        scores = state["scores"]
        goal = self.cfg.zones_to_win
        score_text = self.big.render(
            f"AZUL {scores[0]}/{goal}  -  {scores[1]}/{goal} ROJO", True, (240, 240, 250)
        )
        score_rect = score_text.get_rect(center=(self.cfg.width // 2, 26))
        self.screen.blit(score_text, score_rect)

        y = 10
        for line in self.hud_lines:
            text = self.small.render(line, True, (215, 215, 230))
            self.screen.blit(text, (14, y))
            y += 20

        if self.mode == "play":
            controls = "[flechas/WASD mover | ESPACIO pausa | ESC salir]"
        else:
            controls = "[ESPACIO pausa | flechas velocidad | ESC salir]"

        info = self.small.render(
            f"tiempo {elapsed:4.1f}s/{state['max_time']:.0f}s   velocidad x{self.speed:.2f}   {controls}",
            True,
            (170, 175, 195),
        )
        self.screen.blit(info, (14, self.cfg.height - 26))
        pygame.display.flip()

    def _draw_zone(self, zone, elapsed):
        pos = (int(zone.pos[0]), int(zone.pos[1]))
        r = int(zone.radius)
        cap_t = self.cfg.capture_time

        if zone.owner == 0:
            fill = (35, 75, 130)
            border = (90, 175, 255)
            label = f"Z{zone.id + 1} AZUL"
        elif zone.owner == 1:
            fill = (130, 40, 45)
            border = (255, 100, 100)
            label = f"Z{zone.id + 1} ROJO"
        elif not zone.unlocked:
            fill = (24, 26, 36)
            border = (65, 70, 90)
            rem = max(0.0, zone.unlock_time - elapsed)
            label = f"Z{zone.id + 1} ({rem:.0f}s)"
        else:
            fill = (55, 48, 28) if not zone.contested else (80, 62, 32)
            border = (245, 200, 70) if not zone.contested else (255, 255, 255)
            best_p = max(zone.progress[0], zone.progress[1])
            label = "DISPUTA" if zone.contested else f"{best_p:.1f}s"

        pygame.draw.circle(self.screen, fill, pos, r)
        pygame.draw.circle(self.screen, border, pos, r, 3 if zone.active else 2)

        # Arcos de progreso para cada robot en zonas activas
        if zone.active:
            arc_rect = pygame.Rect(pos[0] - r - 5, pos[1] - r - 5, (r + 5) * 2, (r + 5) * 2)
            p0 = min(zone.progress[0] / cap_t, 1.0)
            p1 = min(zone.progress[1] / cap_t, 1.0)
            if p0 > 0.01:
                pygame.draw.arc(
                    self.screen,
                    (80, 175, 255),
                    arc_rect,
                    math.pi / 2.0 - 2.0 * math.pi * p0,
                    math.pi / 2.0,
                    4,
                )
            if p1 > 0.01:
                inner_rect = pygame.Rect(pos[0] - r - 10, pos[1] - r - 10, (r + 10) * 2, (r + 10) * 2)
                pygame.draw.arc(
                    self.screen,
                    (255, 95, 95),
                    inner_rect,
                    math.pi / 2.0 - 2.0 * math.pi * p1,
                    math.pi / 2.0,
                    4,
                )

        txt = self.small.render(label, True, (235, 235, 245))
        txt_rect = txt.get_rect(center=pos)
        self.screen.blit(txt, txt_rect)

    def _draw_robot(self, robot, color):
        pos = (int(robot.pos[0]), int(robot.pos[1]))
        r = int(self.cfg.robot_radius)
        pygame.draw.circle(self.screen, color, pos, r)
        pygame.draw.circle(self.screen, (240, 240, 250), pos, r, 2)

        fwd = robot.forward_vec()
        tip = robot.pos + fwd * r
        pygame.draw.line(self.screen, (255, 255, 255), pos, (int(tip[0]), int(tip[1])), 3)

    def close(self):
        pygame.quit()
