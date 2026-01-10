from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO


@dataclass(frozen=True)
class ViewerConfig:
    env_id: str
    model_path: Path
    seed: int
    fps: int
    pixels_per_meter: float
    window_width: int
    window_height: int
    episodes: int
    deterministic: bool


class PygameBuggyJumpViewer:
    def __init__(self, cfg: ViewerConfig):
        self.cfg = cfg

        pygame.init()
        pygame.display.set_caption("BuggyJump - PPO Viewer")
        self.screen = pygame.display.set_mode((cfg.window_width, cfg.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        # Ensure custom env registration exists (BuggyJump-v0).
        try:
            import register_envs  # noqa: F401
        except Exception:
            pass

        self.env = gym.make(cfg.env_id)
        self.model = PPO.load(str(cfg.model_path))

        self.last_action: float = 0.0
        self.last_reward: float = 0.0

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            pygame.quit()

    def _world_to_screen_y(self, y_m: float) -> int:
        ground_y = self.cfg.window_height - 80
        return int(ground_y - y_m * self.cfg.pixels_per_meter)

    def _draw_text(self, lines: list[str]) -> None:
        x = 16
        y = 16
        for line in lines:
            surf = self.font.render(line, True, (230, 230, 230))
            self.screen.blit(surf, (x, y))
            y += 22

    def _draw_scene(self) -> None:
        cfg = self.cfg
        env = self.env.unwrapped

        self.screen.fill((18, 18, 24))

        ground_y = cfg.window_height - 80
        pygame.draw.line(self.screen, (160, 160, 160), (0, ground_y), (cfg.window_width, ground_y), 3)

        # Draw agent as a circle.
        x = cfg.window_width // 2
        y = self._world_to_screen_y(float(env.y))
        radius = 14
        color = (80, 200, 120) if bool(env.on_ground) else (80, 140, 220)
        pygame.draw.circle(self.screen, color, (x, y), radius)

        # Draw max height marker.
        max_y = self._world_to_screen_y(float(env.max_y))
        pygame.draw.line(self.screen, (255, 180, 60), (x - 40, max_y), (x + 40, max_y), 2)

        # HUD
        t_sec = float(env.step_count) * float(env.dt)
        lines = [
            f"env: {cfg.env_id}",
            f"t={t_sec:5.2f}s  step={env.step_count}/{env.max_steps}",
            f"y={env.y:7.3f}m  v={env.v:7.3f}m/s  max_y={env.max_y:7.3f}m",
            f"action={self.last_action:+.3f}  reward={self.last_reward:+.4f}",
        ]
        self._draw_text(lines)

        pygame.display.flip()

    def run(self) -> None:
        obs, _ = self.env.reset(seed=self.cfg.seed)
        ep = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return

            action, _state = self.model.predict(obs, deterministic=self.cfg.deterministic)
            # SB3 returns np.ndarray for continuous actions
            self.last_action = float(np.asarray(action).reshape(-1)[0])

            obs, reward, terminated, truncated, _info = self.env.step(action)
            self.last_reward = float(reward)

            self._draw_scene()
            self.clock.tick(self.cfg.fps)

            if terminated or truncated:
                ep += 1
                if self.cfg.episodes > 0 and ep >= self.cfg.episodes:
                    return
                obs, _ = self.env.reset(seed=self.cfg.seed + ep)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a trained SB3 PPO policy in BuggyJump using pygame.")
    p.add_argument("--env", default="BuggyJump-v0")
    p.add_argument(
        "--model",
        required=True,
        help="Path to a saved SB3 PPO model (e.g., runs/ppo1/final_model.zip)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--ppm", type=float, default=120.0, help="Pixels per meter")
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--episodes", type=int, default=0, help="0 = run forever")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ViewerConfig(
        env_id=args.env,
        model_path=Path(args.model),
        seed=args.seed,
        fps=args.fps,
        pixels_per_meter=args.ppm,
        window_width=args.width,
        window_height=args.height,
        episodes=args.episodes,
        deterministic=bool(args.deterministic),
    )

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")

    viewer = PygameBuggyJumpViewer(cfg)
    try:
        viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
