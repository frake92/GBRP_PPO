from __future__ import annotations

import argparse
from collections import deque
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
    steps_per_frame: int

    fullscreen: bool

    width: int
    height: int

    pixels_per_meter: float
    y_half_range_m: float

    deterministic: bool
    episodes: int 

    record_path: Optional[Path]
    record_fps: int

    camera: str
    trail_len: int


class BoatRacePygameViewer:
    def __init__(self, cfg: ViewerConfig):
        self.cfg = cfg

        pygame.init()
        pygame.display.set_caption("BoatRace - PPO Viewer")

        if cfg.fullscreen:
            info = pygame.display.Info()
            width = int(getattr(info, "current_w", 0) or 0)
            height = int(getattr(info, "current_h", 0) or 0)
            if width <= 0 or height <= 0:
                width, height = cfg.width, cfg.height

            flags = pygame.FULLSCREEN
            try:
                self.screen = pygame.display.set_mode((width, height), flags | pygame.SCALED)
            except pygame.error:
                self.screen = pygame.display.set_mode((width, height), flags)
        else:
            self.screen = pygame.display.set_mode((cfg.width, cfg.height))

        self.view_width, self.view_height = self.screen.get_size()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        try:
            import register_envs
        except Exception:
            pass

        self.env = gym.make(cfg.env_id)
        self.model = PPO.load(str(cfg.model_path))

        self.obs, _ = self.env.reset(seed=cfg.seed)
        self.episode_idx = 0
        self.episode_return = 0.0

        self.last_action = np.zeros((2,), dtype=np.float32)
        self.last_reward = 0.0
        self.last_nonzero_reward = 0.0
        self.last_nonzero_at_step = -1
        self.last_info = {}

        self.trail = deque(maxlen=max(0, int(cfg.trail_len)))
        self._reward_flash_frames = 0

        self._video_writer = None

        if cfg.record_path is not None:
            try:
                import imageio
            except Exception as e:
                raise RuntimeError(
                    "Recording requires imageio + imageio-ffmpeg. Install via: pip install -r requirements.txt"
                ) from e

            cfg.record_path.parent.mkdir(parents=True, exist_ok=True)
            self._video_writer = imageio.get_writer(
                str(cfg.record_path),
                fps=int(cfg.record_fps),
                codec="libx264",
                quality=8,
            )

    def close(self) -> None:
        try:
            if self._video_writer is not None:
                self._video_writer.close()
            self.env.close()
        finally:
            pygame.quit()

    def _draw_text(self, lines: list[str]) -> None:
        x = 16
        y = 16
        for line in lines:
            surf = self.font.render(line, True, (235, 235, 235))
            self.screen.blit(surf, (x, y))
            y += 22

    def _world_to_screen(self, x_m: float, y_m: float) -> tuple[int, int]:
        cfg = self.cfg
        env = self.env.unwrapped

        x_min = -5.0
        x_max = float(env.finish_line_x) + 5.0

        ppm, cam_left, cam_bottom = self._camera_params(x_min=x_min, x_max=x_max)

        sx = int((x_m - cam_left) * ppm)
        sy = int(self.view_height - (y_m - cam_bottom) * ppm)
        return sx, sy

    def _camera_params(self, *, x_min: float, x_max: float) -> tuple[float, float, float]:
        cfg = self.cfg
        env = self.env.unwrapped

        track_width_m = max(1e-6, x_max - x_min)
        ppm_fit = self.view_width / track_width_m

        def cam_bottom_for(ppm: float, center_y: float) -> float:
            view_height_m = self.view_height / max(1e-6, ppm)
            return float(center_y) - 0.5 * float(view_height_m)

        if cfg.camera == "global":
            ppm = float(min(ppm_fit, cfg.pixels_per_meter))
            view_width_m = self.view_width / max(1e-6, ppm)
            track_center = 0.5 * (x_min + x_max)
            cam_left = track_center - 0.5 * view_width_m
            cam_bottom = cam_bottom_for(ppm, center_y=0.0)
            return ppm, cam_left, cam_bottom

        ppm = float(cfg.pixels_per_meter)
        view_width_m = self.view_width / max(1e-6, ppm)
        cam_left = float(env.x) - view_width_m * 0.5
        cam_left = max(x_min, min(cam_left, x_max - view_width_m))
        cam_bottom = cam_bottom_for(ppm, center_y=float(env.y))
        return ppm, cam_left, cam_bottom

    def _draw_boat(self, x_m: float, y_m: float, heading_rad: float) -> None:
        cfg = self.cfg
        px, py = self._world_to_screen(x_m, y_m)

        length = 0.9
        width = 0.5

        pts_local = np.array(
            [
                [length, 0.0],
                [-length * 0.6, -width],
                [-length * 0.6, width],
            ],
            dtype=np.float32,
        )

        c = float(np.cos(heading_rad))
        s = float(np.sin(heading_rad))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts_world = (pts_local @ rot.T) + np.array([x_m, y_m], dtype=np.float32)

        pts_screen = [self._world_to_screen(float(x), float(y)) for x, y in pts_world]
        pygame.draw.polygon(self.screen, (80, 170, 240), pts_screen)
        pygame.draw.polygon(self.screen, (20, 40, 70), pts_screen, 2)

    def _draw_scene(self) -> None:
        cfg = self.cfg
        env = self.env.unwrapped

        self.screen.fill((16, 18, 22))

        if self._reward_flash_frames > 0:
            alpha = min(90, 20 + self._reward_flash_frames * 3)
            overlay = pygame.Surface((self.view_width, self.view_height), pygame.SRCALPHA)
            overlay.fill((60, 220, 140, alpha))
            self.screen.blit(overlay, (0, 0))
            self._reward_flash_frames -= 1

        x0, y0 = self._world_to_screen(-5.0, 0.0)
        x1, y1 = self._world_to_screen(float(env.finish_line_x) + 5.0, 0.0)
        pygame.draw.line(self.screen, (90, 90, 90), (x0, y0), (x1, y1), 2)

        x_min = -5.0
        x_max = float(env.finish_line_x) + 5.0
        ppm, _cam_left, cam_bottom = self._camera_params(x_min=x_min, x_max=x_max)
        view_height_m = self.view_height / max(1e-6, ppm)
        cam_top = cam_bottom + view_height_m
        for i, (cx, cy) in enumerate(env.checkpoints):
            sx, sy = self._world_to_screen(float(cx), float(cy))
            r_px = max(3, int(float(env.checkpoint_radius) * ppm))

            hit_count = int(env.checkpoint_hit_count.get(i, 0))
            if hit_count > 0:
                color = (60, 220, 120)  # hit
            else:
                color = (60, 140, 60)  # not hit

            pygame.draw.circle(self.screen, color, (sx, sy), r_px, 0)
            pygame.draw.circle(self.screen, (10, 20, 10), (sx, sy), r_px, 2)

            if hit_count > 1:
                pygame.draw.circle(self.screen, (230, 60, 60), (sx, sy), r_px + 4, 2)

            label = f"{i}" if hit_count == 0 else f"{i}x{hit_count}"
            surf = self.font.render(label, True, (240, 240, 240))
            self.screen.blit(surf, (sx + r_px + 6, sy - 10))

        fx = float(env.finish_line_x)
        sx0, sy0 = self._world_to_screen(fx, float(cam_bottom))
        sx1, sy1 = self._world_to_screen(fx, float(cam_top))
        pygame.draw.line(self.screen, (250, 210, 80), (sx0, sy0), (sx1, sy1), 4)

        if self.trail:
            pts = [self._world_to_screen(float(x), float(y)) for x, y in self.trail]
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, (200, 200, 255), False, pts, 2)

        self._draw_boat(float(env.x), float(env.y), float(env.heading))

        speed = float(np.sqrt(env.vx**2 + env.vy**2))
        throttle = float(self.last_action[0])
        steering = float(self.last_action[1])

        farming_total = sum(max(0, int(c) - 1) for c in env.checkpoint_hit_count.values())

        last_nz = "never" if self.last_nonzero_at_step < 0 else f"step {self.last_nonzero_at_step} (+{self.last_nonzero_reward:.3f})"

        lines = [
            f"env: {cfg.env_id}",
            f"episode={self.episode_idx}  step={env.step_count}/{env.max_steps}  t={env.step_count*env.dt:5.1f}s",
            f"pos=({env.x:6.2f},{env.y:6.2f})  speed={speed:5.2f}  heading={env.heading:+.2f}",
            f"progress={env.progress}/{env.num_checkpoints}  farming={farming_total}  bug={bool(env.bug.enable_bug)}",
            f"action throttle={throttle:+.2f} steering={steering:+.2f}  reward={self.last_reward:+.6f}",
            f"episode_return={self.episode_return:.3f}  last_nonzero={last_nz}",
            f"camera={cfg.camera}  steps/frame={cfg.steps_per_frame}  trail={cfg.trail_len}",
            "ESC to quit",
        ]
        self._draw_text(lines)

        pygame.display.flip()

        if self._video_writer is not None:
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))
            self._video_writer.append_data(frame)

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def run(self) -> None:
        while True:
            if not self._handle_events():
                return

            for _ in range(max(1, self.cfg.steps_per_frame)):
                action, _ = self.model.predict(self.obs, deterministic=self.cfg.deterministic)
                self.last_action = np.asarray(action, dtype=np.float32).reshape(2)

                self.obs, reward, terminated, truncated, info = self.env.step(action)
                self.last_reward = float(reward)
                self.episode_return += self.last_reward
                self.last_info = dict(info) if isinstance(info, dict) else {}

                env = self.env.unwrapped
                if self.trail.maxlen and self.trail.maxlen > 0:
                    self.trail.append((float(env.x), float(env.y)))

                if abs(self.last_reward) > 1e-9:
                    self.last_nonzero_reward = self.last_reward
                    self.last_nonzero_at_step = int(self.env.unwrapped.step_count)
                    self._reward_flash_frames = 18

                if terminated or truncated:
                    self.episode_idx += 1
                    if self.cfg.episodes > 0 and self.episode_idx >= self.cfg.episodes:
                        return
                    self.obs, _ = self.env.reset(seed=self.cfg.seed + self.episode_idx)
                    self.episode_return = 0.0
                    self.last_reward = 0.0
                    self.last_nonzero_reward = 0.0
                    self.last_nonzero_at_step = -1
                    self.last_info = {}
                    self.trail.clear()
                    self._reward_flash_frames = 0
                    break

            self._draw_scene()
            self.clock.tick(max(1, self.cfg.fps))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pygame viewer for a trained PPO model on BoatRace-v0")
    p.add_argument("--env", default="BoatRace-v0")
    p.add_argument("--model", required=True, help="Path to SB3 model .zip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--steps-per-frame", type=int, default=1, help="Increase to speed up; keep 1 for real-time")
    p.add_argument(
        "--windowed",
        action="store_true",
        help="Start in a normal window (fullscreen is the default).",
    )
    p.add_argument("--ppm", type=float, default=8.0, help="Pixels per meter (camera zoom)")
    p.add_argument("--y-half-range", type=float, default=15.0, help="Visible +/- y range in meters")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=650)
    p.add_argument("--episodes", type=int, default=0, help="0 = run forever")
    p.add_argument("--deterministic", action="store_true")

    p.add_argument(
        "--camera",
        choices=["follow", "global"],
        default="follow",
        help="Camera mode: follow the boat (default) or show the whole track.",
    )
    p.add_argument(
        "--trail",
        type=int,
        default=800,
        help="How many recent positions to draw as a trail (0 disables).",
    )

    p.add_argument(
        "--record",
        default="",
        help="If set, write an mp4 to this path (e.g. runs/BoatRacePPOHackable/playback.mp4)",
    )
    p.add_argument(
        "--record-fps",
        type=int,
        default=30,
        help="FPS of the saved mp4 (default: 30)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ViewerConfig(
        env_id=args.env,
        model_path=Path(args.model),
        seed=args.seed,
        fps=args.fps,
        steps_per_frame=args.steps_per_frame,
        fullscreen=not bool(args.windowed),
        width=args.width,
        height=args.height,
        pixels_per_meter=args.ppm,
        y_half_range_m=args.y_half_range,
        deterministic=bool(args.deterministic),
        episodes=args.episodes,
        record_path=Path(args.record) if str(args.record).strip() else None,
        record_fps=int(args.record_fps),
        camera=str(args.camera),
        trail_len=int(args.trail),
    )

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")

    viewer = BoatRacePygameViewer(cfg)
    try:
        viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
