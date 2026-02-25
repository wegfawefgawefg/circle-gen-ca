"""From-scratch circle generator based on signal choreography from the paper.

This rewrite intentionally does not reuse archived implementations.

Implemented pieces:
- Square(s) timing check: verifies first-arrival i^2 on a 1D row.
- 2D signal engine with per-signal speeds.
- Circle generation flow with signals j/k/j_bar, l1/l2/m, y/n, p, 0.
- Quadrant copy flow using q/r/t/u/t_bar/v/w/0'.
- Optional pygame visualization to watch propagation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

Point = Tuple[int, int]


# -------------------------------
# Signal constants
# -------------------------------
SIG_S = "s"
SIG_SB = "s_bar"
SIG_J = "j"
SIG_JB = "j_bar"
SIG_K = "k"
SIG_L1 = "l1"
SIG_L2 = "l2"
SIG_M = "m"
SIG_Y = "y"
SIG_N = "n"
SIG_0 = "0"
SIG_P = "p"
SIG_Q = "q"
SIG_R = "r"
SIG_T = "t"
SIG_U = "u"
SIG_TB = "t_bar"
SIG_V = "v"
SIG_W = "w"
SIG_0P = "0'"

SIGNAL_COLORS: Dict[str, Tuple[int, int, int]] = {
    SIG_S: (255, 255, 0),
    SIG_SB: (255, 190, 0),
    SIG_J: (220, 70, 220),
    SIG_JB: (255, 140, 255),
    SIG_K: (0, 255, 255),
    SIG_L1: (255, 255, 255),
    SIG_L2: (150, 150, 255),
    SIG_M: (170, 0, 170),
    SIG_Y: (120, 255, 120),
    SIG_N: (110, 110, 110),
    SIG_0: (30, 80, 220),
    SIG_P: (180, 120, 60),
    SIG_Q: (0, 180, 0),
    SIG_R: (170, 170, 0),
    SIG_T: (90, 90, 180),
    SIG_U: (140, 210, 210),
    SIG_TB: (170, 90, 180),
    SIG_V: (180, 180, 50),
    SIG_W: (255, 80, 80),
    SIG_0P: (0, 210, 90),
}

SIGNAL_SPEED_PERIOD: Dict[str, int] = {
    SIG_S: 1,
    SIG_SB: 1,
    SIG_J: 1,
    SIG_JB: 1,
    SIG_K: 2,  # speed 1/2
    SIG_L1: 1,
    SIG_L2: 1,
    SIG_M: 1,
    SIG_Y: 1,
    SIG_N: 1,
    SIG_0: 1,
    SIG_P: 1,
    SIG_Q: 1,
    SIG_R: 1,
    SIG_T: 1,
    SIG_U: 3,  # speed 1/3
    SIG_TB: 1,
    SIG_V: 1,
    SIG_W: 1,
    SIG_0P: 1,
}


@dataclass
class Particle:
    kind: str
    path: List[Point]
    period: int
    meta: Dict[str, object] = field(default_factory=dict)
    idx: int = 0
    cooldown: int = 1

    def __post_init__(self) -> None:
        self.cooldown = max(1, self.period)

    @property
    def pos(self) -> Point:
        return self.path[self.idx]

    @property
    def done(self) -> bool:
        return self.idx >= len(self.path) - 1


@dataclass
class SquareTimingState:
    width: int
    step: int = 0
    mode: str = "ready"
    pos: int = 0
    first_arrival: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width < 2:
            raise ValueError("width must be at least 2")
        self.first_arrival[0] = 0

    def advance(self) -> None:
        self.step += 1

        if self.mode == "ready":
            self.pos = 1
            self._on_s_arrival()
            return

        if self.mode == "s":
            nxt = self.pos + 1
            if nxt >= self.width:
                self.mode = "done"
                return
            self.pos = nxt
            self._on_s_arrival()
            return

        if self.mode == "s_bar":
            self.pos -= 1
            if self.pos <= 0:
                self.pos = 0
                self.mode = "ready"
            return

        if self.mode == "done":
            return

        raise RuntimeError(f"invalid square mode: {self.mode}")

    def _on_s_arrival(self) -> None:
        if self.pos not in self.first_arrival:
            self.first_arrival[self.pos] = self.step
            self.mode = "s_bar"
        else:
            self.mode = "s"


class CircleEngine:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        visual: bool,
        cell_size: int,
        step_rate: float,
        render_fps: float,
        max_steps: int,
    ) -> None:
        if width < 15 or height < 15:
            raise ValueError("width/height must be at least 15")

        self.width = width
        self.height = height
        self.visual = visual
        self.cell_size = cell_size
        self.step_rate = step_rate
        self.render_fps = render_fps
        self.max_steps = max_steps

        self.step = 0
        self.particles: List[Particle] = []
        self.inside: Set[Point] = set()
        self.outside_checked: Set[Point] = set()

        # Geometry for the centered maximal square.
        self.cx = width // 2
        self.cy = height // 2
        side = min(width, height) - 4
        half = side // 2
        self.left = self.cx - half
        self.right = self.cx + half
        self.bottom = self.cy - half
        self.top = self.cy + half
        self.O: Point = (self.cx, self.cy)
        self.K: Point = (self.right, self.cy)

        self.current_p: Optional[Point] = None

        self._screen = None
        self._font = None
        self._last_step_wall = time.perf_counter()
        self._last_render_wall = time.perf_counter()
        if self.visual:
            import pygame

            pygame.init()
            self._pygame = pygame
            self._screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            pygame.display.set_caption("Circle Signals (From Scratch)")
            self._font = pygame.font.SysFont(None, 18)

    # ---------- basic helpers ----------
    def in_bounds(self, p: Point) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def manhattan_path(self, start: Point, end: Point, prefer: str = "xy") -> List[Point]:
        x, y = start
        tx, ty = end
        path: List[Point] = [(x, y)]

        axes: Iterable[str] = ("x", "y") if prefer == "xy" else ("y", "x")
        for axis in axes:
            if axis == "x":
                while x != tx:
                    x += 1 if tx > x else -1
                    path.append((x, y))
            else:
                while y != ty:
                    y += 1 if ty > y else -1
                    path.append((x, y))

        return path

    def path_via(self, start: Point, via: Point, end: Point) -> List[Point]:
        a = self.manhattan_path(start, via, prefer="xy")
        b = self.manhattan_path(via, end, prefer="xy")
        return a + b[1:]

    def spawn(
        self,
        kind: str,
        path: List[Point],
        *,
        period_override: Optional[int] = None,
        **meta: object,
    ) -> None:
        if len(path) < 2:
            return
        period = SIGNAL_SPEED_PERIOD[kind] if period_override is None else max(1, period_override)
        self.particles.append(
            Particle(kind=kind, path=path, period=period, meta=dict(meta))
        )

    # ---------- simulation step ----------
    def tick(self, on_arrival: Optional[Callable[[Particle], None]] = None) -> None:
        if self.visual:
            self._pump_events()
            self._throttle_step_rate()

        if self.step >= self.max_steps:
            raise RuntimeError(
                f"max steps reached ({self.max_steps}). Increase --max-steps to continue."
            )

        self.step += 1
        arrivals: List[Particle] = []
        survivors: List[Particle] = []

        for p in self.particles:
            p.cooldown -= 1
            if p.cooldown <= 0:
                if not p.done:
                    p.idx += 1
                p.cooldown = max(1, p.period)

            # Trail fill signals as they move.
            if p.kind in {SIG_0, SIG_0P}:
                self.inside.add(p.pos)

            if p.done:
                arrivals.append(p)
            else:
                survivors.append(p)

        self.particles = survivors

        for p in arrivals:
            if on_arrival is not None:
                on_arrival(p)

        if self.visual:
            self._maybe_render()

    def run_until(
        self,
        predicate: Callable[[], bool],
        on_arrival: Optional[Callable[[Particle], None]] = None,
    ) -> None:
        while not predicate():
            self._pump_events()
            self.tick(on_arrival=on_arrival)

    # ---------- phases ----------
    def run(self) -> Dict[str, int]:
        # 1) Boundary-driven checks in the first quadrant.
        p = (self.cx, self.top)
        while self.cx <= p[0] <= self.right and self.cy <= p[1] <= self.top:
            self.current_p = p
            inside = self._check_point(p)

            if inside:
                nxt = (p[0] + 1, p[1])
            else:
                nxt = (p[0], p[1] - 1)

            if not (self.cx <= nxt[0] <= self.right and self.cy <= nxt[1] <= self.top):
                break

            # p signal: visual hand-off to next P.
            self.spawn(SIG_P, self.manhattan_path(p, nxt), from_p=p, to_p=nxt)
            self.run_until(lambda: len(self.particles) == 0)
            p = nxt

        self.current_p = None

        # 2) Copy to second quadrant (left side) using q/r/t/u/t_bar/0'.
        self._copy_upper_right_to_upper_left()

        # 3) Copy upper half to lower half using v/w.
        self._copy_upper_to_lower()

        # Final render settle frame.
        if self.visual:
            for _ in range(5):
                self._pump_events()
                self._maybe_render(force=True)
                if self.render_fps > 0:
                    time.sleep(1.0 / self.render_fps)

        return {
            "steps": self.step,
            "inside_cells": len(self.inside),
            "outside_checked": len(self.outside_checked),
        }

    def _check_point(self, p: Point) -> bool:
        px, py = p
        q = (px, self.cy)

        flags = {
            "k_at_k": False,
            "jbar_at_p": False,
            "l_started": False,
            "l2_at_o": None,
            "m_at_o": None,
            "verdict_sent": False,
            "verdict_arrived": False,
            "inside": False,
        }

        def on_arrival(token: Particle) -> None:
            kind = token.kind
            role = str(token.meta.get("role", ""))

            if kind == SIG_J and role == "p_to_k":
                self.spawn(SIG_JB, self.manhattan_path(self.K, p), role="k_to_p")
                return

            if kind == SIG_K and role == "p_to_k":
                flags["k_at_k"] = True
                if self.K == self.O:
                    flags["m_at_o"] = self.step
                else:
                    m_dist = abs(self.K[0] - self.O[0]) + abs(self.K[1] - self.O[1])
                    self.spawn(
                        SIG_M,
                        self.manhattan_path(self.K, self.O),
                        role="k_to_o",
                        period_override=max(1, m_dist),
                    )
                return

            if kind == SIG_JB and role == "k_to_p":
                flags["jbar_at_p"] = True
                return

            if kind == SIG_L1 and role == "p_to_q":
                if q == self.O:
                    flags["l2_at_o"] = self.step
                else:
                    l2_dist = abs(q[0] - self.O[0]) + abs(q[1] - self.O[1])
                    self.spawn(
                        SIG_L2,
                        self.manhattan_path(q, self.O),
                        role="q_to_o",
                        period_override=max(1, l2_dist),
                    )
                return

            if kind == SIG_L2 and role == "q_to_o":
                flags["l2_at_o"] = self.step
                return

            if kind == SIG_M and role == "k_to_o":
                flags["m_at_o"] = self.step
                return

            if kind in {SIG_Y, SIG_N} and role == "o_to_p":
                inside = kind == SIG_Y
                flags["inside"] = inside
                flags["verdict_arrived"] = True
                if inside:
                    self.inside.add(p)
                    # 0 signal fills from P down to axis (Q).
                    if py > self.cy:
                        self.spawn(SIG_0, self.manhattan_path(p, q), role="fill_down")
                else:
                    self.outside_checked.add(p)
                return

        # Start synchronization signals.
        if p == self.K:
            flags["k_at_k"] = True
            flags["jbar_at_p"] = True
            if self.K == self.O:
                flags["m_at_o"] = self.step
            else:
                m_dist = abs(self.K[0] - self.O[0]) + abs(self.K[1] - self.O[1])
                self.spawn(
                    SIG_M,
                    self.manhattan_path(self.K, self.O),
                    role="k_to_o",
                    period_override=max(1, m_dist),
                )
        else:
            self.spawn(SIG_J, self.manhattan_path(p, self.K), role="p_to_k")
            self.spawn(SIG_K, self.manhattan_path(p, self.K), role="p_to_k")

        def done() -> bool:
            return bool(flags["verdict_arrived"] and len(self.particles) == 0)

        while not done():
            self._pump_events()
            self.tick(on_arrival=on_arrival)

            # After j_bar returns to P and k is at K, start l1 exactly once.
            if flags["jbar_at_p"] and flags["k_at_k"] and not flags["l_started"]:
                flags["l_started"] = True
                if p == q:
                    if q == self.O:
                        flags["l2_at_o"] = self.step
                    else:
                        l2_dist = abs(q[0] - self.O[0]) + abs(q[1] - self.O[1])
                        self.spawn(
                            SIG_L2,
                            self.manhattan_path(q, self.O),
                            role="q_to_o",
                            period_override=max(1, l2_dist),
                        )
                else:
                    l1_dist = abs(p[0] - q[0]) + abs(p[1] - q[1])
                    self.spawn(
                        SIG_L1,
                        self.manhattan_path(p, q),
                        role="p_to_q",
                        period_override=max(1, l1_dist),
                    )

            # Once both arrivals at O are known, send verdict once.
            if (
                flags["l2_at_o"] is not None
                and flags["m_at_o"] is not None
                and not flags["verdict_sent"]
            ):
                flags["verdict_sent"] = True
                inside = int(flags["l2_at_o"]) <= int(flags["m_at_o"])
                verdict = SIG_Y if inside else SIG_N
                if self.O == p:
                    flags["inside"] = inside
                    flags["verdict_arrived"] = True
                    if inside:
                        self.inside.add(p)
                else:
                    self.spawn(verdict, self.path_via(self.O, q, p), role="o_to_p")

        return bool(flags["inside"])

    def _copy_upper_right_to_upper_left(self) -> None:
        upper_right = sorted(
            [p for p in self.inside if p[0] >= self.cx and p[1] >= self.cy],
            key=lambda t: (t[1], t[0]),
        )

        for src in upper_right:
            x, y = src
            mx = 2 * self.cx - x
            dst = (mx, y)
            axis = (self.cx, y)

            # q and r are emitted together.
            self.spawn(SIG_Q, self.manhattan_path(src, axis), role="to_axis", src=src, dst=dst)
            self.spawn(SIG_R, self.manhattan_path(src, axis), role="to_axis", src=src, dst=dst)

        def on_arrival(token: Particle) -> None:
            kind = token.kind
            src = token.meta.get("src")
            dst = token.meta.get("dst")
            if not isinstance(src, tuple) or not isinstance(dst, tuple):
                return

            axis = (self.cx, src[1])

            if kind == SIG_Q:
                # At axis: send t to mirrored destination and u downward slowly.
                self.spawn(SIG_T, self.manhattan_path(axis, dst), role="axis_to_mirror", src=src, dst=dst)
                slow_u_target = (self.cx, max(self.bottom, self.cy - (src[1] - self.cy)))
                if slow_u_target != axis:
                    self.spawn(SIG_U, self.manhattan_path(axis, slow_u_target), role="slow_down")
                return

            if kind == SIG_T:
                # Mirror point confirmed.
                self.inside.add(dst)

                # Return trace and fill-to-axis trace for visibility.
                self.spawn(SIG_TB, self.manhattan_path(dst, src), role="mirror_return")
                if dst[1] > self.cy:
                    self.spawn(SIG_0P, self.manhattan_path(dst, (dst[0], self.cy)), role="fill_to_axis")
                return

        self.run_until(lambda: len(self.particles) == 0, on_arrival=on_arrival)

    def _copy_upper_to_lower(self) -> None:
        upper_half = sorted(
            [p for p in self.inside if p[1] >= self.cy],
            key=lambda t: (t[0], t[1]),
        )

        for src in upper_half:
            x, y = src
            my = 2 * self.cy - y
            dst = (x, my)
            axis = (x, self.cy)
            if dst == src:
                continue

            self.spawn(SIG_V, self.manhattan_path(src, axis), role="to_axis", src=src, dst=dst)

        def on_arrival(token: Particle) -> None:
            kind = token.kind
            src = token.meta.get("src")
            dst = token.meta.get("dst")
            if not isinstance(src, tuple) or not isinstance(dst, tuple):
                return

            axis = (src[0], self.cy)
            if kind == SIG_V:
                self.spawn(SIG_W, self.manhattan_path(axis, dst), role="axis_to_lower", src=src, dst=dst)
                return

            if kind == SIG_W:
                self.inside.add(dst)
                return

        self.run_until(lambda: len(self.particles) == 0, on_arrival=on_arrival)

    # ---------- rendering ----------
    def _throttle_step_rate(self) -> None:
        if self.step_rate <= 0:
            return
        target_dt = 1.0 / self.step_rate
        while True:
            now = time.perf_counter()
            elapsed = now - self._last_step_wall
            if elapsed >= target_dt:
                self._last_step_wall = now
                return
            # Sleep in short slices so UI still updates.
            time.sleep(min(0.002, target_dt - elapsed))
            self._pump_events()
            self._maybe_render()

    def _maybe_render(self, force: bool = False) -> None:
        if not self.visual:
            return
        now = time.perf_counter()
        if force or self.render_fps <= 0:
            self._draw_frame()
            self._last_render_wall = now
            return
        render_dt = 1.0 / self.render_fps
        if now - self._last_render_wall >= render_dt:
            self._draw_frame()
            self._last_render_wall = now

    def _pump_events(self) -> None:
        if not self.visual:
            return
        assert self._pygame is not None
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                raise KeyboardInterrupt

    def _draw_frame(self) -> None:
        assert self._screen is not None
        assert self._font is not None

        self._screen.fill((0, 0, 0))

        # Base cells.
        for y in range(self.height):
            for x in range(self.width):
                color = (28, 28, 28)
                if (x, y) in self.inside:
                    color = (25, 140, 35)
                elif (x, y) in self.outside_checked:
                    color = (70, 70, 70)

                rect = self._cell_rect((x, y))
                self._pygame.draw.rect(self._screen, color, rect)

        # Square boundary.
        self._draw_square_outline()

        # Center O and reference K.
        self._draw_marker(self.O, (255, 255, 255))
        self._draw_marker(self.K, (0, 255, 255))

        # Current P highlight.
        if self.current_p is not None:
            rect = self._cell_rect(self.current_p)
            self._pygame.draw.rect(self._screen, (255, 80, 80), rect, width=2)

        # Active signals.
        for p in self.particles:
            col = SIGNAL_COLORS.get(p.kind, (255, 0, 0))
            cx, cy = self._cell_center(p.pos)
            self._pygame.draw.circle(self._screen, col, (cx, cy), max(2, self.cell_size // 3))

        txt = self._font.render(
            f"step={self.step} particles={len(self.particles)} inside={len(self.inside)}",
            True,
            (245, 245, 245),
        )
        self._screen.blit(txt, (6, 6))

        self._pygame.display.flip()

    def _cell_rect(self, p: Point):
        assert self._pygame is not None
        x, y = p
        sx = x * self.cell_size
        sy = (self.height - 1 - y) * self.cell_size
        return self._pygame.Rect(sx, sy, self.cell_size, self.cell_size)

    def _cell_center(self, p: Point) -> Tuple[int, int]:
        x, y = p
        sx = x * self.cell_size + self.cell_size // 2
        sy = (self.height - 1 - y) * self.cell_size + self.cell_size // 2
        return sx, sy

    def _draw_square_outline(self) -> None:
        assert self._pygame is not None
        for x in range(self.left, self.right + 1):
            for y in (self.bottom, self.top):
                self._pygame.draw.rect(self._screen, (90, 90, 130), self._cell_rect((x, y)), width=1)
        for y in range(self.bottom, self.top + 1):
            for x in (self.left, self.right):
                self._pygame.draw.rect(self._screen, (90, 90, 130), self._cell_rect((x, y)), width=1)

    def _draw_marker(self, p: Point, col: Tuple[int, int, int]) -> None:
        assert self._pygame is not None
        cx, cy = self._cell_center(p)
        r = max(2, self.cell_size // 4)
        self._pygame.draw.circle(self._screen, col, (cx, cy), r)

    def close(self) -> None:
        if self.visual:
            assert self._pygame is not None
            self._pygame.quit()


# ---------------------------------
# Public run functions
# ---------------------------------
def run_square_check(width: int, max_i: int, max_steps: int, verbose: bool) -> int:
    if max_i >= width:
        raise ValueError(f"max_i must be < width (got max_i={max_i}, width={width})")

    state = SquareTimingState(width=width)
    while state.step < max_steps and max_i not in state.first_arrival and state.mode != "done":
        state.advance()
        if verbose:
            print(f"step={state.step:4d} mode={state.mode:>6s} pos={state.pos:3d}")

    print("i | first_arrival | expected(i^2) | ok")
    print("--+---------------+---------------+----")
    ok = True
    for i in range(1, max_i + 1):
        actual = state.first_arrival.get(i)
        expected = i * i
        line_ok = actual == expected
        ok = ok and line_ok
        astr = "-" if actual is None else str(actual)
        print(f"{i:2d} | {astr:13s} | {expected:13d} | {str(line_ok):>4s}")

    if ok:
        print("\nSquare(s) timing check PASSED for requested range.")
        return 0

    print("\nSquare(s) timing check FAILED for requested range.")
    return 1


# ---------------------------------
# CLI
# ---------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="From-scratch signal-based circle generator")
    p.add_argument(
        "--mode",
        choices=["square-check", "circle"],
        default="circle",
        help="square-check: verify i^2 timing only; circle: full 2D signal run",
    )
    p.add_argument("--width", type=int, default=61, help="Grid width")
    p.add_argument("--height", type=int, default=61, help="Grid height")
    p.add_argument("--max-i", type=int, default=20, help="Square-check upper i")
    p.add_argument("--max-steps", type=int, default=200000, help="Simulation hard step limit")
    p.add_argument("--visual", action="store_true", help="Enable pygame visualization")
    p.add_argument(
        "--step-rate",
        type=float,
        default=240.0,
        help="Simulation steps per second in visual mode (0 = unlimited)",
    )
    p.add_argument(
        "--render-fps",
        type=float,
        default=60.0,
        help="Render frames per second in visual mode (0 = render every loop)",
    )
    p.add_argument("--cell-size", type=int, default=10, help="Pixel size per cell when --visual")
    p.add_argument("--verbose", action="store_true", help="Verbose step logging for square-check")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "square-check":
        raise SystemExit(
            run_square_check(
                width=args.width,
                max_i=args.max_i,
                max_steps=args.max_steps,
                verbose=args.verbose,
            )
        )

    engine = CircleEngine(
        width=args.width,
        height=args.height,
        visual=args.visual,
        cell_size=args.cell_size,
        step_rate=args.step_rate,
        render_fps=args.render_fps,
        max_steps=args.max_steps,
    )

    try:
        summary = engine.run()
        print("Circle run complete")
        print(f"- steps: {summary['steps']}")
        print(f"- inside cells: {summary['inside_cells']}")
        print(f"- outside checked: {summary['outside_checked']}")
        if args.visual:
            print("Close the pygame window to exit.")
            while True:
                engine._pump_events()
                engine._maybe_render(force=True)
                if args.render_fps > 0:
                    time.sleep(1.0 / args.render_fps)
                else:
                    time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
