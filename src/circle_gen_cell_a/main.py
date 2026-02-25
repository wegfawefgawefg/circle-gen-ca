"""Strict local-signal circle generator.

Goal: keep control in local cell state + moving local signals, avoiding a global
check/phase orchestrator.

Notes:
- This is still a practical simulator for a finite grid, not a formal proof artifact.
- The previous path-planned simulator is preserved in signal_simulator.py.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

# Paper signal names
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

# Internal helper signal: carries "k reached K" back to P over j trace.
SIG_KH = "k_hit"

# Directions
N = "N"
S = "S"
E = "E"
W = "W"

DIR_DELTA = {
    N: (0, 1),
    S: (0, -1),
    E: (1, 0),
    W: (-1, 0),
}

OPPOSITE = {
    N: S,
    S: N,
    E: W,
    W: E,
}

SIGNAL_COLORS: Dict[str, Tuple[int, int, int]] = {
    SIG_J: (220, 70, 220),
    SIG_JB: (255, 140, 255),
    SIG_K: (0, 255, 255),
    SIG_KH: (255, 220, 80),
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

BASE_PERIOD = {
    SIG_J: 1,
    SIG_JB: 1,
    SIG_K: 2,
    SIG_KH: 1,
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
    SIG_U: 3,
    SIG_TB: 1,
    SIG_V: 1,
    SIG_W: 1,
    SIG_0P: 1,
}


@dataclass
class Token:
    kind: str
    x: int
    y: int
    period: int
    cooldown: int = 1
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cooldown = max(1, self.period)


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


class StrictLocalCA:
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

        # Geometry for centered maximal square.
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

        # Local cell state planes
        self.inside: Set[Point] = set()
        self.outside_checked: Set[Point] = set()

        self.p_active = [[False for _ in range(width)] for __ in range(height)]
        self.p_busy = [[False for _ in range(width)] for __ in range(height)]
        self.p_id = [[0 for _ in range(width)] for __ in range(height)]
        self.p_jbar = [[False for _ in range(width)] for __ in range(height)]
        self.p_khit = [[False for _ in range(width)] for __ in range(height)]
        self.p_l_started = [[False for _ in range(width)] for __ in range(height)]

        start = (self.cx, self.top)
        self.p_active[start[1]][start[0]] = True
        self.p_id[start[1]][start[0]] = 1

        # Trace planes: direction + check-id ownership.
        self.trace_j_dir = [[None for _ in range(width)] for __ in range(height)]
        self.trace_j_id = [[0 for _ in range(width)] for __ in range(height)]
        self.trace_l1_dir = [[None for _ in range(width)] for __ in range(height)]
        self.trace_l1_id = [[0 for _ in range(width)] for __ in range(height)]
        self.trace_l2_dir = [[None for _ in range(width)] for __ in range(height)]
        self.trace_l2_id = [[0 for _ in range(width)] for __ in range(height)]

        # O-local arrival memory keyed by check id.
        self.o_l2_arrival: Dict[int, int] = {}
        self.o_m_arrival: Dict[int, int] = {}
        self.o_verdict_sent: Set[int] = set()

        # Copy emission guards for inside cells.
        self.sent_lr: Set[Point] = set()
        self.sent_tb: Set[Point] = set()

        self.tokens: List[Token] = []

        # Rendering internals
        self._screen = None
        self._font = None
        self._last_step_wall = time.perf_counter()
        self._last_render_wall = time.perf_counter()

        if self.visual:
            import pygame

            pygame.init()
            self._pygame = pygame
            self._screen = pygame.display.set_mode((width * cell_size, height * cell_size))
            pygame.display.set_caption("Strict Local CA Circle")
            self._font = pygame.font.SysFont(None, 18)

    # ---------- low-level helpers ----------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _dir_to_k(self, x: int, y: int) -> Optional[str]:
        kx, ky = self.K
        if y > ky:
            return S
        if y < ky:
            return N
        if x < kx:
            return E
        if x > kx:
            return W
        return None

    def _dir_to_o(self, x: int, y: int) -> Optional[str]:
        ox, oy = self.O
        if y > oy:
            return S
        if y < oy:
            return N
        if x > ox:
            return W
        if x < ox:
            return E
        return None

    def _step_dir(self, x: int, y: int, d: str) -> Optional[Point]:
        dx, dy = DIR_DELTA[d]
        nx, ny = x + dx, y + dy
        if self.in_bounds(nx, ny):
            return (nx, ny)
        return None

    def _dist_to_o(self, x: int, y: int) -> int:
        ox, oy = self.O
        return abs(x - ox) + abs(y - oy)

    def _spawn(self, kind: str, x: int, y: int, *, period: Optional[int] = None, **meta: object) -> None:
        p = BASE_PERIOD[kind] if period is None else max(1, period)
        self.tokens.append(Token(kind=kind, x=x, y=y, period=p, meta=dict(meta)))

    def _set_trace(
        self,
        dir_plane: List[List[Optional[str]]],
        id_plane: List[List[int]],
        x: int,
        y: int,
        d: str,
        cid: int,
    ) -> None:
        dir_plane[y][x] = d
        id_plane[y][x] = cid

    def _get_trace(
        self,
        dir_plane: List[List[Optional[str]]],
        id_plane: List[List[int]],
        x: int,
        y: int,
        cid: int,
    ) -> Optional[str]:
        if id_plane[y][x] == cid:
            return dir_plane[y][x]
        return None

    # ---------- local-check progression ----------
    def _activate_pending_checks(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                if self.p_active[y][x] and not self.p_busy[y][x]:
                    cid = self.p_id[y][x]
                    self.p_busy[y][x] = True
                    self.p_jbar[y][x] = False
                    self.p_khit[y][x] = False
                    self.p_l_started[y][x] = False
                    self.o_l2_arrival.pop(cid, None)
                    self.o_m_arrival.pop(cid, None)
                    self.o_verdict_sent.discard(cid)
                    if (x, y) == self.K:
                        # Degenerate case: P already at K.
                        self.p_jbar[y][x] = True
                        self.p_khit[y][x] = True
                        self._spawn(
                            SIG_M,
                            x,
                            y,
                            id=cid,
                            period=max(1, self._dist_to_o(x, y)),
                        )
                    else:
                        self._spawn(SIG_J, x, y, id=cid)
                        self._spawn(SIG_K, x, y, id=cid)

    def _maybe_send_verdict(self, cid: int, x: int, y: int, spawned: List[Token]) -> None:
        if cid in self.o_verdict_sent:
            return
        l2_t = self.o_l2_arrival.get(cid)
        m_t = self.o_m_arrival.get(cid)
        if l2_t is None or m_t is None:
            return
        self.o_verdict_sent.add(cid)
        verdict = SIG_Y if l2_t <= m_t else SIG_N
        spawned.append(Token(verdict, x, y, period=1, meta={"id": cid, "stage": "l2"}))

    def _start_l1_if_ready(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                if not self.p_active[y][x] or not self.p_busy[y][x]:
                    continue
                if self.p_l_started[y][x]:
                    continue
                if not (self.p_jbar[y][x] and self.p_khit[y][x]):
                    continue

                cid = self.p_id[y][x]
                self.p_l_started[y][x] = True

                if y > self.cy:
                    self._spawn(SIG_L1, x, y, id=cid, period=max(1, y - self.cy))
                else:
                    self._spawn(SIG_L2, x, y, id=cid, period=max(1, self._dist_to_o(x, y)))

    def _emit_copy_signals(self) -> None:
        points = list(self.inside)
        for x, y in points:
            p = (x, y)

            # left-right mirror copy from upper-right side
            if p not in self.sent_lr and x > self.cx and y >= self.cy:
                d = x - self.cx
                self._spawn(SIG_Q, x, y, remaining=d, mirror_dist=d)
                self._spawn(SIG_R, x, y, remaining=d)
                self.sent_lr.add(p)

            # top-bottom mirror copy from upper half
            if p not in self.sent_tb and y > self.cy:
                d = y - self.cy
                self._spawn(SIG_V, x, y, remaining=d, mirror_dist=d)
                self.sent_tb.add(p)

    def _pending_copy_work(self) -> bool:
        for x, y in self.inside:
            p = (x, y)
            if x > self.cx and y >= self.cy and p not in self.sent_lr:
                return True
            if y > self.cy and p not in self.sent_tb:
                return True
        return False

    def _has_active_p(self) -> bool:
        for row in self.p_active:
            for v in row:
                if v:
                    return True
        return False

    def _advance_to_next_p(self, x: int, y: int, cid: int, inside: bool) -> None:
        self.p_active[y][x] = False
        self.p_busy[y][x] = False

        if inside:
            self.inside.add((x, y))
            if y > self.cy:
                self._spawn(SIG_0, x, y, id=cid, remaining=y - self.cy)
            nx, ny = x + 1, y
        else:
            self.outside_checked.add((x, y))
            nx, ny = x, y - 1

        if self.cx <= nx <= self.right and self.cy <= ny <= self.top:
            self.p_active[ny][nx] = True
            self.p_id[ny][nx] = cid + 1
            self._spawn(SIG_P, x, y, to_x=nx, to_y=ny)

    # ---------- token dynamics ----------
    def _handle_token(self, tok: Token, spawned: List[Token]) -> Optional[Token]:
        x, y = tok.x, tok.y
        cid = int(tok.meta.get("id", 0))

        if tok.kind == SIG_J:
            if (x, y) == self.K:
                spawned.append(Token(SIG_JB, x, y, period=1, meta={"id": cid}))
                return None
            d = self._dir_to_k(x, y)
            if d is None:
                return None
            nxt = self._step_dir(x, y, d)
            if nxt is None:
                return None
            nx, ny = nxt
            self._set_trace(self.trace_j_dir, self.trace_j_id, nx, ny, OPPOSITE[d], cid)
            tok.x, tok.y = nx, ny
            if (nx, ny) == self.K:
                spawned.append(Token(SIG_JB, nx, ny, period=1, meta={"id": cid}))
                return None
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_K:
            if (x, y) == self.K:
                # k reached K: send a local notification back to P and launch m to O.
                spawned.append(Token(SIG_KH, x, y, period=1, meta={"id": cid}))
                spawned.append(
                    Token(
                        SIG_M,
                        x,
                        y,
                        period=max(1, self._dist_to_o(x, y)),
                        meta={"id": cid},
                    )
                )
                return None
            d = self._dir_to_k(x, y)
            if d is None:
                return None
            nxt = self._step_dir(x, y, d)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            if (tok.x, tok.y) == self.K:
                spawned.append(Token(SIG_KH, tok.x, tok.y, period=1, meta={"id": cid}))
                spawned.append(
                    Token(
                        SIG_M,
                        tok.x,
                        tok.y,
                        period=max(1, self._dist_to_o(tok.x, tok.y)),
                        meta={"id": cid},
                    )
                )
                return None
            tok.cooldown = tok.period
            return tok

        if tok.kind in {SIG_JB, SIG_KH}:
            d = self._get_trace(self.trace_j_dir, self.trace_j_id, x, y, cid)
            if d is None:
                return None
            nxt = self._step_dir(x, y, d)
            if nxt is None:
                return None
            nx, ny = nxt
            tok.x, tok.y = nx, ny
            if self.p_active[ny][nx] and self.p_busy[ny][nx] and self.p_id[ny][nx] == cid:
                if tok.kind == SIG_JB:
                    self.p_jbar[ny][nx] = True
                else:
                    self.p_khit[ny][nx] = True
                return None
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_L1:
            if y == self.cy:
                spawned.append(
                    Token(
                        SIG_L2,
                        x,
                        y,
                        period=max(1, self._dist_to_o(x, y)),
                        meta={"id": cid},
                    )
                )
                return None
            nxt = self._step_dir(x, y, S)
            if nxt is None:
                return None
            nx, ny = nxt
            self._set_trace(self.trace_l1_dir, self.trace_l1_id, nx, ny, N, cid)
            tok.x, tok.y = nx, ny
            if ny == self.cy:
                spawned.append(
                    Token(
                        SIG_L2,
                        nx,
                        ny,
                        period=max(1, self._dist_to_o(nx, ny)),
                        meta={"id": cid},
                    )
                )
                return None
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_L2:
            if (x, y) == self.O:
                if cid not in self.o_l2_arrival:
                    self.o_l2_arrival[cid] = self.step
                self._maybe_send_verdict(cid, x, y, spawned)
                return None
            d = self._dir_to_o(x, y)
            if d is None:
                return None
            nxt = self._step_dir(x, y, d)
            if nxt is None:
                return None
            nx, ny = nxt
            self._set_trace(self.trace_l2_dir, self.trace_l2_id, nx, ny, OPPOSITE[d], cid)
            tok.x, tok.y = nx, ny
            if (nx, ny) == self.O:
                if cid not in self.o_l2_arrival:
                    self.o_l2_arrival[cid] = self.step
                self._maybe_send_verdict(cid, nx, ny, spawned)
                return None
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_M:
            if (x, y) == self.O:
                if cid not in self.o_m_arrival:
                    self.o_m_arrival[cid] = self.step
                self._maybe_send_verdict(cid, x, y, spawned)
                return None
            d = self._dir_to_o(x, y)
            if d is None:
                return None
            nxt = self._step_dir(x, y, d)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            if (tok.x, tok.y) == self.O:
                if cid not in self.o_m_arrival:
                    self.o_m_arrival[cid] = self.step
                self._maybe_send_verdict(cid, tok.x, tok.y, spawned)
                return None
            tok.cooldown = tok.period
            return tok

        if tok.kind in {SIG_Y, SIG_N}:
            if self.p_active[y][x] and self.p_busy[y][x] and self.p_id[y][x] == cid:
                self._advance_to_next_p(x, y, cid, tok.kind == SIG_Y)
                return None

            stage = str(tok.meta.get("stage", "l2"))
            d: Optional[str]
            if stage == "l2":
                d = self._get_trace(self.trace_l2_dir, self.trace_l2_id, x, y, cid)
                if d is None:
                    tok.meta["stage"] = "l1"
                    d = self._get_trace(self.trace_l1_dir, self.trace_l1_id, x, y, cid)
            else:
                d = self._get_trace(self.trace_l1_dir, self.trace_l1_id, x, y, cid)

            if d is None:
                return None
            nxt = self._step_dir(x, y, d)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            nx, ny = nxt
            if self.p_active[ny][nx] and self.p_busy[ny][nx] and self.p_id[ny][nx] == cid:
                self._advance_to_next_p(nx, ny, cid, tok.kind == SIG_Y)
                return None
            tok.cooldown = tok.period
            return tok

        if tok.kind in {SIG_0, SIG_0P}:
            self.inside.add((x, y))
            rem = int(tok.meta.get("remaining", 0))
            if rem <= 0 or y <= self.cy:
                return None
            nxt = self._step_dir(x, y, S)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            tok.meta["remaining"] = rem - 1
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_P:
            tx = int(tok.meta.get("to_x", x))
            ty = int(tok.meta.get("to_y", y))
            if (x, y) == (tx, ty):
                return None
            if x < tx:
                d = E
            elif x > tx:
                d = W
            elif y < ty:
                d = N
            else:
                d = S
            nxt = self._step_dir(x, y, d)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            if (tok.x, tok.y) == (tx, ty):
                return None
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_Q:
            rem = int(tok.meta.get("remaining", 0))
            full = int(tok.meta.get("mirror_dist", rem))
            if rem <= 0:
                spawned.append(Token(SIG_T, x, y, period=1, meta={"remaining": full}))
                return None
            nxt = self._step_dir(x, y, W)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            tok.meta["remaining"] = rem - 1
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_R:
            return None

        if tok.kind == SIG_T:
            rem = int(tok.meta.get("remaining", 0))
            if rem <= 0:
                self.inside.add((x, y))
                if y > self.cy:
                    spawned.append(Token(SIG_0P, x, y, period=1, meta={"remaining": y - self.cy}))
                return None
            nxt = self._step_dir(x, y, W)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            tok.meta["remaining"] = rem - 1
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_U:
            rem = int(tok.meta.get("remaining", 0))
            if rem <= 0:
                return None
            nxt = self._step_dir(x, y, S)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            tok.meta["remaining"] = rem - 1
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_TB:
            return None

        if tok.kind == SIG_V:
            rem = int(tok.meta.get("remaining", 0))
            full = int(tok.meta.get("mirror_dist", rem))
            if rem <= 0:
                spawned.append(Token(SIG_W, x, y, period=1, meta={"remaining": full}))
                return None
            nxt = self._step_dir(x, y, S)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            tok.meta["remaining"] = rem - 1
            tok.cooldown = tok.period
            return tok

        if tok.kind == SIG_W:
            rem = int(tok.meta.get("remaining", 0))
            if rem <= 0:
                self.inside.add((x, y))
                return None
            nxt = self._step_dir(x, y, S)
            if nxt is None:
                return None
            tok.x, tok.y = nxt
            tok.meta["remaining"] = rem - 1
            tok.cooldown = tok.period
            return tok

        return None

    # ---------- timing / rendering ----------
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

    # ---------- stepping ----------
    def _done(self) -> bool:
        if self.tokens:
            return False
        if self._has_active_p():
            return False
        if self._pending_copy_work():
            return False
        return True

    def tick(self) -> None:
        if self.visual:
            self._pump_events()
            self._throttle_step_rate()

        if self.step >= self.max_steps:
            raise RuntimeError(
                f"max steps reached ({self.max_steps}). Increase --max-steps to continue."
            )

        self.step += 1

        # Local state emits checks and copy signals.
        self._activate_pending_checks()
        self._start_l1_if_ready()
        self._emit_copy_signals()

        spawned: List[Token] = []
        survivors: List[Token] = []

        for tok in self.tokens:
            tok.cooldown -= 1
            if tok.cooldown > 0:
                survivors.append(tok)
                continue

            nxt = self._handle_token(tok, spawned)
            if nxt is not None:
                survivors.append(nxt)

        self.tokens = survivors + spawned

        if self.visual:
            self._maybe_render()

    def run(self) -> Dict[str, int]:
        while not self._done():
            self.tick()

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
            "tokens": len(self.tokens),
        }

    # ---------- drawing ----------
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

    def _draw_marker(self, p: Point, col: Tuple[int, int, int]) -> None:
        assert self._pygame is not None
        cx, cy = self._cell_center(p)
        r = max(2, self.cell_size // 4)
        self._pygame.draw.circle(self._screen, col, (cx, cy), r)

    def _draw_square_outline(self) -> None:
        assert self._pygame is not None
        for x in range(self.left, self.right + 1):
            for y in (self.bottom, self.top):
                self._pygame.draw.rect(
                    self._screen, (90, 90, 130), self._cell_rect((x, y)), width=1
                )
        for y in range(self.bottom, self.top + 1):
            for x in (self.left, self.right):
                self._pygame.draw.rect(
                    self._screen, (90, 90, 130), self._cell_rect((x, y)), width=1
                )

    def _draw_frame(self) -> None:
        assert self._screen is not None
        assert self._font is not None

        self._screen.fill((0, 0, 0))

        for y in range(self.height):
            for x in range(self.width):
                color = (26, 26, 26)
                if (x, y) in self.inside:
                    color = (25, 140, 35)
                elif (x, y) in self.outside_checked:
                    color = (70, 70, 70)
                self._pygame.draw.rect(self._screen, color, self._cell_rect((x, y)))

        self._draw_square_outline()
        self._draw_marker(self.O, (255, 255, 255))
        self._draw_marker(self.K, (0, 255, 255))

        # Highlight active P cells.
        for y in range(self.height):
            for x in range(self.width):
                if self.p_active[y][x]:
                    self._pygame.draw.rect(
                        self._screen,
                        (255, 80, 80),
                        self._cell_rect((x, y)),
                        width=2,
                    )

        for tok in self.tokens:
            col = SIGNAL_COLORS.get(tok.kind, (255, 0, 0))
            cx, cy = self._cell_center((tok.x, tok.y))
            self._pygame.draw.circle(self._screen, col, (cx, cy), max(2, self.cell_size // 3))

        txt = self._font.render(
            f"step={self.step} tok={len(self.tokens)} inside={len(self.inside)}",
            True,
            (245, 245, 245),
        )
        self._screen.blit(txt, (6, 6))

        self._pygame.display.flip()

    def close(self) -> None:
        if self.visual:
            assert self._pygame is not None
            self._pygame.quit()


# ---------- utility mode: square timing ----------
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


# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict local-signal CA circle generator")
    p.add_argument(
        "--mode",
        choices=["square-check", "circle"],
        default="circle",
        help="square-check: verify i^2 timing only; circle: strict local signal run",
    )
    p.add_argument("--width", type=int, default=61, help="Grid width")
    p.add_argument("--height", type=int, default=61, help="Grid height")
    p.add_argument("--max-i", type=int, default=20, help="Square-check upper i")
    p.add_argument("--max-steps", type=int, default=600000, help="Simulation hard step limit")
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

    engine = StrictLocalCA(
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
        print("Strict local CA run complete")
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
