"""Thin adapters that let root policies operate on variant boards and prompts."""

import json
import sys
import time

import agents as root_agents
import questions as root_questions

from . import questions


class JevAgent(root_agents.JevAgent):
    def __init__(self, mode="rich", variant="standard", lean=False, payload="top12", pipeline=False, api_key=None):
        super().__init__(mode=mode, api_key=api_key, payload=payload, pipeline=pipeline)
        self.variant = variant
        self.lean = lean
        self._width = 12 if variant == "pento" else 10
        self._height = 20

    def _rich_candidates(self, board, piece, next_piece, candidates):
        infos = []
        for rot, col in candidates:
            sim, cleared = board.simulate(piece, rot, col)
            cells = board.placement_cells(piece, rot, col)
            infos.append({"id": f"r{rot}c{col}", "rotation": rot, "column": col,
                          "landing_row": board.placement_bottom_row(piece, rot, col),
                          "occupied_cells": [[c, r] for c, r in sorted(cells, key=lambda x: (x[1], x[0]))],
                          "lines_cleared": cleared,
                          "afterstate": {"holes": sim.hole_count(), "max_height": sim.max_height(),
                                         "aggregate_height": sim.aggregate_height(), "bumpiness": sim.bumpiness()},
                          "lookahead": root_agents.one_piece_lookahead(sim, next_piece) if next_piece else None})
        return infos

    def _rich_state(self, board, piece, next_piece, next_pieces, lines_cleared_total, candidates):
        infos = self._rich_candidates(board, piece, next_piece, candidates)
        piece_table = getattr(board, "pieces", root_agents.tetris.PIECES)
        width = getattr(board, "width", root_agents.tetris.WIDTH)
        height = getattr(board, "height", root_agents.tetris.HEIGHT)
        shape = piece_table[piece][0]
        spawn_col = max(0, (width - max(c for c, _ in shape) - 1) // 2)
        state = {"coordinate_system": questions.coordinate_system(
                    f"integers 0 through {width - 1}, left to right",
                    f"integers 0 through {height - 1}, top to bottom (row 0 is top)",
                    "12 free pentominoes (F, I, L, N, P, T, U, V, W, X, Y, Z)" if self.variant == "pento" else "standard 7 tetrominoes",
                    self.variant == "hatetris"),
                 "board": ["".join("#" if x else "." for x in row) for row in board.grid],
                 "active_piece": {"kind": piece, "rotation": 0,
                                  "occupied_cells": [[spawn_col + c, r] for c, r in shape]},
                 "next_pieces": [] if self.variant == "hatetris" else list(next_pieces or ([next_piece] if next_piece else [])),
                 "lines": lines_cleared_total,
                 "allowed_placements": [x["id"] for x in infos]}
        if self.payload not in ("nostate", "both") and not self.lean:
            state["legal_placements"] = [{k: v for k, v in x.items() if k != "lookahead"} for x in infos]
        if self.payload in ("top12", "both"):
            state["legal_placements_note"] = "The candidate list was pre-filtered by code to the 12 best oracle-scored placements."
        return state

    def _build_state(self, board, piece, next_piece, next_pieces=None, lines_cleared_total=0, candidates=None):
        if self.mode == "rich":
            return self._rich_state(board, piece, next_piece, next_pieces, lines_cleared_total, candidates or [])
        table = getattr(board, "pieces", root_agents.tetris.PIECES)
        current = root_agents.tetris.shape_ascii(table[piece][0])
        upcoming = root_agents.tetris.shape_ascii(table[next_piece][0]) if next_piece else "unknown (adversarial)"
        return {"board": board.render(), "current_piece": current, "next_piece": upcoming}

    def _build_questions(self, board, piece, next_piece, candidates):
        if self.mode != "rich":
            criteria = {}
            table = getattr(board, "pieces", root_agents.tetris.PIECES)
            for rot, col in candidates:
                if self.mode == "board":
                    width = max(c for c, _ in table[piece][rot]) + 1
                    row = board.placement_bottom_row(piece, rot, col)
                    criteria[f"r{rot}c{col}"] = f"rotation {rot}, columns {col}-{col + width - 1}, resting row {row}"
                else:
                    sim, cleared = board.simulate(piece, rot, col)
                    criteria[f"r{rot}c{col}"] = f"resulting board:\n{sim.render()}\nlines cleared: {cleared}"
            return {"placement": {"type": "choice", "instructions": questions.hatetris_instructions() if self.variant == "hatetris" else root_questions.BASIC_PLACEMENT_INSTRUCTIONS, "criteria": criteria},
                    "topout": {"type": "score", "instructions": root_questions.TOPOUT_INSTRUCTIONS, "criteria": root_questions.TOPOUT_SCORE_LEVELS}}
        infos = self._rich_candidates(board, piece, next_piece, candidates)
        criteria = {x["id"]: questions.rich_candidate_criteria(
            x, lean=self.lean, include_lookahead=self.variant != "hatetris") for x in infos}
        return {"placement": {"type": "choice", "instructions": questions.hatetris_instructions() if self.variant == "hatetris" else root_questions.RICH_PLACEMENT_INSTRUCTIONS,
                               "criteria": criteria},
                "topout": {"type": "score", "instructions": root_questions.TOPOUT_INSTRUCTIONS,
                           "criteria": root_questions.TOPOUT_SCORE_LEVELS}}


def install_latency_recorder():
    samples = []
    real = root_agents.call_jev
    def wrapped(payload, api_key):
        start = time.perf_counter()
        try:
            return real(payload, api_key)
        finally:
            samples.append((time.perf_counter() - start) * 1000)
    root_agents.call_jev = wrapped
    return samples, lambda: setattr(root_agents, "call_jev", real)


def percentiles(samples):
    if not samples:
        return None, None
    ordered = sorted(samples)
    def pct(p):
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p / 100))]
    return pct(50), pct(95)
