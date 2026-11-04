"""Decision policies for Tetris: random, heuristic, oracle, and Jev (TypeSafe API)."""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
import time
import urllib.error
import urllib.request
import http.client

import jevhttp

import questions
import tetris

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000
MAX_ATTEMPTS = 3
TOP_REGION_MARGIN = 2

# How many upcoming pieces to surface in rich-mode state's `next_pieces` queue.
# tetris.PieceBag.peek(ahead) can look arbitrarily far ahead (it lazily refills
# the 7-bag), so this is a presentation choice, not an engine limit.
NEXT_QUEUE_LEN = 3

# Shared afterstate-value formula: used both as the score rich mode hands Jev's
# per-candidate lookahead, and as the oracle agent's argmax objective.
AFTERSTATE_LINES_WEIGHT = 8.0
AFTERSTATE_HOLES_WEIGHT = 5.0
AFTERSTATE_AGG_HEIGHT_WEIGHT = 0.35
AFTERSTATE_MAX_HEIGHT_WEIGHT = 0.4
AFTERSTATE_BUMPINESS_WEIGHT = 0.5
ORACLE_LOOKAHEAD_WEIGHT = 0.25
# Penalty used in place of a lookahead value when the next piece has no legal
# placement at all on a candidate's resulting board (that candidate would hand
# off a dead position).
TOPOUT_LOOKAHEAD_PENALTY = -1000.0


def _new_stats():
    return {
        "requests": 0,
        "input_tokens": 0,
        "cost_usd": 0.0,
        "invalid_choices": 0,
        "confidences": [],
        "topout_scores": [],
        "capped": False,
        "speculative_requests": 0,
        "pipeline_hits": 0,
        "pipeline_misses": 0,
        "wasted_requests": 0,
        "blocked_ms": [],
        "latencies_ms": [],
    }


def heuristic_score(sim_board, lines_cleared):
    return (
        -0.51 * sim_board.aggregate_height()
        + 0.76 * lines_cleared
        - 0.36 * sim_board.hole_count()
        - 0.18 * sim_board.bumpiness()
    )


def best_heuristic_placement(board, piece, placements=None):
    placements = placements if placements is not None else board.legal_placements(piece)
    best = None
    best_score = None
    for rot, col in placements:
        sim_board, lines_cleared = board.simulate(piece, rot, col)
        score = heuristic_score(sim_board, lines_cleared)
        if best_score is None or score > best_score:
            best_score = score
            best = (rot, col)
    return best


def afterstate_value(lines_cleared, holes, aggregate_height, max_height, bumpiness):
    """lines_cleared * 8.0 - holes * 5.0 - aggregate_height * 0.35 - max_height * 0.4 - bumpiness * 0.5"""
    return (
        lines_cleared * AFTERSTATE_LINES_WEIGHT
        - holes * AFTERSTATE_HOLES_WEIGHT
        - aggregate_height * AFTERSTATE_AGG_HEIGHT_WEIGHT
        - max_height * AFTERSTATE_MAX_HEIGHT_WEIGHT
        - bumpiness * AFTERSTATE_BUMPINESS_WEIGHT
    )


def one_piece_lookahead(board_after, next_piece):
    """Best afterstate_value achievable by placing `next_piece` on `board_after`.

    Returns None if `next_piece` has no legal placement on `board_after` (i.e. that
    resulting board would already be a dead end for the next piece).
    """
    if next_piece is None:
        return None
    best = None
    for rot, col in board_after.legal_placements(next_piece):
        sim_board, lines_cleared = board_after.simulate(next_piece, rot, col)
        value = afterstate_value(
            lines_cleared,
            sim_board.hole_count(),
            sim_board.aggregate_height(),
            sim_board.max_height(),
            sim_board.bumpiness(),
        )
        if best is None or value > best:
            best = value
    return best


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.stats = {"requests": 0, "input_tokens": 0, "cost_usd": 0.0}

    def choose(self, board, piece, next_piece, **kwargs):
        placements = board.legal_placements(piece)
        return self.rng.choice(placements)


class HeuristicAgent:
    def __init__(self):
        self.stats = {"requests": 0, "input_tokens": 0, "cost_usd": 0.0}

    def choose(self, board, piece, next_piece, **kwargs):
        return best_heuristic_placement(board, piece)


class OracleAgent:
    """Pure-code control, no API calls. Picks argmax of:

        afterstate_value(candidate) + 0.25 * one_piece_lookahead(candidate)

    This is the exact formula whose component numbers (lines cleared, holes,
    aggregate height, max height, bumpiness, and the one-piece lookahead score)
    rich mode hands to Jev in words. It answers: how close does Jev get when
    the ingredients of the answer are handed to it directly?
    """

    def __init__(self):
        self.stats = {"requests": 0, "input_tokens": 0, "cost_usd": 0.0}

    def choose(self, board, piece, next_piece, next_pieces=None, lines_cleared_total=0, **kwargs):
        placements = board.legal_placements(piece)
        best = None
        best_score = None
        for rot, col in placements:
            sim_board, lines_cleared = board.simulate(piece, rot, col)
            value = afterstate_value(
                lines_cleared,
                sim_board.hole_count(),
                sim_board.aggregate_height(),
                sim_board.max_height(),
                sim_board.bumpiness(),
            )
            lookahead = one_piece_lookahead(sim_board, next_piece)
            if lookahead is None:
                lookahead = TOPOUT_LOOKAHEAD_PENALTY
            score = value + ORACLE_LOOKAHEAD_WEIGHT * lookahead
            if best_score is None or score > best_score:
                best_score = score
                best = (rot, col)
        return best


class JevAPIError(Exception):
    def __init__(self, status, body):
        self.status = status
        self.body = body
        super().__init__(f"TypeSafe API error {status}: {body}")


class JevValidationError(JevAPIError):
    pass


def _load_api_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    raise RuntimeError("Set TYPESAFE_API_KEY in the environment to use the JEV agent")


def _post(payload, api_key):
    try:
        status, raw = jevhttp.post_json(JEV_URL, payload, api_key, timeout=60)
    except (OSError, http.client.HTTPException) as e:
        return 0, {"error": f"network error: {e}"}
    try:
        body = json.loads(raw)
    except ValueError:
        body = raw.decode(errors="replace")
    return status, body


def call_jev(payload, api_key):
    status, body = None, None
    for attempt in range(MAX_ATTEMPTS):
        status, body = _post(payload, api_key)
        if status == 200:
            return body
        if status == 422:
            raise JevValidationError(status, body)
        if status in (0, 429, 529):
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
            continue
        raise JevAPIError(status, body)
    raise JevAPIError(status, body)


def _placement_width(piece, rot):
    shape = tetris.PIECES[piece][rot]
    return max(c for c, r in shape) + 1


def _board_mode_description(board, piece, rot, col):
    width = _placement_width(piece, rot)
    resting_row = board.placement_bottom_row(piece, rot, col)
    return questions.board_mode_criteria(rot, col, width, resting_row)


def _bottom_region(board, rows):
    lines = ["".join("#" if cell else "." for cell in row) for row in board.grid[-rows:]]
    footer = "".join(str(c % 10) for c in range(tetris.WIDTH))
    return "\n".join(lines + [footer])


def _preview_mode_description(board, piece, rot, col, top_rows):
    sim_board, lines_cleared = board.simulate(piece, rot, col)
    ascii_region = _bottom_region(sim_board, top_rows)
    return questions.preview_mode_criteria(top_rows, ascii_region, lines_cleared)


class JevAgent:
    def __init__(self, mode="preview", api_key=None, payload="top12", pipeline=False):
        if mode not in ("board", "preview", "rich"):
            raise ValueError("mode must be 'board', 'preview', or 'rich'")
        if payload not in ("full", "nostate", "top12", "both"):
            raise ValueError("payload must be 'full', 'nostate', 'top12', or 'both'")
        self.mode = mode
        self.api_key = api_key or _load_api_key()
        self.payload = payload
        self.pipeline = bool(pipeline and mode == "rich")
        self.stats = _new_stats()
        self._pipeline_executor = ThreadPoolExecutor(max_workers=4) if self.pipeline else None
        self._pipeline_futures = {}
        self._pipeline_orphans = []

    def close(self):
        """Finish speculative work so its usage is included in final stats."""
        if self._pipeline_executor is None:
            return
        for key, entry in list(self._pipeline_futures.items()):
            self.stats["wasted_requests"] += 1
            self._discard_pipeline_entry(entry)
            self._pipeline_futures.pop(key, None)
        self._pipeline_executor.shutdown(wait=True)
        for entry in self._pipeline_orphans:
            self._finish_pipeline_entry(entry)
        self._pipeline_orphans.clear()
        self._pipeline_executor = None

    def _finish_pipeline_entry(self, entry):
        try:
            response = entry["future"].result()
        except Exception:
            return None
        self._record_response(response)
        return response

    def _discard_pipeline_entry(self, entry):
        future = entry["future"]
        if future.done():
            self._finish_pipeline_entry(entry)
        elif not future.cancel():
            self._pipeline_orphans.append(entry)

    def _select_candidates(self, board, piece, next_piece=None):
        placements = board.legal_placements(piece)
        if self.mode == "rich" and self.payload in ("top12", "both"):
            return self._oracle_cap(board, piece, next_piece, placements)
        if not self.stats["capped"] or len(placements) <= questions.CAP_CANDIDATES:
            return placements
        return self._cap(board, piece, placements)

    def _oracle_cap(self, board, piece, next_piece, placements):
        scored = []
        for rot, col in placements:
            sim_board, lines_cleared = board.simulate(piece, rot, col)
            value = afterstate_value(
                lines_cleared,
                sim_board.hole_count(),
                sim_board.aggregate_height(),
                sim_board.max_height(),
                sim_board.bumpiness(),
            )
            lookahead = one_piece_lookahead(sim_board, next_piece)
            if lookahead is None:
                lookahead = TOPOUT_LOOKAHEAD_PENALTY
            scored.append((value + ORACLE_LOOKAHEAD_WEIGHT * lookahead, rot, col))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [(rot, col) for _, rot, col in scored[:questions.CAP_CANDIDATES]]

    def _cap(self, board, piece, placements):
        scored = []
        for rot, col in placements:
            sim_board, _ = board.simulate(piece, rot, col)
            scored.append((sim_board.max_height(), rot, col))
        scored.sort(key=lambda t: t[0])
        return [(rot, col) for _, rot, col in scored[:questions.CAP_CANDIDATES]]

    def _rich_candidates(self, board, piece, next_piece, candidates):
        infos = []
        for rot, col in candidates:
            sim_board, lines_cleared = board.simulate(piece, rot, col)
            cells = board.placement_cells(piece, rot, col)
            occupied_cells = [[c, r] for c, r in sorted(cells, key=lambda cr: (cr[1], cr[0]))]
            lookahead = one_piece_lookahead(sim_board, next_piece)
            infos.append({
                "id": f"r{rot}c{col}",
                "rotation": rot,
                "column": col,
                "landing_row": board.placement_bottom_row(piece, rot, col),
                "occupied_cells": occupied_cells,
                "lines_cleared": lines_cleared,
                "afterstate": {
                    "holes": sim_board.hole_count(),
                    "max_height": sim_board.max_height(),
                    "aggregate_height": sim_board.aggregate_height(),
                    "bumpiness": sim_board.bumpiness(),
                },
                "lookahead": lookahead,
            })
        return infos

    def _rich_state(self, board, piece, next_piece, next_pieces, lines_cleared_total, candidates):
        infos = self._rich_candidates(board, piece, next_piece, candidates)
        width0 = _placement_width(piece, 0)
        spawn_col = max(0, (tetris.WIDTH - width0) // 2)
        active_cells = [[spawn_col + c, r] for c, r in tetris.PIECES[piece][0]]
        queue = list(next_pieces) if next_pieces else ([next_piece] if next_piece else [])
        state = {
            "coordinate_system": questions.RICH_COORDINATE_SYSTEM,
            "board": ["".join("#" if cell else "." for cell in row) for row in board.grid],
            "active_piece": {
                "kind": piece,
                "rotation": 0,
                "occupied_cells": active_cells,
            },
            "next_pieces": queue,
            "lines": lines_cleared_total,
            "allowed_placements": [info["id"] for info in infos],
        }
        if self.payload not in ("nostate", "both"):
            state["legal_placements"] = [
                {k: v for k, v in info.items() if k != "lookahead"} for info in infos
            ]
        if self.payload in ("top12", "both"):
            state["legal_placements_note"] = "The candidate list was pre-filtered by code to the 12 best oracle-scored placements."
        return state

    def _build_state(self, board, piece, next_piece, next_pieces=None, lines_cleared_total=0, candidates=None):
        if self.mode == "rich":
            return self._rich_state(board, piece, next_piece, next_pieces, lines_cleared_total, candidates or [])
        return {
            "board": board.render(),
            "current_piece": tetris.shape_ascii(tetris.PIECES[piece][0]),
            "next_piece": tetris.shape_ascii(tetris.PIECES[next_piece][0]),
        }

    def _build_questions(self, board, piece, next_piece, candidates):
        if self.mode == "board":
            criteria = {
                f"r{rot}c{col}": _board_mode_description(board, piece, rot, col)
                for rot, col in candidates
            }
            instructions = questions.BASIC_PLACEMENT_INSTRUCTIONS
        elif self.mode == "preview":
            top_rows = min(
                tetris.HEIGHT,
                max(
                    (board.simulate(piece, rot, col)[0].max_height() for rot, col in candidates),
                    default=0,
                )
                + TOP_REGION_MARGIN,
            )
            top_rows = max(top_rows, 4)
            criteria = {
                f"r{rot}c{col}": _preview_mode_description(board, piece, rot, col, top_rows)
                for rot, col in candidates
            }
            instructions = questions.BASIC_PLACEMENT_INSTRUCTIONS
        else:  # rich
            infos = self._rich_candidates(board, piece, next_piece, candidates)
            criteria = {info["id"]: questions.rich_candidate_criteria(info) for info in infos}
            instructions = questions.RICH_PLACEMENT_INSTRUCTIONS
        placement_question = {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
        topout_question = {
            "type": "score",
            "instructions": questions.TOPOUT_INSTRUCTIONS,
            "criteria": questions.TOPOUT_SCORE_LEVELS,
        }
        return {"placement": placement_question, "topout": topout_question}

    def _build_payload(self, board, piece, next_piece, next_pieces, lines_cleared_total, candidates):
        return {
            "state": self._build_state(board, piece, next_piece, next_pieces, lines_cleared_total, candidates),
            "model": JEV_MODEL,
            "questions": self._build_questions(board, piece, next_piece, candidates),
        }

    def _request_once(self, board, piece, next_piece, next_pieces, lines_cleared_total, candidates):
        payload = self._build_payload(board, piece, next_piece, next_pieces, lines_cleared_total, candidates)
        started = time.perf_counter()
        try:
            return call_jev(payload, self.api_key)
        finally:
            self.stats["latencies_ms"].append((time.perf_counter() - started) * 1000)

    def _record_response(self, response):
        self.stats["requests"] += 1
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        self.stats["input_tokens"] += input_tokens
        self.stats["cost_usd"] += input_tokens * JEV_PRICE_PER_INPUT_TOKEN

        answers = response["answers"]
        placement_answer = answers["placement"]
        topout_answer = answers["topout"]
        self.stats["topout_scores"].append(topout_answer.get("score"))
        confidence = placement_answer.get("confidence")
        if confidence is not None:
            self.stats["confidences"].append(confidence)

    def build_request(self, board, piece, next_piece, next_pieces=None, lines_cleared_total=0):
        """The payload `choose` would send for this position, without calling the API."""
        candidates = self._select_candidates(board, piece, next_piece)
        return self._build_payload(board, piece, next_piece, next_pieces, lines_cleared_total, candidates)

    def choose(self, board, piece, next_piece, next_pieces=None, lines_cleared_total=0):
        placements = board.legal_placements(piece)
        candidates = self._select_candidates(board, piece, next_piece)
        response = None
        hit_key = None
        if self.pipeline:
            hit_key = (tuple(tuple(row) for row in board.grid), piece, lines_cleared_total)
            for key in list(self._pipeline_futures):
                if key != hit_key:
                    self.stats["wasted_requests"] += 1
                    self._discard_pipeline_entry(self._pipeline_futures.pop(key))
            entry = self._pipeline_futures.pop(hit_key, None)
            if entry is not None and entry["next_piece"] != next_piece:
                self.stats["wasted_requests"] += 1
                self._discard_pipeline_entry(entry)
                entry = None
            if entry is not None:
                self.stats["pipeline_hits"] += 1
                started = time.perf_counter()
                response = self._finish_pipeline_entry(entry)
                self.stats["blocked_ms"].append((time.perf_counter() - started) * 1000)
            else:
                self.stats["pipeline_misses"] += 1

            current_future = None
            if response is None:
                current_future = self._pipeline_executor.submit(
                    self._request_once, board, piece, next_piece, next_pieces,
                    lines_cleared_total, candidates,
                )
            self._submit_speculative(
                board, piece, next_piece, next_pieces, lines_cleared_total, candidates,
                reserved=1 if response is None else 0,
            )

        if response is None:
            started = time.perf_counter()
            try:
                if self.pipeline:
                    response = current_future.result()
                else:
                    response = self._request_once(board, piece, next_piece, next_pieces, lines_cleared_total, candidates)
            except JevValidationError as e:
                self.stats["requests"] += 1
                if self.stats["capped"] or len(placements) <= questions.CAP_CANDIDATES:
                    raise
                self.stats["capped"] = True
                candidates = self._cap(board, piece, placements)
                response = self._request_once(board, piece, next_piece, next_pieces, lines_cleared_total, candidates)
            finally:
                self.stats["blocked_ms"].append((time.perf_counter() - started) * 1000) if self.pipeline else None
            self._record_response(response)

        placement_answer = response["answers"]["placement"]
        choice_key = placement_answer.get("choice")
        candidate_map = {f"r{rot}c{col}": (rot, col) for rot, col in candidates}
        if choice_key in candidate_map:
            return candidate_map[choice_key]

        self.stats["invalid_choices"] += 1
        return best_heuristic_placement(board, piece, placements)

    def _submit_speculative(self, board, piece, next_piece, next_pieces, lines_cleared_total, candidates, reserved=0):
        if not next_piece or self._pipeline_executor is None:
            return
        ranked = []
        for rot, col in candidates:
            sim_board, lines_cleared = board.simulate(piece, rot, col)
            score = afterstate_value(
                lines_cleared,
                sim_board.hole_count(),
                sim_board.aggregate_height(),
                sim_board.max_height(),
                sim_board.bumpiness(),
            )
            lookahead = one_piece_lookahead(sim_board, next_piece)
            if lookahead is None:
                lookahead = TOPOUT_LOOKAHEAD_PENALTY
            ranked.append((score + ORACLE_LOOKAHEAD_WEIGHT * lookahead, rot, col, sim_board, lines_cleared))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        next_next = (list(next_pieces)[1:] if next_pieces else [])
        occupied = sum(not entry["future"].done() for entry in self._pipeline_futures.values())
        occupied += sum(not entry["future"].done() for entry in self._pipeline_orphans)
        slots = max(0, 6 - occupied - reserved)
        for _, rot, col, sim_board, lines_cleared in ranked[:min(3, slots)]:
            key = (tuple(tuple(row) for row in sim_board.grid), next_piece, lines_cleared_total + lines_cleared)
            if key in self._pipeline_futures:
                continue
            future = self._pipeline_executor.submit(
                self._request_once, sim_board, next_piece,
                next_next[0] if next_next else None, next_next,
                lines_cleared_total + lines_cleared,
                self._select_candidates(sim_board, next_piece, next_next[0] if next_next else None),
            )
            self._pipeline_futures[key] = {"future": future, "next_piece": next_next[0] if next_next else None}
            self.stats["speculative_requests"] += 1
