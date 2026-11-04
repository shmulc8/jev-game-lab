"""Frame-producing adapters for the existing Tetris and Snake experiments."""
import json
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SNAKE = os.path.join(ROOT, "snake")
if SNAKE not in sys.path:
    sys.path.append(SNAKE)

import agents
import tetris
from snake import client as snake_client
from snake import engine as snake_engine
_tetris_questions = sys.modules.get("questions")
if _tetris_questions is not None:
    del sys.modules["questions"]
_root_index = sys.path.index(ROOT)
sys.path.insert(0, SNAKE)
from snake import policies as snake_policies
sys.path.pop(0)
if _tetris_questions is not None:
    sys.modules["questions"] = _tetris_questions

PRICE = getattr(agents, "JEV_PRICE_PER_INPUT_TOKEN", snake_client.INPUT_TOKEN_PRICE_USD)


def approx_tokens(value):
    return len(json.dumps(value, separators=(",", ":"))) // 4


def pct(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    x = (len(values) - 1) * p / 100
    lo, hi = int(x), min(int(x) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


class Capture:
    def __init__(self, game):
        self.game = game
        self.payload = None
        self.response = None
        self.latency_ms = None
        self._real = None

    def install(self):
        if self.game == "tetris":
            self._real = agents.call_jev
            def wrapped(payload, api_key):
                started = time.perf_counter()
                response = self._real(payload, api_key)
                self.payload, self.response = payload, response
                self.latency_ms = (time.perf_counter() - started) * 1000
                return response
            agents.call_jev = wrapped
        else:
            self._target = snake_policies.client
            self._real = self._target.ask
            def wrapped(state, questions, model="jev-latest", max_attempts=3):
                started = time.perf_counter()
                response, latency = self._real(state, questions, model=model, max_attempts=max_attempts)
                self.payload = {"state": state, "model": model, "questions": questions}
                self.response, self.latency_ms = response, (time.perf_counter() - started) * 1000
                return response, latency
            self._target.ask = wrapped

    def clear(self):
        self.payload = self.response = self.latency_ms = None

    def restore(self):
        if self.game == "tetris" and self._real:
            agents.call_jev = self._real
        elif self.game == "snake" and self._real:
            self._target.ask = self._real


def _stats_summary(stats, elapsed):
    lat = stats.get("latencies_ms", [])
    requests = stats.get("requests", 0)
    conf = stats.get("confidences", [])
    return {
        "moves": elapsed, "lines_or_food": 0, "requests": requests,
        "skipped_calls": stats.get("skipped_calls", 0),
        "p50_latency_ms": round(pct(lat, 50), 2), "p95_latency_ms": round(pct(lat, 95), 2),
        "decisions_per_sec": round(1000 * elapsed / sum(lat), 3) if lat and sum(lat) else 0,
        "tokens": stats.get("input_tokens", 0), "cost": round(stats.get("cost_usd", 0.0), 8),
        "mean_confidence": round(statistics.mean(conf), 4) if conf else None,
        "invalid_choices": stats.get("invalid_choices", 0),
        "speculative_requests": stats.get("speculative_requests", 0),
        "pipeline_hits": stats.get("pipeline_hits", 0),
        "pipeline_misses": stats.get("pipeline_misses", 0),
        "wasted_requests": stats.get("wasted_requests", 0),
        "budget_misses": stats.get("budget_misses", 0),
        "blocked_ms_p50": round(pct(stats.get("blocked_ms", []), 50), 2),
        "blocked_ms_p95": round(pct(stats.get("blocked_ms", []), 95), 2),
    }


def _tetris_frame(i, total, piece, queue, mode, placements, chosen, applied, cleared, board, cells, cap, started, lines, agent):
    payload, response = cap.payload, cap.response
    stats = agent.stats
    answer = response.get("answers") if response else None
    chosen_tuple = None
    if isinstance(chosen, str) and chosen.startswith("r") and "c" in chosen:
        try:
            chosen_tuple = tuple(map(int, chosen[1:].split("c")))
        except ValueError:
            pass
    return {
        "index": i - 1, "game": "tetris", "agent": agent.__class__.__name__.replace("Agent", "").lower(), "mode": mode,
        "code_enumerates": {"legal_moves": len(placements), "candidates_sent": len(payload["questions"]["placement"]["criteria"]) if payload else len(placements),
            "question_ids": list(payload["questions"].keys()) if payload else [],
            "state_tokens": approx_tokens(payload["state"]) if payload else 0,
            "criteria_tokens": approx_tokens(payload["questions"]) if payload else 0},
        "request": payload,
        "answer": ({"answers": answer, "latency_ms": cap.latency_ms, "input_tokens": response.get("usage", {}).get("input_tokens", 0),
                    "cost_this_call": response.get("usage", {}).get("input_tokens", 0) * PRICE,
                    "cost_running": stats.get("cost_usd", 0.0), "raw_response": response} if response else None),
        "code_verifies": {"returned_key_in_candidate_set": (chosen in payload["questions"]["placement"]["criteria"] if payload else chosen is None),
            "move_applied": applied is not None, "lines_cleared": cleared,
            "holes": board.hole_count(), "max_height": board.max_height(), "bumpiness": board.bumpiness()},
        "board": {"rows": ["".join("#" if c else "." for c in row) for row in board.grid], "just_placed": [list(x) for x in (cells or [])],
                  "piece": piece, "next": queue},
        "narrative": {"enumerates": f"{len(placements)} legal placements; {len(payload['questions']['placement']['criteria']) if payload else len(placements)} candidates sent.",
            "request": "Exact state and questions sent to Jev." if payload else "Decided entirely by code; no request.",
            "answer": "Jev returned probabilities and speculative topout." if response else "Decided in code, no request.",
            "verifies": f"Applied {applied}; cleared {cleared} line(s); holes {board.hole_count()}, max height {board.max_height()}, bumpiness {board.bumpiness()}."},
        "wall_ms": (time.perf_counter() - started) * 1000,
    }


def run_tetris(agent_name="jev", mode="rich", seed=7, steps=40, frame_delay=0.15, on_frame=None):
    board, bag, lines = tetris.Board(), tetris.PieceBag(seed), 0
    if agent_name == "random": agent = agents.RandomAgent(seed)
    elif agent_name == "heuristic": agent = agents.HeuristicAgent()
    elif agent_name == "oracle": agent = agents.OracleAgent()
    else: agent = agents.JevAgent(mode=mode)
    cap = Capture("tetris"); cap.install(); frames = []; started_at = time.perf_counter()
    closed = False
    try:
        for n in range(1, steps + 1):
            piece = bag.next(); queue = [bag.peek(k) for k in range(1, agents.NEXT_QUEUE_LEN + 1)]
            placements = board.legal_placements(piece)
            if not placements: break
            cap.clear(); t0 = time.perf_counter()
            rot, col = agent.choose(board, piece, queue[0], next_pieces=queue, lines_cleared_total=lines)
            chosen_key = cap.response.get("answers", {}).get("placement", {}).get("choice") if cap.response else None
            cells = board.placement_cells(piece, rot, col); cleared = board.drop(piece, rot, col); lines += cleared
            frame = _tetris_frame(n, steps, piece, queue, mode, placements, chosen_key, (rot, col), cleared, board, cells, cap, t0, lines, agent)
            frames.append(frame)
            if on_frame: on_frame(frame)
            if frame_delay: time.sleep(frame_delay)
        if hasattr(agent, "close"):
            agent.close()
        closed = True
        summary = _stats_summary(agent.stats, len(frames)); lat=[f["answer"]["latency_ms"] for f in frames if f.get("answer")]; summary.update({"p50_latency_ms": round(pct(lat,50),2), "p95_latency_ms": round(pct(lat,95),2), "decisions_per_sec": round(1000*len(frames)/sum(lat),3) if lat else 0, "lines_or_food": lines, "outcome": "topout" if len(frames) < steps else "step_limit"})
        return frames, summary
    finally:
        try:
            if not closed and hasattr(agent, "close"):
                agent.close()
        finally:
            cap.restore()


def _snake_frame(i, game, policy, mode, moves, chosen, before_food, cap, t0):
    payload = getattr(policy, "last_payload", cap.payload)
    response = getattr(policy, "last_response", cap.response)
    stats = policy.stats
    answer = response.get("answers") if response else None
    head = list(game.head); rows = []
    for y in range(game.grid):
        rows.append("".join("@" if (x, y) == game.head else "#" if (x, y) in game.snake else "*" if (x, y) == game.food else "." for x in range(game.grid)))
    conf = answer.get("direction", {}).get("confidence") if answer else None
    feature_moves = [game.move_features(m["direction"]) for m in moves if not m.get("fatal")]
    reachable_area = max([m.get("reachable_cells", 0) for m in feature_moves] or [0])
    return {"index": i, "game": "snake", "agent": policy.__class__.__name__.replace("Policy", "").lower(), "mode": mode,
        "source": getattr(policy, "last_decision", {}).get("source", "jev"),
        "blocked_ms": getattr(policy, "last_decision", {}).get("blocked_ms", (time.perf_counter() - t0) * 1000),
        "code_enumerates": {"legal_moves": len(moves), "candidates_sent": len(payload["questions"].get("direction", {} ).get("criteria", {})) if payload else 0,
            "question_ids": list(payload["questions"].keys()) if payload else [], "state_tokens": approx_tokens(payload["state"]) if payload else 0,
            "criteria_tokens": approx_tokens(payload["questions"]) if payload else 0}, "request": payload,
        "answer": ({"answers": answer, "latency_ms": cap.latency_ms, "input_tokens": response.get("usage", {}).get("input_tokens", 0),
                    "cost_this_call": response.get("usage", {}).get("input_tokens", 0) * snake_client.INPUT_TOKEN_PRICE_USD, "cost_running": stats.get("cost_usd", 0.0), "raw_response": response} if response else None),
        "code_verifies": {"returned_key_in_candidate_set": chosen in {m["direction"] for m in moves} if response else True, "move_applied": True,
            "food_eaten": game.food_eaten - before_food, "length": len(game.snake), "reachable_area": reachable_area},
        "board": {"rows": rows, "head": head, "food": list(game.food) if game.food else None},
        "narrative": {"enumerates": f"{len(moves)} legal moves; {len(payload['questions']) if payload else 0} question groups.", "request": "Exact state and questions sent to Jev." if payload else "Decided entirely by code; no request.",
            "answer": "Jev returned direction probabilities, seal risk and danger." if response else "Decided in code, no request.", "verifies": f"Moved {chosen}; food eaten {game.food_eaten - before_food}; length {len(game.snake)}, reachable area {reachable_area}."},
        "wall_ms": (time.perf_counter() - t0) * 1000,
        "pipeline": {key: stats.get(key, 0) for key in ("pipeline_hits", "pipeline_misses", "wasted_requests", "budget_misses")}}


def run_snake(agent_name="jev", mode="features", seed=7, steps=150, frame_delay=0.15, grid=12, on_frame=None, pipeline=False, tick_ms=0):
    game = snake_engine.SnakeGame(grid=grid, seed=seed)
    if agent_name == "random": policy = snake_policies.RandomPolicy(seed)
    elif agent_name == "greedy": policy = snake_policies.GreedyPolicy()
    elif agent_name == "flood": policy = snake_policies.FloodPolicy()
    else: policy = snake_policies.JevPolicy(mode=mode, pipeline=pipeline)
    policy.tick_ms = tick_ms
    cap = Capture("snake"); cap.install(); frames = []; started_at = time.perf_counter()
    closed = False
    try:
        while game.alive and game.steps < steps:
            moves = game.survivable_moves() if agent_name == "jev" else game.legal_moves(); cap.clear(); before = game.food_eaten; t0 = time.perf_counter()
            chosen = policy.choose(game, deadline_s=tick_ms / 1000.0 if tick_ms else None); game.step(chosen)
            frame = _snake_frame(game.steps - 1, game, policy, mode, moves, chosen, before, cap, t0); frames.append(frame)
            if on_frame: on_frame(frame)
            delay = max(0.0, tick_ms / 1000.0 - frame["blocked_ms"] / 1000.0) if tick_ms else frame_delay
            if delay: time.sleep(delay)
        if hasattr(policy, "close"):
            policy.close()
        closed = True
        summary = _stats_summary(policy.stats, len(frames)); summary.update({"lines_or_food": game.food_eaten, "outcome": game.death_cause or "step_limit", "length": len(game.snake)})
        return frames, summary
    finally:
        try:
            if not closed and hasattr(policy, "close"):
                policy.close()
        finally:
            cap.restore()


def run(game, agent, mode, seed, steps, frame_delay=0.15, on_frame=None, pipeline=False, tick_ms=0):
    return (run_tetris(agent, mode, seed, steps, frame_delay, on_frame=on_frame) if game == "tetris"
            else run_snake(agent, mode, seed, steps, frame_delay, on_frame=on_frame, pipeline=pipeline, tick_ms=tick_ms))
