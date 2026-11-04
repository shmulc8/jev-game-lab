"""Snake move policies: code baselines and the Jev-backed policy."""

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import client
import questions


def new_stats():
    return {
        "requests": 0,
        "skipped_calls": 0,
        "input_tokens": 0,
        "cost_usd": 0.0,
        "latencies_ms": [],
        "confidences": [],
        "confidence_fallbacks": 0,
        "invalid_choices": 0,
        "speculative_requests": 0,
        "pipeline_hits": 0,
        "pipeline_misses": 0,
        "wasted_requests": 0,
        "budget_misses": 0,
        "blocked_ms": [],
    }


def flood_choice(game):
    moves = game.legal_moves()
    scored = []
    for move in moves:
        if move["fatal"]:
            area, dist = 0, float("inf")
        else:
            obstacles = set(move["body_after"]) - {move["new_head"]}
            area = game.flood_fill_count(move["new_head"], obstacles)
            dist = game.bfs_distance(move["new_head"], obstacles)
            dist = dist if dist is not None else float("inf")
        scored.append((move, area, dist))
    safe = [s for s in scored if s[1] >= len(game.snake)]
    best = min(safe, key=lambda s: s[2]) if safe else max(scored, key=lambda s: s[1])
    return best[0]["direction"]


class RandomPolicy:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.stats = new_stats()

    def choose(self, game):
        moves = game.survivable_moves() or game.legal_moves()
        return self.rng.choice(moves)["direction"]


class GreedyPolicy:
    def __init__(self):
        self.stats = new_stats()

    def choose(self, game):
        best_move, best_dist = None, None
        for move in game.legal_moves():
            if move["fatal"]:
                dist = float("inf")
            else:
                obstacles = set(move["body_after"]) - {move["new_head"]}
                dist = game.bfs_distance(move["new_head"], obstacles)
                dist = dist if dist is not None else float("inf")
            if best_dist is None or dist < best_dist:
                best_move, best_dist = move, dist
        return best_move["direction"]


class FloodPolicy:
    def __init__(self):
        self.stats = new_stats()

    def choose(self, game):
        return flood_choice(game)


def build_state(game):
    return {
        "grid_size": game.grid,
        "current_direction": game.direction,
        "length": len(game.snake),
        "head": list(game.head),
        "body": [list(p) for p in game.snake],
        "food": list(game.food) if game.food is not None else None,
        "walls": sorted(list(w) for w in game.walls),
        "ascii_grid": game.render(),
        "ascii_legend": "@ head, # body, * food, X wall (fatal), . empty",
    }


def describe_candidate(game, move, mode):
    direction = move["direction"]
    x, y = move["new_head"]
    if mode == "features":
        feats = game.move_features(direction)
        return questions.feature_candidate_criteria(direction, x, y, feats)
    return questions.raw_candidate_criteria(direction, x, y)


def build_direction_questions(game, moves, mode):
    candidates = {m["direction"]: describe_candidate(game, m, mode) for m in moves}
    return {
        "direction": {
            "type": "choice",
            "instructions": questions.DIRECTION_INSTRUCTIONS,
            "criteria": candidates,
        },
    }


def build_composite_questions(moves):
    built = {}
    for move in moves:
        d = move["direction"]
        x, y = move["new_head"]
        built[f"room_{d}"] = {
            "type": "score",
            "instructions": questions.room_instructions(d, x, y),
            "criteria": questions.ROOM_LEVELS,
        }
        built[f"food_{d}"] = {
            "type": "score",
            "instructions": questions.food_instructions(d, x, y),
            "criteria": questions.FOOD_LEVELS,
        }
    return built


def weighted_value(candidate, weight_room, weight_food):
    room_norm = candidate["room_score"] / (candidate["room_levels"] - 1)
    food_norm = candidate["food_score"] / (candidate["food_levels"] - 1)
    return weight_room * room_norm + weight_food * food_norm


class JevPolicy:
    def __init__(
        self,
        mode="raw",
        min_confidence=questions.DEFAULT_MIN_CONFIDENCE,
        weight_room=questions.DEFAULT_WEIGHT_ROOM,
        weight_food=questions.DEFAULT_WEIGHT_FOOD,
        trace_path=None,
        model="jev-latest",
        pipeline=False,
    ):
        self.mode = mode
        self.min_confidence = min_confidence
        self.weight_room = weight_room
        self.weight_food = weight_food
        self.model = model
        self.pipeline = bool(pipeline and mode in ("raw", "features"))
        self.stats = new_stats()
        self._executor = ThreadPoolExecutor(max_workers=4) if self.pipeline else None
        self._budget_executor = None
        self._futures = {}
        self._future_meta = {}
        self._future_lock = threading.RLock()
        self.last_decision = {"source": "jev", "blocked_ms": 0.0}
        self.last_payload = None
        self.last_response = None
        self.last_api_latency = 0.0
        self.trace_path = trace_path
        self._trace_file = None
        if trace_path and mode == "composite":
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            self._trace_file = open(trace_path, "w")

    def start_trace(self, game, steps_limit, seed):
        if not self._trace_file:
            return
        meta = {
            "type": "meta",
            "seed": seed,
            "grid": game.grid,
            "walls": sorted([list(cell) for cell in game.walls]),
            "initial_food": game.food,
            "rng_state": game.rng.getstate(),
            "steps_limit": steps_limit,
            "weight_room": self.weight_room,
            "weight_food": self.weight_food,
        }
        self._trace_file.write(json.dumps(meta) + "\n")
        self._trace_file.flush()

    def finish_trace(self, game):
        if not self._trace_file:
            return
        summary = {
            "type": "summary",
            "food_eaten": game.food_eaten,
            "steps": game.steps,
            "death_cause": game.death_cause,
        }
        self._trace_file.write(json.dumps(summary) + "\n")
        self._trace_file.flush()
        self._trace_file.close()

    def _record_usage(self, payload, elapsed_ms):
        self.stats["latencies_ms"].append(elapsed_ms)
        tokens = payload.get("usage", {}).get("input_tokens", 0)
        self.stats["input_tokens"] += tokens
        self.stats["cost_usd"] += tokens * client.INPUT_TOKEN_PRICE_USD

    @staticmethod
    def state_key(game):
        return (tuple(game.snake), game.food, game.direction)

    def _request(self, payload):
        return client.ask(payload["state"], payload["questions"], model=payload["model"])

    def _on_future_done(self, future):
        with self._future_lock:
            meta = self._future_meta.get(future)
        try:
            payload, elapsed_ms = future.result()
        except Exception:
            return
        with self._future_lock:
            if meta and not meta["recorded"]:
                meta["recorded"] = True
                self._record_usage(payload, elapsed_ms)
                if meta["discarded"]:
                    self.stats["wasted_requests"] += 1

    def _submit(self, game, speculative=False):
        payload = self.build_request(game)
        future = self._executor.submit(self._request, payload)
        with self._future_lock:
            self._futures[self.state_key(game)] = future
            self._future_meta[future] = {"recorded": False, "discarded": False, "payload": payload}
            self.stats["requests"] += 1
            if speculative:
                self.stats["speculative_requests"] += 1
        future.add_done_callback(self._on_future_done)
        return future

    def _submit_budget(self, game):
        if self._budget_executor is None:
            self._budget_executor = ThreadPoolExecutor(max_workers=4)
        payload = self.build_request(game)
        future = self._budget_executor.submit(self._request, payload)
        with self._future_lock:
            self._future_meta[future] = {"recorded": False, "discarded": False, "payload": payload}
            self.stats["requests"] += 1
        future.add_done_callback(self._on_future_done)
        return future

    def _discard(self, future):
        with self._future_lock:
            meta = self._future_meta.get(future)
            if meta:
                meta["discarded"] = True
                if future.done() and meta["recorded"]:
                    self.stats["wasted_requests"] += 1

    def _evict(self, keep_keys):
        with self._future_lock:
            for key, future in list(self._futures.items()):
                if key not in keep_keys:
                    self._futures.pop(key, None)
                    self._discard(future)

    def _pipeline_children(self, game):
        children = {}
        for move in game.survivable_moves():
            simulated = game.clone()
            simulated.step(move["direction"])
            if len(simulated.survivable_moves()) >= 2:
                children[self.state_key(simulated)] = simulated
        return children

    def _prepare_pipeline(self, game, preserve_keys=()):
        children = self._pipeline_children(game)
        self._evict(set(children) | set(preserve_keys))
        with self._future_lock:
            available = len(self._futures)
        for child in children.values():
            if available >= 6:
                break
            key = self.state_key(child)
            with self._future_lock:
                if key in self._futures:
                    continue
            self._submit(child, speculative=True)
            available += 1

    def _future_for(self, game):
        with self._future_lock:
            return self._futures.pop(self.state_key(game), None)

    def _answer_future(self, game, future, deadline_s, started):
        remaining = None if deadline_s is None else max(0.0, deadline_s - (time.perf_counter() - started))
        try:
            payload, elapsed_ms = future.result(timeout=remaining)
        except Exception as exc:
            if deadline_s is not None and not future.done():
                self.stats["budget_misses"] += 1
                self._discard(future)
                self.last_decision["source"] = "fallback-timeout"
                return flood_choice(game)
            raise exc
        self.last_decision["source"] = "jev-pipelined" if self.pipeline else "jev"
        with self._future_lock:
            meta = self._future_meta.get(future, {})
        self.last_payload = meta.get("payload")
        self.last_response = payload
        self.last_api_latency = elapsed_ms
        return payload

    def choose(self, game, deadline_s=None):
        if self.mode == "composite" and deadline_s is not None:
            raise ValueError("composite mode does not support a tick budget")
        started = time.perf_counter()
        self.last_decision = {"source": "jev", "blocked_ms": 0.0}
        self.last_payload = None
        self.last_response = None
        moves = game.survivable_moves()
        if len(moves) <= 1:
            self.stats["skipped_calls"] += 1
            result = moves[0]["direction"] if moves else flood_choice(game)
        elif self.pipeline:
            future = self._future_for(game)
            if future is None:
                self.stats["pipeline_misses"] += 1
                future = self._submit(game)
                self._prepare_pipeline(game, preserve_keys=(self.state_key(game),))
            else:
                self.stats["pipeline_hits"] += 1
            payload = self._answer_future(game, future, deadline_s, started)
            result = payload if isinstance(payload, str) else self._choose_direction(game, moves, payload)
            self._prepare_pipeline(game)
        elif deadline_s is not None:
            submitted = self._submit_budget(game)
            try:
                payload, _ = submitted.result(timeout=max(0.0, deadline_s))
                result = self._choose_direction(game, moves, payload)
            except Exception:
                if not submitted.done():
                    self.stats["budget_misses"] += 1
                    self._discard(submitted)
                    result = flood_choice(game)
                else:
                    raise
        elif self.mode == "composite":
            result = self._choose_composite(game, moves)
        else:
            result = self._choose_direction(game, moves)
        self.last_decision["blocked_ms"] = (time.perf_counter() - started) * 1000
        self.stats["blocked_ms"].append(self.last_decision["blocked_ms"])
        return result

    def build_request(self, game):
        """The payload `choose` would send for this game state, without calling the API."""
        moves = game.survivable_moves()
        if len(moves) <= 1:
            raise ValueError("no request would be sent: at most one survivable move")
        state = build_state(game)
        if self.mode == "composite":
            question_set = build_composite_questions(moves)
        else:
            question_set = build_direction_questions(game, moves, self.mode)
        return {"state": state, "model": self.model, "questions": question_set}

    def _choose_direction(self, game, moves, payload=None):
        state = build_state(game)
        question_set = build_direction_questions(game, moves, self.mode)
        if payload is None:
            self.stats["requests"] += 1
            payload, elapsed_ms = client.ask(state, question_set, model=self.model)
            self._record_usage(payload, elapsed_ms)
            self.last_api_latency = elapsed_ms
        self.last_payload = {"state": state, "model": self.model, "questions": question_set}
        self.last_response = payload
        answers = payload["answers"]

        choice_answer = answers["direction"]
        confidence = choice_answer.get("confidence")
        if confidence is not None:
            self.stats["confidences"].append(confidence)
            if confidence < self.min_confidence:
                self.stats["confidence_fallbacks"] += 1
                return flood_choice(game)

        chosen = choice_answer.get("choice")
        valid_keys = {m["direction"] for m in moves}
        if chosen not in valid_keys:
            self.stats["invalid_choices"] += 1
            return flood_choice(game)
        return chosen

    def _choose_composite(self, game, moves):
        state = build_state(game)
        question_set = build_composite_questions(moves)
        self.stats["requests"] += 1
        payload, elapsed_ms = client.ask(state, question_set, model=self.model)
        self._record_usage(payload, elapsed_ms)
        answers = payload["answers"]

        candidates = {}
        for move in moves:
            d = move["direction"]
            room = answers[f"room_{d}"]
            food = answers[f"food_{d}"]
            candidates[d] = {
                "room_score": room["score"],
                "room_levels": len(questions.ROOM_LEVELS),
                "room_confidence": room.get("confidence"),
                "food_score": food["score"],
                "food_levels": len(questions.FOOD_LEVELS),
                "food_confidence": food.get("confidence"),
            }
            for conf in (room.get("confidence"), food.get("confidence")):
                if conf is not None:
                    self.stats["confidences"].append(conf)

        best_direction = max(
            candidates, key=lambda d: weighted_value(candidates[d], self.weight_room, self.weight_food)
        )
        confidence_fallback = False
        avg_conf = None
        winner = candidates[best_direction]
        confs = [c for c in (winner["room_confidence"], winner["food_confidence"]) if c is not None]
        if confs:
            avg_conf = sum(confs) / len(confs)
            if avg_conf < self.min_confidence:
                confidence_fallback = True
                self.stats["confidence_fallbacks"] += 1

        final_direction = flood_choice(game) if confidence_fallback else best_direction
        self._write_trace(game, candidates, final_direction, confidence_fallback)
        return final_direction

    def close(self, wait=True):
        if self._executor:
            self._executor.shutdown(wait=wait, cancel_futures=False)
        if self._budget_executor:
            self._budget_executor.shutdown(wait=wait, cancel_futures=False)

    def _write_trace(self, game, candidates, chosen, confidence_fallback):
        if not self._trace_file:
            return
        record = {
            "tick": game.steps,
            "weight_room": self.weight_room,
            "weight_food": self.weight_food,
            "candidates": candidates,
            "chosen": chosen,
            "confidence_fallback": confidence_fallback,
        }
        self._trace_file.write(json.dumps(record) + "\n")
        self._trace_file.flush()
