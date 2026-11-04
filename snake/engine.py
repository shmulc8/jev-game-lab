"""Pure Snake engine: rules, collisions, food spawn, and geometry features. No AI, no network."""

import random
from collections import deque

DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class SnakeGame:
    def __init__(self, grid=12, seed=0, walls=0):
        self.grid = grid
        self.rng = random.Random(seed)
        cx, cy = grid // 2, grid // 2
        self.snake = deque([(cx - 1, cy), (cx, cy)])
        self.direction = "right"
        self.steps = 0
        self.food_eaten = 0
        self.alive = True
        self.death_cause = None
        self.walls = self._spawn_walls(walls)
        self.food = self._spawn_food()

    def _spawn_walls(self, count):
        """Fixed obstacle cells. The starting row stays clear so the first moves are always safe."""
        walls = set()
        hy = self.head[1]
        keep_clear = set(self.snake) | {(x, hy) for x in range(self.grid)}
        max_walls = max(0, min(int(count), self.grid * self.grid // 4))
        attempts = 0
        while len(walls) < max_walls and attempts < max_walls * 50:
            attempts += 1
            cell = (self.rng.randrange(self.grid), self.rng.randrange(self.grid))
            if cell in keep_clear or cell in walls:
                continue
            walls.add(cell)
            if not self._all_connected(walls):
                walls.discard(cell)
        return walls

    def _all_connected(self, walls):
        """True when every non-wall cell is reachable from the head, so food can never be sealed off."""
        total_free = self.grid * self.grid - len(walls)
        seen = {self.head}
        queue = deque([self.head])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS.values():
                nxt = (x + dx, y + dy)
                if nxt in seen or nxt in walls or not self.in_bounds(nxt):
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return len(seen) == total_free

    def clone(self):
        """Return an independent game with exactly the same future randomness."""
        clone = object.__new__(SnakeGame)
        clone.grid = self.grid
        clone.walls = set(self.walls)
        clone.rng = random.Random()
        clone.rng.setstate(self.rng.getstate())
        clone.snake = deque(self.snake)
        clone.direction = self.direction
        clone.steps = self.steps
        clone.food_eaten = self.food_eaten
        clone.alive = self.alive
        clone.death_cause = self.death_cause
        clone.food = self.food
        return clone

    @property
    def head(self):
        return self.snake[-1]

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.grid and 0 <= y < self.grid

    def blocked(self, pos):
        return not self.in_bounds(pos) or pos in self.walls

    def _spawn_food(self):
        occupied = set(self.snake) | self.walls
        total_cells = self.grid * self.grid
        if len(occupied) >= total_cells:
            return None
        for _ in range(total_cells * 4):
            candidate = (self.rng.randrange(self.grid), self.rng.randrange(self.grid))
            if candidate not in occupied:
                return candidate
        free = [
            (x, y)
            for x in range(self.grid)
            for y in range(self.grid)
            if (x, y) not in occupied
        ]
        return self.rng.choice(free) if free else None

    def preview_move(self, direction):
        dx, dy = DIRECTIONS[direction]
        hx, hy = self.head
        new_head = (hx + dx, hy + dy)
        grow = new_head == self.food
        wall_hit = self.blocked(new_head)
        body_after_move = list(self.snake)
        if not grow:
            body_after_move.pop(0)
        self_hit = (not wall_hit) and new_head in body_after_move
        fatal = wall_hit or self_hit
        if not fatal:
            body_after_move.append(new_head)
        return {
            "direction": direction,
            "new_head": new_head,
            "grow": grow,
            "fatal": fatal,
            "cause": "wall" if wall_hit else ("self" if self_hit else None),
            "body_after": body_after_move,
        }

    def legal_moves(self):
        reverse = OPPOSITE[self.direction]
        candidates = [d for d in DIRECTIONS if d != reverse or len(self.snake) == 1]
        return [self.preview_move(d) for d in candidates]

    def survivable_moves(self):
        return [m for m in self.legal_moves() if not m["fatal"]]

    def step(self, direction):
        if not self.alive:
            return
        if len(self.snake) > 1 and direction == OPPOSITE[self.direction]:
            direction = self.direction
        preview = self.preview_move(direction)
        self.direction = direction
        self.steps += 1
        if preview["fatal"]:
            self.alive = False
            self.death_cause = preview["cause"]
            return
        self.snake.append(preview["new_head"])
        if preview["grow"]:
            self.food_eaten += 1
            self.food = self._spawn_food()
            if self.food is None:
                self.alive = False
                self.death_cause = "board_full"
        else:
            self.snake.popleft()

    def bfs_distance(self, start, obstacles):
        if start == self.food:
            return 0
        seen = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            for dx, dy in DIRECTIONS.values():
                nxt = (pos[0] + dx, pos[1] + dy)
                if nxt in seen or nxt in obstacles or self.blocked(nxt):
                    continue
                if nxt == self.food:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def flood_fill_count(self, start, obstacles):
        if start in obstacles or self.blocked(start):
            return 0
        seen = {start}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            for dx, dy in DIRECTIONS.values():
                nxt = (pos[0] + dx, pos[1] + dy)
                if nxt in seen or nxt in obstacles or self.blocked(nxt):
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return len(seen)

    def move_features(self, direction):
        move = self.preview_move(direction)
        if move["fatal"]:
            move["bfs_distance"] = None
            move["reachable_cells"] = 0
            move["adjacent_to_body"] = True
            move["free_ratio"] = 0.0
            return move
        obstacles = set(move["body_after"]) - {move["new_head"]}
        move["bfs_distance"] = self.bfs_distance(move["new_head"], obstacles)
        move["reachable_cells"] = self.flood_fill_count(move["new_head"], obstacles)
        nx, ny = move["new_head"]
        neighbors = [(nx + dx, ny + dy) for dx, dy in DIRECTIONS.values()]
        move["adjacent_to_body"] = any(n in obstacles for n in neighbors)
        move["free_ratio"] = move["reachable_cells"] / max(1, self.grid * self.grid - len(self.walls))
        return move

    def render(self):
        head = self.head
        body = set(list(self.snake)[:-1])
        lines = [f"len={len(self.snake)} food_eaten={self.food_eaten} steps={self.steps}"]
        for y in range(self.grid):
            row = []
            for x in range(self.grid):
                pos = (x, y)
                if pos == head:
                    row.append("@")
                elif pos in body:
                    row.append("#")
                elif pos == self.food:
                    row.append("*")
                elif pos in self.walls:
                    row.append("X")
                else:
                    row.append(".")
            lines.append("".join(row))
        return "\n".join(lines)
