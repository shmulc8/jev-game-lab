"""Generalized, stdlib-only falling-piece engine used by the variants."""

import random


def _normalize(shape):
    min_c = min(c for c, _ in shape)
    min_r = min(r for _, r in shape)
    cells = {(c - min_c, r - min_r) for c, r in shape}
    return tuple(sorted(cells, key=lambda p: (p[1], p[0])))


def all_orientations(shape):
    """Return distinct rotations and reflections in deterministic order."""
    result = []
    seen = set()
    for reflected in (False, True):
        for turns in range(4):
            points = []
            for c, r in shape:
                if reflected:
                    c = -c
                for _ in range(turns):
                    c, r = -r, c
                points.append((c, r))
            normalized = _normalize(points)
            if normalized not in seen:
                seen.add(normalized)
                result.append(list(normalized))
    return result


class PieceBag:
    def __init__(self, seed, pieces):
        self.rng = random.Random(seed)
        self.pieces = list(pieces)
        self.queue = []

    def _refill(self):
        bag = self.pieces[:]
        self.rng.shuffle(bag)
        self.queue.extend(bag)

    def next(self):
        if not self.queue:
            self._refill()
        return self.queue.pop(0)

    def peek(self, ahead=1):
        while len(self.queue) < ahead:
            self._refill()
        return self.queue[ahead - 1]


class Board:
    def __init__(self, pieces, width, height, grid=None):
        self.pieces = pieces
        self.width = width
        self.height = height
        self.grid = grid if grid is not None else [[0] * width for _ in range(height)]

    def copy(self):
        return Board(self.pieces, self.width, self.height, [row[:] for row in self.grid])

    def col_top(self, col):
        for r in range(self.height):
            if self.grid[r][col]:
                return r
        return self.height

    def _drop_row_offset(self, shape, col):
        width = max(c for c, _ in shape) + 1
        if col < 0 or col + width > self.width:
            return None
        bottoms = {}
        tops = {}
        for c, r in shape:
            bottoms[c] = max(bottoms.get(c, r), r)
            tops[c] = min(tops.get(c, r), r)
        row = min(self.col_top(col + c) - 1 - bottoms[c] for c in bottoms)
        if any(row + tops[c] < 0 for c in tops):
            return None
        return row

    def legal_placements(self, piece):
        result = []
        for rot, shape in enumerate(self.pieces[piece]):
            width = max(c for c, _ in shape) + 1
            for col in range(self.width - width + 1):
                if self._drop_row_offset(shape, col) is not None:
                    result.append((rot, col))
        return result

    def placement_cells(self, piece, rot, col):
        shape = self.pieces[piece][rot]
        row = self._drop_row_offset(shape, col)
        if row is None:
            return None
        return [(col + c, row + r) for c, r in shape]

    def placement_bottom_row(self, piece, rot, col):
        cells = self.placement_cells(piece, rot, col)
        return max(r for _, r in cells) if cells else None

    def _place(self, piece, rot, col):
        cells = self.placement_cells(piece, rot, col)
        if cells is None:
            raise ValueError(f"illegal placement: {piece} rot={rot} col={col}")
        for c, r in cells:
            self.grid[r][c] = 1

    def _clear_lines(self):
        remaining = [row for row in self.grid if not all(row)]
        cleared = self.height - len(remaining)
        self.grid = [[0] * self.width for _ in range(cleared)] + remaining
        return cleared

    def drop(self, piece, rot, col):
        self._place(piece, rot, col)
        return self._clear_lines()

    def simulate(self, piece, rot, col):
        board = self.copy()
        board._place(piece, rot, col)
        return board, board._clear_lines()

    def column_heights(self):
        return [self.height - self.col_top(c) for c in range(self.width)]

    def aggregate_height(self):
        return sum(self.column_heights())

    def hole_count(self):
        return sum(
            not self.grid[r][c]
            for c in range(self.width)
            for r in range(self.col_top(c) + 1, self.height)
        )

    def bumpiness(self):
        heights = self.column_heights()
        return sum(abs(a - b) for a, b in zip(heights, heights[1:]))

    def max_height(self):
        return max(self.column_heights(), default=0)

    def render(self):
        rows = ["".join("#" if cell else "." for cell in row) for row in self.grid]
        return "\n".join(rows + ["".join(str(c % 10) for c in range(self.width))])
