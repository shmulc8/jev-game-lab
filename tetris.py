"""Pure Tetris engine: board, pieces, placement search, simulation. No AI, no network."""

import random

WIDTH = 10
HEIGHT = 20

# Each rotation is a list of (col, row) offsets in a minimal bounding box,
# top-left origin, row increasing downward. Only distinct shapes are listed
# (I and O collapse to fewer than 4 rotations since some rotations repeat).
PIECES = {
    "I": [
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
    ],
    "O": [
        [(0, 0), (1, 0), (0, 1), (1, 1)],
    ],
    "T": [
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
    ],
    "S": [
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
    ],
    "Z": [
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(1, 0), (0, 1), (1, 1), (0, 2)],
    ],
    "J": [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    "L": [
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (0, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
}

PIECE_NAMES = list(PIECES.keys())


def shape_ascii(shape):
    width = max(c for c, r in shape) + 1
    height = max(r for c, r in shape) + 1
    cells = set(shape)
    rows = []
    for r in range(height):
        rows.append("".join("#" if (c, r) in cells else "." for c in range(width)))
    return "\n".join(rows)


class PieceBag:
    """Deterministic 7-bag piece sequence from a seeded RNG."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.queue = []

    def _refill(self):
        bag = list(PIECE_NAMES)
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
    def __init__(self, grid=None):
        self.grid = grid if grid is not None else [[0] * WIDTH for _ in range(HEIGHT)]

    def copy(self):
        return Board([row[:] for row in self.grid])

    def col_top(self, col):
        for r in range(HEIGHT):
            if self.grid[r][col]:
                return r
        return HEIGHT

    def _drop_row_offset(self, shape, col):
        width = max(c for c, r in shape) + 1
        if col < 0 or col + width > WIDTH:
            return None
        piece_bottom = {}
        piece_top = {}
        for c, r in shape:
            piece_bottom[c] = max(piece_bottom.get(c, r), r)
            piece_top[c] = min(piece_top.get(c, r), r)
        row_offset = min(
            self.col_top(col + dc) - 1 - piece_bottom[dc] for dc in piece_bottom
        )
        if any(row_offset + piece_top[dc] < 0 for dc in piece_top):
            return None
        return row_offset

    def legal_placements(self, piece):
        placements = []
        for rot, shape in enumerate(PIECES[piece]):
            width = max(c for c, r in shape) + 1
            for col in range(0, WIDTH - width + 1):
                if self._drop_row_offset(shape, col) is not None:
                    placements.append((rot, col))
        return placements

    def _place(self, piece, rot, col):
        shape = PIECES[piece][rot]
        row_offset = self._drop_row_offset(shape, col)
        if row_offset is None:
            raise ValueError(f"illegal placement: {piece} rot={rot} col={col}")
        for c, r in shape:
            self.grid[row_offset + r][col + c] = 1
        return row_offset

    def _clear_lines(self):
        remaining = [row for row in self.grid if not all(row)]
        cleared = HEIGHT - len(remaining)
        self.grid = [[0] * WIDTH for _ in range(cleared)] + remaining
        return cleared

    def drop(self, piece, rot, col):
        self._place(piece, rot, col)
        return self._clear_lines()

    def simulate(self, piece, rot, col):
        new_board = self.copy()
        new_board._place(piece, rot, col)
        lines_cleared = new_board._clear_lines()
        return new_board, lines_cleared

    def placement_bottom_row(self, piece, rot, col):
        """Board row of the lowest cell the piece would occupy at this placement."""
        shape = PIECES[piece][rot]
        row_offset = self._drop_row_offset(shape, col)
        if row_offset is None:
            return None
        return row_offset + max(r for c, r in shape)

    def placement_cells(self, piece, rot, col):
        """Absolute (col, row) board cells the piece would occupy at this placement,
        before any line clears are applied. Returns None if the placement is illegal."""
        shape = PIECES[piece][rot]
        row_offset = self._drop_row_offset(shape, col)
        if row_offset is None:
            return None
        return [(col + c, row_offset + r) for c, r in shape]

    def render(self):
        lines = ["".join("#" if cell else "." for cell in row) for row in self.grid]
        footer = "".join(str(c % 10) for c in range(WIDTH))
        return "\n".join(lines + [footer])

    def column_heights(self):
        return [HEIGHT - self.col_top(c) for c in range(WIDTH)]

    def aggregate_height(self):
        return sum(self.column_heights())

    def hole_count(self):
        holes = 0
        for c in range(WIDTH):
            top = self.col_top(c)
            for r in range(top + 1, HEIGHT):
                if not self.grid[r][c]:
                    holes += 1
        return holes

    def bumpiness(self):
        heights = self.column_heights()
        return sum(abs(heights[i] - heights[i + 1]) for i in range(WIDTH - 1))

    def max_height(self):
        heights = self.column_heights()
        return max(heights) if heights else 0
