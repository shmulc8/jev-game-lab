"""Adversarial piece selection shared by variant and versus games."""

from agents import afterstate_value


class HatetrisGenerator:
    def __init__(self, pieces):
        self.pieces = pieces

    def best_placement(self, board, piece):
        best, best_value = None, None
        for rot, col in board.legal_placements(piece):
            after, cleared = board.simulate(piece, rot, col)
            value = afterstate_value(cleared, after.hole_count(), after.aggregate_height(), after.max_height(), after.bumpiness())
            if best_value is None or value > best_value:
                best, best_value = (rot, col), value
        return best

    def next(self, board):
        worst, worst_value = None, None
        for kind in self.pieces:
            placement = self.best_placement(board, kind)
            if placement is None:
                value = float("-inf")
            else:
                after, cleared = board.simulate(kind, *placement)
                value = afterstate_value(cleared, after.hole_count(), after.aggregate_height(), after.max_height(), after.bumpiness())
            if worst_value is None or value < worst_value:
                worst, worst_value = kind, value
        return worst

    def preview_after(self, board, piece):
        placement = self.best_placement(board, piece)
        if placement is None:
            return None
        after, _ = board.simulate(piece, *placement)
        return self.next(after)
