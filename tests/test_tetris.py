import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agents
import play
import tetris
from viz import frames, replay


def answer(payload, _key):
    choice = next(iter(payload['questions']['placement']['criteria']))
    return {'answers': {'placement': {'choice': choice, 'confidence': 1}, 'topout': {'score': 0}}, 'usage': {'input_tokens': 1}}


class TetrisTests(unittest.TestCase):
    def test_preview_preserves_landing_shapes(self):
        policy = agents.JevAgent(mode='preview', api_key='offline-test')
        request = policy.build_request(tetris.Board(), 'O', 'I')
        criteria = request['questions']['placement']['criteria']
        self.assertEqual(len(criteria), 9)
        self.assertEqual(len(set(criteria.values())), 9)
        self.assertTrue(all('#' in value and 'bottom' in value for value in criteria.values()))

    def test_key_is_environment_only(self):
        with patch.dict(os.environ, {}, clear=True), patch('builtins.open', side_effect=AssertionError('must not read .env')):
            with self.assertRaisesRegex(RuntimeError, 'TYPESAFE_API_KEY'):
                agents._load_api_key()

    def test_pipeline_reuses_correct_next_piece(self):
        policy = agents.JevAgent(mode='rich', api_key='offline-test', pipeline=True)
        bag, board = tetris.PieceBag(7), tetris.Board()
        with patch.object(agents, 'call_jev', side_effect=answer):
            try:
                piece = bag.next()
                queue = [bag.peek(i) for i in range(1, 4)]
                chosen = policy.choose(board, piece, queue[0], next_pieces=queue)
                cleared = board.drop(piece, *chosen)
                piece = bag.next()
                queue = [bag.peek(i) for i in range(1, 4)]
                policy.choose(board, piece, queue[0], next_pieces=queue, lines_cleared_total=cleared)
                self.assertEqual(policy.stats['pipeline_hits'], 1)
            finally:
                policy.close()

    def test_capture_restored_when_callback_fails(self):
        original = agents.call_jev
        with self.assertRaisesRegex(RuntimeError, 'callback failed'):
            frames.run_tetris('heuristic', steps=1, frame_delay=0, on_frame=lambda _: (_ for _ in ()).throw(RuntimeError('callback failed')))
        self.assertIs(agents.call_jev, original)

    def test_verification_checks_sent_candidates(self):
        board = tetris.Board()
        capture = frames.Capture('tetris')
        capture.payload = {'state': {}, 'questions': {'placement': {'criteria': {'r0c0': 'only candidate'}}}}
        capture.response = {'answers': {'placement': {'choice': 'r0c1'}}}
        capture.latency_ms = 0
        import time
        frame = frames._tetris_frame(1, 1, 'O', [], 'rich', board.legal_placements('O'), 'r0c1', (0, 0), 0, board, [], capture, time.perf_counter(), 0, agents.HeuristicAgent())
        self.assertFalse(frame['code_verifies']['returned_key_in_candidate_set'])

    def test_offline_baseline_clears_lines(self):
        result = play.run('heuristic', 'rich', 40, 7, True)
        self.assertEqual((result['pieces_placed'], result['lines_cleared']), (40, 13))

    def test_static_replays_keep_recordings_separate_and_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = replay.build(directory)
            pages = [p for p in paths if p.parent.name == 'replays']
            self.assertEqual(len(pages), 6)
            for page in pages:
                content = page.read_text()
                embedded = json.loads(re.search(r'<script id="recording" type="application/json">(.*?)</script>', content).group(1))
                original = json.loads((replay.RECORDINGS / (page.stem + '.json')).read_text())
                self.assertEqual(embedded, original)
                self.assertGreater(len(embedded['frames']), 0)
                self.assertNotIn('fetch(', content)
                self.assertNotIn('/api/', content)
                self.assertNotIn('fonts.googleapis', content)
                if embedded['game'] == 'tetris':
                    self.assertIn('width="400" height="800"', content)


if __name__ == '__main__':
    unittest.main()
