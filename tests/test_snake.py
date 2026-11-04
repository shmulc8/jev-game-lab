"""Offline Snake regressions; all model responses are synthetic."""
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
SNAKE = ROOT / "snake"


def python(code):
    return subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(code)],
        cwd=SNAKE, capture_output=True, text=True, check=True,
    ).stdout


class SnakeTests(unittest.TestCase):
    def test_seed7_baselines(self):
        result = python('''
            import engine, policies
            for cls, expected in [(policies.GreedyPolicy,15),(policies.FloodPolicy,15)]:
                game=engine.SnakeGame(seed=7)
                policy=cls()
                for _ in range(150): game.step(policy.choose(game))
                assert game.food_eaten == expected
                assert game.alive
            game=engine.SnakeGame(seed=7)
            policy=policies.RandomPolicy(seed=7)
            for _ in range(150): game.step(policy.choose(game))
            assert game.food_eaten == 0
            print('ok')
        ''')
        self.assertEqual(result.strip(), 'ok')

    def test_key_environment_only_and_invalid_choice_fallback(self):
        python('''
            import os
            from unittest.mock import patch
            import client, engine, policies
            with patch.dict(os.environ, {}, clear=True), patch('builtins.open', side_effect=AssertionError('file read')):
                client._api_key_cache=None
                try: client.api_key()
                except RuntimeError: pass
                else: raise AssertionError('missing key accepted')
            game=engine.SnakeGame(seed=7)
            policy=policies.JevPolicy(mode='features')
            with patch.object(client,'ask',return_value=({'answers':{'direction':{'choice':'invalid','confidence':1}}},1)):
                assert policy.choose(game)==policies.flood_choice(game)
            assert policy.stats['invalid_choices']==1
        ''')

    def test_composite_budget_rejected_without_request(self):
        result=subprocess.run(
            [sys.executable,'-B',str(SNAKE/'play.py'),'--policy','jev','--mode','composite','--tick-ms','10'],
            capture_output=True,text=True,
        )
        self.assertEqual(result.returncode,2)
        self.assertIn('does not support',result.stderr)
        python('''
            from unittest.mock import patch
            import client, engine, policies
            with patch.object(client,'ask',side_effect=AssertionError('network')):
                try: policies.JevPolicy(mode='composite').choose(engine.SnakeGame(), deadline_s=.01)
                except ValueError: pass
                else: raise AssertionError('budget accepted')
        ''')

    def test_replay_stops_at_changed_action_and_restores_walls(self):
        python('''
            import argparse, contextlib, io, tempfile, json
            from pathlib import Path
            from unittest.mock import patch
            import client, engine, policies, play
            def answer(state, questions, model):
                return {'answers':{key:{'score': (3 if key.startswith('room_') else 0), 'confidence':1} for key in questions}}, 1
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'trace.jsonl'
                game=engine.SnakeGame(seed=7,walls=12)
                policy=policies.JevPolicy(mode='composite',trace_path=str(path))
                policy.start_trace(game,20,7)
                with patch.object(client,'ask',side_effect=answer):
                    while game.alive and game.steps<20: game.step(policy.choose(game))
                policy.finish_trace(game)
                args=argparse.Namespace(rerank=str(path),weight_room=.5,weight_food=.5)
                output=io.StringIO()
                with patch.object(client,'ask',side_effect=AssertionError('network')), contextlib.redirect_stdout(output): play.rerank(args)
                assert 'entire game' in output.getvalue(), output.getvalue()
                assert f'food_eaten={game.food_eaten} steps={game.steps}' in output.getvalue()

                # Both candidate sets remain legal after the first changed action.
                # Replay must stop immediately, before reading the next judgment.
                records=play.load_trace(path)
                meta=records[0]
                fresh=engine.SnakeGame(seed=7,walls=12)
                moves=fresh.survivable_moves()
                candidates={m['direction']:{'room_score':i,'room_levels':4,'food_score':3-i,'food_levels':4} for i,m in enumerate(moves)}
                original=max(candidates,key=lambda d:policies.weighted_value(candidates[d],1,0))
                records=[meta,{'tick':0,'candidates':candidates,'chosen':original,'confidence_fallback':False}]
                path.write_text(''.join(json.dumps(r)+'\\n' for r in records))
                args.weight_room=0;args.weight_food=1
                output=io.StringIO()
                with contextlib.redirect_stdout(output): play.rerank(args)
                assert 'trajectory diverged at tick 0' in output.getvalue(), output.getvalue()
        ''')


if __name__ == '__main__':
    unittest.main()
