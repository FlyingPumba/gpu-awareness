"""Check the fresh-run boundary and fixed 100-update generation cadence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from induced_coding_language.control import InducedController, RUN, CONFIG, SUITE


class Control(unittest.TestCase):
    def test_fresh_training_uses_new_suite_without_resume(self):
        c = InducedController.__new__(InducedController)
        c.run_id, c.run, c.suite_path = RUN, 'runs/'+RUN, SUITE
        c.update = Mock()
        idle = dict(exit=None,status=None,process='',gpu='')
        done = dict(exit='0',status={'event':'paused','step':100},process='',gpu='')
        with tempfile.TemporaryDirectory() as d:
            c.local = Path(d)
            with patch('coding2.interval_control.probe',return_value=idle), \
                 patch('coding2.interval_control.wait_stage',return_value=done), \
                 patch('coding2.interval_control.command',return_value='started') as launch:
                self.assertEqual(c.train_segment(0,100),(100,False))
            args = launch.call_args.args[0]
            self.assertNotIn('--resume',args)
            self.assertEqual(args[args.index('--suite')+1],SUITE)
            self.assertEqual(args[-1],'100')

    def test_baseline_then_hundred_intervals_then_selected(self):
        c = InducedController.__new__(InducedController)
        c.run_id, c.run = RUN, 'runs/'+RUN
        c.cfg = {'steps':1950}
        c.state = {'current_step':0,'training_terminal':False}
        c.update = lambda **kw:c.state.update(kw)
        calls = []
        c.full_evaluation = lambda stage,step:calls.append(('full',stage,step))
        c.train_segment = lambda start,end:calls.append(('train',start,end)) or ((100,False) if start==0 else (175,True))
        c.interval = lambda step,backfill:calls.append(('interval',step,backfill))
        with tempfile.TemporaryDirectory() as d:
            c.local=Path(d)
            with patch('induced_coding_language.control.remote',return_value={'best_step':150}):
                c.run_all()
        self.assertEqual(calls,[('full','baseline',0),('train',0,100),('interval',100,False),
                                ('train',100,200),('full','selected',150)])
        self.assertEqual(c.state['phase'],'evaluation_complete')
        self.assertEqual(c.state['current_step'],175)


if __name__ == '__main__':
    unittest.main()
