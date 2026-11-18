"""Submission tests use fake scheduler responses and never launch GPU work."""
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cluster_train as launcher
from train import TrainConfig, training_run_name

INVENTORY = '''node1|gpu:h100:4(S:0-1)
node2|gpu:h200:8(S:0-1)
node3-c1|gpu:b200:8
node4|gpu:l40s:4
node5|(null)
node6|gpu:h100:1,gpu:l40s:1
'''


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="launcher test '$ ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = TrainConfig(model_dir=self.root/'models', wandb_online=False)
        self.paths = SimpleNamespace(temp_script_dir=str(self.root/'stage'),
                                     slurm_log_dir=str(self.root/'logs'))
        self.patch = patch('utils.USER_CONFIG', self.paths)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def scheduler(self, command, **kwargs):
        if command[0] == 'sinfo':
            return SimpleNamespace(stdout=INVENTORY)
        self.assertEqual(command[0], 'sbatch')
        self.assertNotIn('shell', kwargs)
        return SimpleNamespace(stdout='12345;eofe7\n', stderr='')

    def test_gpu_filter_excludes_wrong_mixed_and_cpu_nodes(self):
        self.assertEqual(launcher.node_selection(INVENTORY, launcher.SUPPORTED_GPUS, 1),
                         (['node1', 'node2', 'node3-c1'], ['node4', 'node5', 'node6']))
        eligible, excluded = launcher.node_selection(INVENTORY, ('H200',), 1)
        self.assertEqual(eligible, ['node2'])
        self.assertIn('node1', excluded)
        with self.assertRaises(ValueError):
            launcher.node_selection(INVENTORY, ('H100',), 8)
        with self.assertRaises(ValueError):
            launcher.gpu_types('A100')

    def test_pool_selection_and_default_array(self):
        for value in ('any', None, ('H100', 'H200', 'B200'), 'h100,h200,b200'):
            self.assertEqual(launcher.gpu_types(value), launcher.SUPPORTED_GPUS)
        with patch('cluster_train.subprocess.run', side_effect=self.scheduler):
            jobs = launcher.launch_training_jobs([self.config], dry_run=True)
        command = jobs[0]['submission_command']
        self.assertIn('--gpus-per-task=1', command)
        self.assertIn('--exclude=node4,node5,node6', command)
        self.assertFalse(any(arg.startswith('--gres=') for arg in command))
        payload = json.loads(Path(jobs[0]['manifest_path']).read_text())
        self.assertEqual(payload['gpu'], list(launcher.SUPPORTED_GPUS))

    def test_subset_pool_uses_one_throttled_array(self):
        configs = [self.config, replace(self.config, learning_rate=.001)]
        with patch('cluster_train.subprocess.run', side_effect=self.scheduler) as run:
            jobs = launcher.launch_training_jobs(configs, gpu=('h100', 'H200'),
                                                 max_parallel_runs=1)
        self.assertEqual(run.call_count, 2)
        command = jobs[0]['submission_command']
        self.assertIn('--array=0-1%1', command)
        self.assertIn('--gpus-per-task=1', command)
        self.assertIn('--exclude=node3-c1,node4,node5,node6', command)
        self.assertEqual(jobs[0]['slurm_array_id'], jobs[1]['slurm_array_id'])

    def test_dry_run_stages_roundtrippable_configs_and_valid_shell(self):
        with patch('cluster_train.subprocess.run', side_effect=self.scheduler) as run:
            jobs = launcher.launch_training_jobs([self.config], gpu='H200', dry_run=True)
        self.assertEqual(run.call_count, 1)  # sinfo only
        self.assertIsNone(jobs[0]['slurm_job_id'])
        payload = json.loads(Path(jobs[0]['manifest_path']).read_text())
        decoded = launcher.decode_config(payload['jobs'][0]['config'])
        self.assertEqual(decoded.model_dir, str(self.root/'models'))
        self.assertEqual(decoded.train_dataset, self.config.train_dataset)
        self.assertEqual(decoded.metric_loggers[0].fn, 'module_rms_logging:log_module_rms')
        subprocess.run(['bash', '-n', jobs[0]['script_path']], check=True)
        command = jobs[0]['submission_command']
        self.assertIn('--exclude=node1,node3-c1,node4,node5,node6', command)
        self.assertIn('--gres=gpu:h200:1', command)
        self.assertFalse(any(c.startswith('--nodelist') for c in command))

    def test_submission_array_throttle_and_original_index_after_skip(self):
        completed = self.root/'models'/training_run_name(self.config)
        completed.mkdir(parents=True)
        (completed/'config.json').write_text('{}')
        (completed/'model.pt').write_bytes(b'test fixture')
        configs = [self.config, replace(self.config, learning_rate=.001),
                   replace(self.config, learning_rate=.002)]
        with patch('cluster_train.subprocess.run', side_effect=self.scheduler) as run:
            jobs = launcher.launch_training_jobs(configs, gpu='H200', max_parallel_runs=1)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(jobs[0]['skipped'])
        self.assertEqual(jobs[1]['slurm_job_id'], '12345_0')
        self.assertEqual(jobs[2]['slurm_job_id'], '12345_1')
        self.assertIn('--array=0-1%1', jobs[1]['submission_command'])
        payload = json.loads(Path(jobs[1]['manifest_path']).read_text())
        self.assertEqual([r['job_index'] for r in payload['jobs']], [1, 2])

    def test_duplicate_outputs_fail_before_scheduler(self):
        with patch('cluster_train.subprocess.run') as run:
            with self.assertRaises(ValueError):
                launcher.launch_training_jobs([self.config, self.config])
        run.assert_not_called()

    def test_no_pending_jobs_does_not_contact_scheduler(self):
        with patch('model_io.is_native_model_dir', return_value=True), patch('cluster_train.subprocess.run') as run:
            jobs = launcher.launch_training_jobs([self.config])
        run.assert_not_called()
        self.assertTrue(jobs[0]['skipped'])

    def test_blackwell_runtime_check(self):
        with patch('torch.__version__', '2.6.0'), patch('torch.version.cuda', '12.4'):
            with self.assertRaisesRegex(RuntimeError, 'B200'):
                launcher.check_runtime(('B200',))
            launcher.check_runtime(('H100', 'H200'))
        with patch('torch.__version__', '2.7.1+cu128'), patch('torch.version.cuda', '12.8'):
            launcher.check_runtime(('B200',))

    def test_a2_factory_and_custom_config_round_trip(self):
        from experiments.a2.cluster_launcher import config
        from model_config import LMConfig
        c = config(optimizer_name='muon', batch=128, model_config=LMConfig(
            'custom', 4096, 1024, 128, 448, 2, 2, 2),
            model_builder_kwargs={'scale': .5}, keep_checkpoint_steps=(1, 2))
        decoded = launcher.decode_config(launcher.encode_config(c))
        self.assertEqual(c, decoded)
        self.assertEqual(c.num_micro_batches, 2)
        self.assertEqual(c.optimizer_builder, 'experiments.a2.optimizers:build_optimizer')
        self.assertNotIn('/root/shared_data', c.train_dataset.path)

    def test_worker_refuses_cpu_or_wrong_gpu(self):
        path = self.root/'manifest.json'
        path.write_text(json.dumps(dict(version=1, gpu=['H200'],
            jobs=[dict(config=launcher.encode_config(self.config))])))
        with patch('torch.cuda.is_available', return_value=False), patch('train.train') as train:
            with self.assertRaisesRegex(RuntimeError, 'no CUDA'):
                launcher.run_task(path, 0)
            train.assert_not_called()
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_name', return_value='NVIDIA L40S'), patch('train.train') as train:
            with self.assertRaisesRegex(RuntimeError, 'unexpected GPU'):
                launcher.run_task(path, 0)
            train.assert_not_called()

    def test_worker_dispatches_selected_config(self):
        path = self.root/'manifest.json'
        second = replace(self.config, learning_rate=.001)
        path.write_text(json.dumps(dict(version=1, gpu=['H200'], jobs=[
            dict(config=launcher.encode_config(c)) for c in (self.config, second)])))
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_name', return_value='NVIDIA H200'), patch('train.train') as train:
            launcher.run_task(path, 1)
            self.assertEqual(train.call_args.args[0].learning_rate, .001)


if __name__ == '__main__':
    unittest.main()
