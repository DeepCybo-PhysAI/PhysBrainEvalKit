import base64
import csv
import io
import json
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
from benchmark.mindcube import MindCubeDataset
from benchmark.robovqa import RoboVQADataset
from benchmark.threedsrbench import ThreeDSRBenchDataset
from benchmark.vlabench import VLABenchDataset
from core.hf_data import resolve_snapshot


class HubDatasetTests(unittest.TestCase):
    def test_local_path_and_hub_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('huggingface_hub.snapshot_download', return_value=directory) as download:
                self.assertEqual(resolve_snapshot(directory), Path(directory).resolve())
                download.assert_not_called()
                self.assertEqual(resolve_snapshot('VLyb/VLABench'), Path(directory))
                download.assert_called_once_with(repo_id='VLyb/VLABench', repo_type='dataset')
                with self.assertRaises(FileNotFoundError):
                    resolve_snapshot(str(Path(directory) / 'missing'))

    def test_robovqa_tar_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'qa.jsonl').write_text(json.dumps({'frame_paths': ['frames/0.png']}) + '\n')
            with tarfile.open(root / 'frames.tar', 'w') as archive:
                info = tarfile.TarInfo('frames/0.png')
                info.size = 1
                archive.addfile(info, io.BytesIO(b'x'))
            with patch('huggingface_hub.snapshot_download', return_value=directory):
                rows = RoboVQADataset(expected_num_frames=1).load_dataset()
            self.assertEqual(rows[0]['images'][0]['member'], 'frames/0.png')
            self.assertEqual(rows[0]['images'][0]['type'], 'tar_image')

    def test_mindcube_snapshot_relative_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/raw').mkdir(parents=True)
            (root / 'data/other_all_image').mkdir()
            image_path = root / 'data/other_all_image/0.png'
            Image.new('RGB', (2, 2)).save(image_path)
            (root / 'data/raw/MindCube_tinybench.jsonl').write_text(json.dumps({
                'images': ['other_all_image/0.png'], 'gt_answer': 'A', 'question': 'Which?'
            }) + '\n')
            with patch('huggingface_hub.snapshot_download', return_value=directory):
                dataset = MindCubeDataset()
                prepared = dataset.prepare_dataset(dataset.load_dataset())
            self.assertEqual(prepared[0]['image'], [image_path])
            self.assertEqual(prepared[0]['answer'], 'A')

    def test_threedsr_snapshot_embedded_image(self):
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.BytesIO()
            Image.new('RGB', (2, 2)).save(buffer, format='PNG')
            with (Path(directory) / '3dsrbench_v1_vlmevalkit_circular.tsv').open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=['question', 'answer', 'A', 'image'], delimiter='\t')
                writer.writeheader()
                writer.writerow({'question': 'Which?', 'answer': 'A', 'A': 'Left',
                                 'image': base64.b64encode(buffer.getvalue()).decode()})
            with patch('huggingface_hub.snapshot_download', return_value=directory):
                dataset = ThreeDSRBenchDataset()
                prepared = dataset.prepare_dataset(dataset.load_dataset())
            self.assertEqual(prepared[0]['image'].size, (2, 2))
            self.assertEqual(prepared[0]['answer'], 'A')

    def test_vlabench_snapshot_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'Spatial/task/example0'
            (root / 'input').mkdir(parents=True)
            (root / 'output').mkdir()
            (root / 'input/instruction.txt').write_text('Move the object.')
            (root / 'output/operation_sequence.json').write_text('[]')
            with patch('huggingface_hub.snapshot_download', return_value=directory):
                rows = VLABenchDataset(subset=['Spatial']).load_dataset()
            self.assertEqual(rows[0]['instruction'], 'Move the object.')
            self.assertEqual(rows[0]['input_image_path'], str(root / 'input/input.png'))

    def test_robovqa_cli_invokes_evaluation(self):
        inference = types.ModuleType('core.inference')
        inference.create_inference_engine = MagicMock(return_value=(MagicMock(), 'hf'))
        logger = types.ModuleType('core.logger')
        logger.setup_logging = MagicMock()
        dataset = MagicMock()
        dataset.prepare_dataset.return_value = []
        script = Path(__file__).resolve().parents[1] / 'eval_robovqa.py'
        with patch.dict(sys.modules, {'core.inference': inference, 'core.logger': logger}), \
             patch('benchmark.robovqa.RoboVQADataset', return_value=dataset) as adapter, \
             patch.object(sys, 'argv', [str(script), '--model_name', 'test', '--model_path', '/model']):
            runpy.run_path(str(script), run_name='__main__')
        self.assertEqual(adapter.call_args.kwargs['dataset_name'], 'VLyb/RoboVQA-16frames')
        self.assertEqual(adapter.call_args.kwargs['expected_num_frames'], 16)
        dataset.load_dataset.assert_called_once()
        dataset.evaluate_results.assert_called_once()


if __name__ == '__main__':
    unittest.main()
