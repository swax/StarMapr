"""All fixtures are synthetic vectors/blank images; no real face identification."""

import contextlib
import importlib.util
import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from celebrity_verifier import CelebrityVerifier, evaluate_response, configured_verifier, jpeg_bytes
from validation import (QualityPolicy, assess_reference, classify_candidate, corroborated_candidates,
                        embedding_spec, file_hash, metadata_path, validate_model_metadata, write_json)
from utils_deepface import cache_spec, get_face_embeddings

ROOT = Path(__file__).resolve().parents[1]


def script(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix('.py'), ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QualityTests(unittest.TestCase):
    def test_clear_reference_passes_and_normalizes_each_sample(self):
        centroid, report = assess_reference([[100, 0], [1, .01], [1, -.01]], QualityPolicy(3))
        self.assertTrue(report['accepted'])
        np.testing.assert_allclose(centroid, [1, 0], atol=1e-8)

    def test_no_count_fallback(self):
        self.assertFalse(assess_reference([[1, 0]] * 14, QualityPolicy())[1]['accepted'])

    def test_incoherent_group_cannot_pass_using_self_similarity(self):
        _, report = assess_reference(np.eye(15), QualityPolicy())
        self.assertFalse(report['accepted'])
        self.assertAlmostEqual(report['median'], 0)

    def test_mixed_and_invalid_references_rejected(self):
        for rows in ([[1, 0], [-1, 0], [0, 1]], [[0, 0]] * 3, [[float('nan'), 0]] * 3):
            self.assertFalse(assess_reference(rows, QualityPolicy(3))[1]['accepted'])

    def test_ambiguous_close_scores_and_wrong_winner_abstain(self):
        query = [1, 0]
        for score, rival in ((.519, .521), (.536, .470)):
            decision = classify_candidate(query, [score, np.sqrt(1-score**2)],
                                          {'other': [rival, np.sqrt(1-rival**2)]})
            self.assertEqual(decision['status'], 'ambiguous')

    def test_clear_match_and_low_score(self):
        self.assertEqual(classify_candidate([1, 0], [1, 0], {'other': [0, 1]})['status'], 'accepted')
        self.assertEqual(classify_candidate([0, 1], [1, 0], {})['status'], 'below_threshold')

    def test_corroboration_requires_separation_and_same_appearance(self):
        def candidate(pos, vector):
            return dict(frame_position=pos, face={'embedding': vector}, decision={'score': .8})
        first = candidate(1, [1, 0])
        self.assertEqual(corroborated_candidates([first, candidate(2, [1, 0])]), [])
        self.assertEqual(corroborated_candidates([first, candidate(100, [0, 1])]), [])
        self.assertEqual(len(corroborated_candidates([first, candidate(100, [1, 0])])), 2)


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'cache.sqlite'
        self.response = {'CelebrityFaces': [{'Id': 'expected', 'MatchConfidence': 99.9}], 'UnrecognizedFaces': []}
        self.client = Mock()
        self.client.recognize_celebrities.return_value = self.response

    def test_exact_id_only_not_name(self):
        response = {'CelebrityFaces': [{'Id': 'different', 'Name': 'Expected Name', 'MatchConfidence': 100}]}
        self.assertEqual(evaluate_response(response, 'expected', 99)['status'], 'identity_mismatch')

    def test_exact_catalog_name_avoids_per_actor_manual_setup(self):
        response = {'CelebrityFaces': [{'Id': 'catalog-id', 'Name': 'Example Actor', 'MatchConfidence': 100}]}
        self.assertEqual(evaluate_response(response, None, 99, 'EXAMPLE  Actor')['status'], 'verified')
        self.assertEqual(evaluate_response(response, None, 99, 'Example Acter')['status'], 'identity_mismatch')
        self.assertEqual(evaluate_response(response, 'different-id', 99, 'Example Actor')['status'], 'identity_mismatch')

    def test_unknown_multiple_and_low_confidence(self):
        for response in ({}, {'CelebrityFaces': self.response['CelebrityFaces'] * 2},
                         dict(self.response, UnrecognizedFaces=[{}])):
            self.assertEqual(evaluate_response(response, 'expected', 99)['status'], 'unknown_or_multiple_faces')
        self.assertEqual(evaluate_response({'CelebrityFaces': [{'Id': 'expected', 'MatchConfidence': 98}]},
                                          'expected', 99)['status'], 'low_confidence')

    def test_cache_rechecks_identity_and_confidence_without_extra_calls(self):
        verifier = CelebrityVerifier(self.path, client=self.client)
        self.assertEqual(verifier.verify(b'fixture', 'expected')['status'], 'verified')
        self.assertEqual(verifier.verify(b'fixture', 'wrong')['status'], 'identity_mismatch')
        stricter = CelebrityVerifier(self.path, client=self.client, confidence=100)
        self.assertEqual(stricter.verify(b'fixture', 'expected')['status'], 'low_confidence')
        self.client.recognize_celebrities.assert_called_once()

    def test_budget_persists_across_instances_and_counts_failures(self):
        self.client.recognize_celebrities.side_effect = RuntimeError('secret must not be logged')
        verifier = CelebrityVerifier(self.path, client=self.client, max_requests=1)
        self.assertEqual(verifier.verify(b'fixture', 'expected'),
                         {'status': 'verifier_unavailable', 'error_type': 'RuntimeError'})
        second = CelebrityVerifier(self.path, client=self.client, max_requests=1)
        self.assertEqual(second.verify(b'fixture', 'expected')['status'], 'budget_exhausted')
        self.client.recognize_celebrities.assert_called_once()

    def test_missing_id_makes_no_request_or_cache(self):
        verifier = CelebrityVerifier(self.path, client=self.client)
        self.assertEqual(verifier.verify(b'fixture', None)['status'], 'missing_identity_mapping')
        self.client.recognize_celebrities.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_disabled_is_explicit_and_unknown_mode_fails(self):
        with patch.dict(os.environ, {'CELEBRITY_VERIFIER': 'off'}):
            self.assertEqual(configured_verifier(self.path), (None, {}))
        with patch.dict(os.environ, {'CELEBRITY_VERIFIER': 'typo'}):
            with self.assertRaises(ValueError):
                configured_verifier(self.path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        old = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, old)
        env = patch.dict(os.environ, {'CELEBRITY_VERIFIER': 'off', 'TRAINING_MIN_IMAGES': '3',
                                     'OPERATIONS_MIN_FRAME_GAP': '25', 'OPERATIONS_MIN_CONFIRMING_FRAMES': '2'})
        env.start()
        self.addCleanup(env.stop)
        self.headshots = script('33_extract_video_headshots.py')
        self.video = Path('video')
        (self.video / 'frames').mkdir(parents=True)
        self.output = self.video / 'headshots' / 'example'
        self.model = Path('04_models/example_average_embedding.pkl')
        self.model.parent.mkdir()
        self.model.write_bytes(pickle.dumps(np.array([1., 0.])))
        _, quality = assess_reference([[1, 0]] * 3, QualityPolicy(3))
        write_json(metadata_path(self.model), dict(quality=quality, embedding_spec=embedding_spec(),
                                                  model_sha256=file_hash(self.model), training_images={}))
        self.make_frame(1)
        self.make_frame(100)

    def make_frame(self, position, vector=(1, 0)):
        path = self.video / 'frames' / f'{position:08}.jpg'
        cv2.imwrite(str(path), np.full((400, 400, 3), position % 255, dtype=np.uint8))
        face = dict(face_id=1, embedding=list(vector), isHeadshotable=True,
                    bounding_box={'x': 170, 'y': 120, 'w': 50, 'h': 50})
        path.with_suffix('.pkl').write_bytes(pickle.dumps(dict(frame_file=path.name, faces=[face], cache_spec=cache_spec(path))))

    def run_extraction(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.headshots.extract_top_headshots('Example', self.video, **kwargs)

    def test_clear_corroborated_candidate_publishes_and_counts_real_files(self):
        result = self.run_extraction()
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(len(result['headshots']), 2)
        self.assertEqual(len(list(self.output.glob('*.jpg'))), 2)
        self.assertEqual(result['verification'], 'local_only')

    def test_unvalidated_model_abstains_and_removes_stale_headshots(self):
        self.run_extraction()
        metadata_path(self.model).unlink()
        result = self.run_extraction()
        self.assertEqual(result['status'], 'model_unvalidated')
        self.assertFalse(result['retryable'])
        self.assertEqual(list(self.output.glob('*.jpg')), [])

    def test_model_hash_prevents_stale_report_reuse(self):
        self.model.write_bytes(pickle.dumps(np.array([0., 1.])))
        self.assertIsNone(validate_model_metadata(self.model))

    def test_cloud_input_is_the_detected_face_not_the_padded_headshot(self):
        path = self.video / 'frames/00000001.jpg'
        encoded = jpeg_bytes(path, {'x': 170, 'y': 120, 'w': 50, 'h': 50})
        crop = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(crop.shape[:2], (50, 50))

    def test_failed_promotion_restores_previous_model(self):
        training = script('02_run_actor_training.py')
        source = Path('02_training/example/example_average_embedding.pkl')
        source.parent.mkdir(parents=True)
        previous_bytes = self.model.read_bytes()
        source.write_bytes(pickle.dumps(np.array([1., .1])))
        report = json.loads(metadata_path(self.model).read_text())
        report['model_sha256'] = file_hash(source)
        write_json(metadata_path(source), report)
        real_replace = os.replace
        def fail_report_replace(source_path, destination):
            if Path(destination) == metadata_path(self.model):
                raise OSError('simulated report promotion failure')
            return real_replace(source_path, destination)
        with patch.object(training.os, 'replace', side_effect=fail_report_replace), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(training.copy_model_to_models_dir('Example'))
        self.assertEqual(self.model.read_bytes(), previous_bytes)
        self.assertIsNotNone(validate_model_metadata(self.model))

    def test_competitor_vetoes_even_if_local_score_is_high(self):
        Path('04_models/rival_average_embedding.pkl').write_bytes(self.model.read_bytes())
        result = self.run_extraction()
        self.assertEqual(result['status'], 'no_reliable_headshot')
        self.assertEqual(result['rejections']['ambiguous'], 2)
        self.assertFalse(result['headshots'])

    def test_cloud_mismatch_never_falls_back_to_local(self):
        fake = Mock()
        fake.verify.return_value = {'status': 'identity_mismatch'}
        with patch.object(self.headshots, 'configured_verifier', return_value=(fake, {'example': 'expected'})):
            result = self.run_extraction()
        self.assertFalse(result['headshots'])
        self.assertEqual(result['rejections']['identity_mismatch'], 2)

    def test_cloud_unavailable_stops_retries(self):
        fake = Mock()
        fake.verify.return_value = {'status': 'verifier_unavailable'}
        with patch.object(self.headshots, 'configured_verifier', return_value=(fake, {'example': 'expected'})):
            result = self.run_extraction()
        self.assertEqual(result['status'], 'verifier_unavailable')
        self.assertFalse(result['retryable'])
        fake.verify.assert_called_once()

    def test_cloud_success_is_required_for_each_output(self):
        fake = Mock()
        fake.verify.side_effect = [{'status': 'verified'}, {'status': 'identity_mismatch'}]
        with patch.object(self.headshots, 'configured_verifier', return_value=(fake, {'example': 'expected'})):
            result = self.run_extraction()
        self.assertEqual(len(result['headshots']), 1)

    def test_dry_run_does_not_write_or_call_aws(self):
        fake = Mock()
        with patch.object(self.headshots, 'configured_verifier', return_value=(fake, {'example': 'expected'})):
            result = self.run_extraction(dry_run=True)
        self.assertEqual(result['status'], 'not_run_dry_run')
        self.assertFalse(self.output.exists())
        fake.verify.assert_not_called()

    def test_stale_frame_cache_is_error_not_no_match(self):
        frame = self.video / 'frames/00000001.jpg'
        frame.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Stale cache'):
            self.run_extraction()

    def test_cache_reuse_and_stale_cache_regeneration(self):
        frame = self.video / 'frames/00000001.jpg'
        self.assertEqual(get_face_embeddings(frame)[0]['embedding'], [1, 0])
        # DeepFace is mocked: this never performs inference, even on blank images.
        deepface = Mock()
        deepface.DeepFace.represent.return_value = [dict(
            facial_area={'x': 170, 'y': 120, 'w': 50, 'h': 50},
            embedding=[1., 0.], face_confidence=.99)]
        with patch.dict('sys.modules', {'deepface': deepface}), patch.dict(os.environ, {'MIN_FACE_SIZE': '51'}):
            regenerated = get_face_embeddings(frame)
        deepface.DeepFace.represent.assert_called_once()
        self.assertFalse(regenerated[0]['isHeadshotable'])

    def test_benchmark_measures_known_labels_and_rejects_training_leakage(self):
        benchmark = script('90_benchmark_validation.py')
        cases = [dict(actor='Example', image='video/frames/00000001.jpg', expected_match=True),
                 dict(actor='Example', image='video/frames/00000100.jpg', expected_match=False)]
        write_json('benchmark.json', cases)
        with patch.object(benchmark, 'get_face_embeddings', side_effect=[[{'embedding': [1, 0]}], [{'embedding': [0, 1]}]]), \
                contextlib.redirect_stdout(io.StringIO()):
            report = benchmark.run_benchmark('benchmark.json', 'result.json')
        self.assertEqual(report['counts']['true_positive'], 1)
        self.assertEqual(report['counts']['true_negative'], 1)
        metadata = json.loads(metadata_path(self.model).read_text())
        metadata['training_images'] = {'seed.jpg': file_hash(Path(cases[0]['image']))}
        write_json(metadata_path(self.model), metadata)
        with self.assertRaisesRegex(ValueError, 'Training image leaked'):
            benchmark.run_benchmark('benchmark.json', 'result.json')

    def test_benchmark_rejects_missing_negatives(self):
        benchmark = script('90_benchmark_validation.py')
        write_json('benchmark.json', [dict(actor='Example', image='video/frames/00000001.jpg', expected_match=True)])
        with self.assertRaisesRegex(ValueError, 'positives AND negatives'):
            benchmark.run_benchmark('benchmark.json', 'result.json')

    def test_training_largest_bad_group_is_not_saved(self):
        training = script('03_run_training_pipeline.py')
        folder = Path('training')
        folder.mkdir()
        for n in range(3):
            (folder / f'{n}.jpg').write_bytes(b'synthetic')
        faces = [[{'embedding': row}] for row in np.eye(3)]
        with patch.object(training, 'get_face_embeddings', side_effect=faces):
            passed, _, best = training.check_image_threshold(folder, 3, 0)
        self.assertFalse(passed)
        self.assertEqual(best, 0)
        self.assertFalse((folder / 'best_group/best_group.txt').exists())

    def test_dbscan_all_noise_removes_every_candidate(self):
        clustering = script('14_cluster_and_keep_largest.py')
        folder = Path('training')
        folder.mkdir()
        for n in range(3):
            (folder / f'{n}.jpg').write_bytes(b'synthetic')
        with patch.object(clustering, 'get_single_face_embedding', side_effect=list(np.eye(3))):
            clustering.cluster_and_keep_largest(folder, eps=.1)
        self.assertEqual(list(folder.glob('*.jpg')), [])
        self.assertEqual(len(list((folder / 'outliers').glob('*.jpg'))), 3)

    def test_completed_actor_is_not_retried_and_errors_are_not_success(self):
        pipeline = script('01_run_headshot_detection.py')
        completed = dict(actor='Example', status='accepted', retryable=False, headshots=[{'file': 'one.jpg'}])
        unknown = dict(actor='Other', status='no_reliable_headshot', retryable=True, headshots=[])
        with patch.object(pipeline, 'extract_frames_from_video', return_value=True), \
                patch.object(pipeline, 'extract_faces_from_frames', return_value=True), \
                patch.object(pipeline, 'extract_actor_headshots', side_effect=lambda actor, _: (True, completed if actor == 'Example' else unknown)) as extract, \
                contextlib.redirect_stdout(io.StringIO()):
            result = pipeline.run_operations_pipeline_with_adaptive_frames(self.video, ['Example', 'Other'])
        self.assertEqual(result, {'Example': 1, 'Other': 0})
        self.assertEqual(sum(call.args[0] == 'Example' for call in extract.call_args_list), 1)
        self.assertEqual(sum(call.args[0] == 'Other' for call in extract.call_args_list), 5)
        self.assertEqual(unknown['stop_reason'], 'frame_attempt_limit')
        with patch.object(pipeline, 'extract_frames_from_video', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                pipeline.run_operations_pipeline_with_adaptive_frames(self.video, ['Example'])


if __name__ == '__main__':
    unittest.main()
