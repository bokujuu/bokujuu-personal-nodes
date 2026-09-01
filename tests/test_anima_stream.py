import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "anima_stream.py"
SPEC = importlib.util.spec_from_file_location("bokujuu_anima_stream", MODULE_PATH)
ANIMA_STREAM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANIMA_STREAM
SPEC.loader.exec_module(ANIMA_STREAM)


def fake_denoiser(x, sigma):
    sigma = sigma.reshape([sigma.shape[0]] + [1] * (x.ndim - 1))
    return x * 0.25 + sigma * 0.1


def sequential_sample(x, sigmas):
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index].expand(x.shape[0])
        sigma_next = sigmas[index + 1].expand(x.shape[0])
        x = ANIMA_STREAM.flow_euler_step(x, fake_denoiser(x, sigma), sigma, sigma_next)
    return x


class AnimaStreamBatchTests(unittest.TestCase):
    def test_mixed_timestep_matches_sequential_euler(self):
        latents = torch.arange(6 * 2 * 3, dtype=torch.float32).reshape(6, 2, 3) / 10
        sigmas = torch.tensor([1.0, 0.72, 0.41, 0.18, 0.0])

        actual, stats = ANIMA_STREAM.sample_stream_batch(fake_denoiser, latents, sigmas, microbatch_size=4)
        expected = sequential_sample(latents, sigmas)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(stats.max_active_slots, 4)
        self.assertEqual(stats.max_forward_batch, 4)

    def test_one_forward_contains_four_distinct_timesteps(self):
        calls = []

        def recording_denoiser(x, sigma):
            calls.append(sigma.clone())
            return fake_denoiser(x, sigma)

        latents = torch.zeros((8, 2, 2))
        sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        ANIMA_STREAM.sample_stream_batch(recording_denoiser, latents, sigmas, microbatch_size=4)

        self.assertTrue(any(len(torch.unique(call)) == 4 for call in calls))

    def test_preserves_frame_order(self):
        latents = torch.arange(12, dtype=torch.float32).reshape(12, 1, 1)
        sigmas = torch.tensor([1.0, 0.66, 0.33, 0.0])

        actual, _ = ANIMA_STREAM.sample_stream_batch(lambda x, sigma: x, latents, sigmas, microbatch_size=3)

        torch.testing.assert_close(actual, latents)

    def test_thousand_frames_keep_ring_bounded(self):
        latents = torch.zeros((1000, 1, 1))
        sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])

        actual, stats = ANIMA_STREAM.sample_stream_batch(fake_denoiser, latents, sigmas, microbatch_size=4)

        self.assertEqual(actual.shape[0], 1000)
        self.assertEqual(stats.ticks, 1003)
        self.assertEqual(stats.max_active_slots, 4)
        self.assertEqual(stats.max_forward_batch, 4)

    def test_eight_steps_are_split_into_microbatches_of_four(self):
        batches = []

        def recording_denoiser(x, sigma):
            batches.append(x.shape[0])
            return fake_denoiser(x, sigma)

        latents = torch.zeros((16, 1, 1))
        sigmas = torch.linspace(1.0, 0.0, 9)
        _, stats = ANIMA_STREAM.sample_stream_batch(recording_denoiser, latents, sigmas, microbatch_size=4)

        self.assertEqual(stats.max_active_slots, 8)
        self.assertEqual(stats.max_forward_batch, 4)
        self.assertLessEqual(max(batches), 4)

    def test_frame_callback_fires_in_output_order(self):
        completed = []
        latents = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)
        sigmas = torch.tensor([1.0, 0.66, 0.33, 0.0])

        actual, _ = ANIMA_STREAM.sample_stream_batch(
            lambda x, sigma: x,
            latents,
            sigmas,
            microbatch_size=3,
            frame_callback=lambda frame_id, latent: completed.append((frame_id, latent.clone())),
        )

        self.assertEqual([frame_id for frame_id, _ in completed], list(range(8)))
        torch.testing.assert_close(torch.cat([latent for _, latent in completed]), actual)

    def test_frames_per_tick_two_starts_with_two_latents(self):
        batches = []

        def recording_denoiser(x, sigma):
            batches.append(x.shape[0])
            return fake_denoiser(x, sigma)

        latents = torch.arange(6 * 2 * 3, dtype=torch.float32).reshape(6, 2, 3) / 10
        sigmas = torch.tensor([1.0, 0.72, 0.41, 0.18, 0.0])
        actual, stats = ANIMA_STREAM.sample_stream_batch(
            recording_denoiser,
            latents,
            sigmas,
            microbatch_size=8,
            frames_per_tick=2,
        )
        expected = sequential_sample(latents, sigmas)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(batches[0], 2)
        self.assertEqual(stats.max_active_slots, 6)
        self.assertEqual(stats.max_forward_batch, 6)

    def test_frames_per_tick_two_completes_pairs(self):
        completed = []
        latents = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        actual, stats = ANIMA_STREAM.sample_stream_batch(
            lambda x, sigma: x,
            latents,
            sigmas,
            microbatch_size=8,
            frames_per_tick=2,
            frame_callback=lambda frame_id, latent: completed.append(frame_id),
        )

        self.assertEqual(completed, list(range(8)))
        self.assertEqual(stats.max_active_slots, 4)
        torch.testing.assert_close(actual, latents)

    def test_stream_loop_keeps_the_ring_full_until_stop(self):
        completed = []
        batches = []
        pool = torch.arange(64, dtype=torch.float32).reshape(64, 1, 1)
        next_id = [0]

        def recording_denoiser(x, sigma):
            batches.append(x.shape[0])
            return x

        def make_latent(frame_id):
            latent = pool[next_id[0]:next_id[0] + 1]
            next_id[0] += 1
            return latent

        def should_stop():
            return len(completed) >= 8

        stats = ANIMA_STREAM.sample_stream_loop(
            recording_denoiser,
            make_latent,
            torch.tensor([1.0, 0.5, 0.0]),
            microbatch_size=8,
            frames_per_tick=2,
            frame_callback=lambda frame_id, latent: completed.append(frame_id),
            should_stop=should_stop,
        )

        self.assertGreaterEqual(len(completed), 8)
        self.assertEqual(completed, list(range(len(completed))))
        self.assertEqual(stats.max_active_slots, 4)
        self.assertEqual(max(batches), 4)
        self.assertGreater(next_id[0], 8)

    def test_nearest_complete_slot_prefers_the_oldest_last_stage(self):
        slots = [
            ANIMA_STREAM.StreamSlot(torch.zeros(1, 1, 1), 1, 4),
            ANIMA_STREAM.StreamSlot(torch.zeros(1, 1, 1), 3, 2),
            ANIMA_STREAM.StreamSlot(torch.zeros(1, 1, 1), 3, 1),
        ]
        slot = ANIMA_STREAM.nearest_complete_slot(slots, 4)
        self.assertEqual(slot.frame_id, 1)
        self.assertEqual(slot.stage_index, 3)
        self.assertIsNone(ANIMA_STREAM.nearest_complete_slot([], 4))

    def test_loop_tick_callback_sees_almost_done_slots(self):
        stages = []

        def tick_callback(active):
            slot = ANIMA_STREAM.nearest_complete_slot(active, 2)
            if slot is not None:
                stages.append(slot.stage_index)

        completed = []
        next_id = [0]

        ANIMA_STREAM.sample_stream_loop(
            lambda x, sigma: x,
            lambda frame_id: torch.zeros(1, 1, 1),
            torch.tensor([1.0, 0.5, 0.0]),
            microbatch_size=8,
            frames_per_tick=2,
            frame_callback=lambda frame_id, latent: completed.append(frame_id),
            should_stop=lambda: len(completed) >= 4,
            tick_callback=tick_callback,
        )
        self.assertIn(1, stages)

    def test_reset_pipeline_drops_in_flight_slots(self):
        completed = []
        reset_once = [False]
        did_reset = [False]

        def tick_callback(active):
            if len(completed) >= 2 and not did_reset[0]:
                reset_once[0] = True

        def reset_pipeline():
            if not reset_once[0]:
                return False
            reset_once[0] = False
            did_reset[0] = True
            return True

        ANIMA_STREAM.sample_stream_loop(
            lambda x, sigma: x,
            lambda frame_id: torch.zeros(1, 1, 1),
            torch.tensor([1.0, 0.5, 0.0]),
            microbatch_size=8,
            frames_per_tick=2,
            frame_callback=lambda frame_id, latent: completed.append(frame_id),
            should_stop=lambda: len(completed) >= 6,
            tick_callback=tick_callback,
            reset_pipeline=reset_pipeline,
        )
        self.assertTrue(did_reset[0])
        self.assertGreaterEqual(len(completed), 6)
        self.assertEqual(completed, sorted(completed))
        self.assertNotIn(2, completed)
        self.assertNotIn(3, completed)

    def test_loop_forwards_a_static_capacity_batch(self):
        batches = []
        completed = []

        ANIMA_STREAM.sample_stream_loop(
            lambda x, sigma: batches.append(x.shape[0]) or x,
            lambda frame_id: torch.zeros(1, 1, 1),
            torch.tensor([1.0, 0.5, 0.0]),
            microbatch_size=8,
            frames_per_tick=2,
            frame_callback=lambda frame_id, latent: completed.append(frame_id),
            should_stop=lambda: len(completed) >= 4,
        )
        self.assertGreaterEqual(len(completed), 4)
        self.assertEqual(set(batches), {4})


if __name__ == "__main__":
    unittest.main()
