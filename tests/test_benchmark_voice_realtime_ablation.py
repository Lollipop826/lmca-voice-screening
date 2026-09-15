"""Unit tests for measurement integrity; these are NOT experimental results."""
import base64
import unittest
from array import array
from scripts.benchmark_voice_realtime_ablation import (
    analyze_events, design, metrics, numeric_summary, cer, greeting_idle_guard,
)


class MeasurementIntegrityTests(unittest.TestCase):
    def test_full_factorial_paired_schedule(self):
        samples = [{'sample_id': str(i)} for i in range(4)]
        warmup, schedule = design(samples, 3, 20260913)
        self.assertEqual(len(warmup), 4)
        self.assertEqual(len(schedule), 48)
        self.assertEqual(len({x['arm'] for x in warmup}), 4)
        for b in range(1, 13):
            rows = [r for r in schedule if r['block_id'] == b]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({r['arm'] for r in rows}), 4)
            self.assertEqual(len({r['sample_id'] for r in rows}), 1)
        self.assertEqual((warmup, schedule), design(samples, 3, 20260913))

    def test_negative_times_are_not_clipped(self):
        m = metrics({'speech_end': 2, 'vad_end': 1, 'asr_final': 3})
        self.assertEqual(m['speech_end_to_vad_end_ms'], -1000)
        self.assertEqual(m['vad_end_to_asr_final_ms'], 2000)
        self.assertNotIn('speech_end_to_first_audio_ms', m)

    def test_event_notification_is_not_audio(self):
        payload = base64.b64encode(array('f', [.1, -.2]).tobytes()).decode()
        events = [
            (1.0, {'type': 'vad_end'}),
            (1.1, {'type': 'asr_result','turn_id': 'a','text': '你好'}),
            (1.2, {'type': 'tts_start','turn_id': 'a'}),
            (1.3, {'type': 'tts_chunk','turn_id': 'a','chunk': ''}),
            (1.4, {'type': 'tts_chunk','turn_id': 'b','chunk': payload}),
            (1.5, {'type': 'tts_chunk','turn_id': 'a','chunk': payload}),
        ]
        marks = {}
        result, audio = analyze_events(events, marks)
        self.assertEqual(marks['first_audio'], 1.5)
        self.assertEqual(marks['tts_start'], 1.2)
        self.assertEqual(len(audio), 8)
        self.assertEqual(result['observed_turn_ids'], ['a'])

    def test_multiple_turns_cannot_be_spliced_together(self):
        events = [(1, {'type': 'asr_result','turn_id': 'a','text': '你'}),
                  (2, {'type': 'asr_result','turn_id': 'b','text': '好'})]
        marks = {}
        result, audio = analyze_events(events, marks)
        self.assertEqual(result['observed_turn_ids'], ['a', 'b'])
        self.assertNotIn('asr_final', marks)
        self.assertFalse(audio)

    def test_nonempty_partial_and_text_only(self):
        events = [(1, {'type': 'asr_partial','text': ''}),
                  (2, {'type': 'asr_partial','text': '你','final': False}),
                  (3, {'type': 'asr_partial','text': '你好','final': True}),
                  (4, {'type': 'asr_result','text': '你好','turn_id': 'a'}),
                  (5, {'type': 'ai_response_chunk','text': ' ','turn_id': 'a'}),
                  (6, {'type': 'ai_response_chunk','text': '你好啊','turn_id': 'a'})]
        marks = {}
        result, _ = analyze_events(events, marks)
        self.assertEqual(marks['first_partial'], 2)
        self.assertEqual(marks['first_ai_text'], 6)
        self.assertEqual(result['partial_events'], 1)
        self.assertEqual(result['final_partial_events'], 1)
        self.assertEqual(result['final_partial_text'], '你好')

    def test_greeting_wait_includes_server_speaking_duration(self):
        self.assertEqual(greeting_idle_guard(9.5), 9.75)
        self.assertEqual(greeting_idle_guard(1), 1.25)
        for bad in [0, -1, float('inf'), float('nan'), 121]:
            with self.assertRaises(ValueError):
                greeting_idle_guard(bad)

    def test_metrics_form_a_nonoverlapping_chain(self):
        m = metrics({'speech_end': 1, 'vad_end': 2, 'asr_final': 2.5,
                     'first_ai_text': 3.5, 'first_audio': 4, 'tts_end': 8})
        chain = sum(m[k] for k in ['speech_end_to_vad_end_ms', 'vad_end_to_asr_final_ms',
                                  'asr_final_to_first_ai_text_ms', 'first_ai_text_to_first_audio_ms'])
        self.assertEqual(chain, m['speech_end_to_first_audio_ms'])
        self.assertEqual(m['first_ai_text_to_tts_end_ms'], 4500)

    def test_percentile_and_cer_conventions(self):
        self.assertEqual(numeric_summary(range(1,13))['p95_nearest_rank'], 12)
        self.assertEqual(numeric_summary([None, float('nan')]), {'n': 0})
        self.assertEqual(cer('你好，这是测试。', '你好这是测试！'), 0)
        self.assertAlmostEqual(cer('你好', '你们'), .5)


if __name__ == '__main__':
    unittest.main()
