"""Harness-only unit checks; NOT experimental model results."""
import asyncio,base64,json,tempfile,unittest
from pathlib import Path
from scripts.benchmark_voice_sqlite_ablation import calc_metrics,wait_for_greeting,Recorder

class FakeSocket:
    def __init__(self,events=()): self.events=iter(events);self.sent=[]
    async def recv(self): return json.dumps(next(self.events))
    async def send(self,value): self.sent.append(value)

class TimingTests(unittest.TestCase):
    def test_stage_differences_not_cumulative(self):
        m=calc_metrics({'vad_end':1.1,'asr':1.5,'first_ai_event':2.0,'tts_start':2.2,'first_audio':2.7,'tts_end':4.0},1.0)
        self.assertEqual(m['input_to_asr_ms'],500)
        self.assertEqual(m['asr_to_first_ai_event_ms'],500)
        self.assertEqual(m['first_ai_to_tts_start_ms'],200)
        self.assertEqual(m['tts_start_to_first_audio_ms'],500)
        self.assertEqual(m['first_ai_to_first_audio_ms'],700)
        self.assertEqual(m['input_to_first_audio_ms'],1700)
    def test_no_audio_latency_from_empty_output(self):
        m=calc_metrics({'asr':2,'tts_start':3,'tts_end':4},1)
        self.assertNotIn('input_to_first_audio_ms',m)
        self.assertNotIn('tts_start_to_first_audio_ms',m)

class GreetingTests(unittest.IsolatedAsyncioTestCase):
    async def test_matching_greeting_audio_required(self):
        s=FakeSocket([{'type':'tts_start','turn_id':'g'},
                      {'type':'tts_chunk','turn_id':'g','audio':base64.b64encode(b'abcd').decode()},
                      {'type':'tts_end','turn_id':'g'}])
        r=await wait_for_greeting(s);self.assertEqual(r['audio_bytes'],4)
    async def test_empty_audio_is_failure(self):
        s=FakeSocket([{'type':'tts_start','turn_id':'g'}, {'type':'tts_chunk','turn_id':'g','audio':''}, {'type':'tts_end','turn_id':'g'}])
        with self.assertRaises(RuntimeError): await wait_for_greeting(s)
    async def test_other_turn_audio_not_accepted(self):
        s=FakeSocket([{'type':'tts_start','turn_id':'g'}, {'type':'tts_chunk','turn_id':'other','audio':'YWJjZA=='}, {'type':'tts_end','turn_id':'g'}])
        with self.assertRaises(RuntimeError): await wait_for_greeting(s)
    async def test_error_end_rejected(self):
        s=FakeSocket([{'type':'tts_start','turn_id':'g'}, {'type':'tts_chunk','turn_id':'g','audio':'YWJjZA=='}, {'type':'tts_end','turn_id':'g','reason':'error'}])
        with self.assertRaises(RuntimeError): await wait_for_greeting(s)
    async def test_frozen_profile_sent_for_existing_patient(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=FakeSocket();r=Recorder(s,Path(tmp)/'events.jsonl',profile={'name':'合成用户','age':68,'education_years':9})
            await r.send(json.dumps({'type':'start_session','patient_id':'synthetic-only','long_term_memory_writes_enabled':False}))
            sent=json.loads(s.sent[0]);r.close()
            self.assertEqual(sent['profile']['age'],68)
            self.assertEqual(sent['profile']['education_years'],9)
            self.assertFalse(sent['long_term_memory_writes_enabled'])
            self.assertEqual(sent['patient_id'],'synthetic-only')
if __name__=='__main__': unittest.main()
