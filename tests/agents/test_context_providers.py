import json
import unittest

from src.tools.agent_tools.conversation_memory_tool import ConversationMemoryTool
from src.context_management.providers import (
    ContextVariantConfigurationError,
    ContextVariantRegistry,
    RemoteContextMemoryTool,
    RemoteContextVariantConfig,
    UnknownContextVariantError,
)


class _BaselineMemory:
    def __init__(self):
        self.reset_count = 0
        self.suggestion = None

    def get_context(self, chat_history, agent_state=None):
        return {"summary": "baseline summary", "recent": list(chat_history)}

    def reset(self):
        self.reset_count += 1

    def get_discussed_topics(self):
        return ["baseline topic"]

    def get_discussed_topics_snapshot(self):
        return ["baseline topic"]

    def get_and_clear_suggestion(self):
        suggestion = self.suggestion
        self.suggestion = None
        return suggestion


class _FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, _limit):
        return self.payload


class ContextVariantRegistryTests(unittest.TestCase):
    def test_environment_registry_supports_simple_candidate_configuration(self):
        registry = ContextVariantRegistry.from_environment(
            {
                "CONTEXT_PROVIDER_URL": "http://context-service:8080/v1/context/build",
                "CONTEXT_PROVIDER_VARIANT": "candidate-v1",
                "CONTEXT_PROVIDER_API_KEY": "secret",
                "CONTEXT_PROVIDER_DEFAULT_VARIANT": "baseline",
            }
        )
        self.assertEqual(registry.available_variants, ["baseline", "candidate-v1"])
        self.assertEqual(registry.resolve(None), "baseline")
        self.assertEqual(registry.resolve("CANDIDATE-V1"), "candidate-v1")

    def test_unknown_variant_is_rejected(self):
        registry = ContextVariantRegistry()
        with self.assertRaises(UnknownContextVariantError):
            registry.resolve("candidate-v9")
        with self.assertRaises(ValueError):
            registry.resolve("http://attacker.invalid")

    def test_invalid_json_fails_closed_in_strict_mode(self):
        with self.assertRaises(ContextVariantConfigurationError):
            ContextVariantRegistry.from_environment(
                {"CONTEXT_PROVIDER_VARIANTS_JSON": "{"}
            )


class RemoteContextMemoryToolTests(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "name": "candidate-v1",
            "url": "http://context-service:8080/v1/context/build",
            "timeout_seconds": 0.5,
            "failure_threshold": 2,
            "cooldown_seconds": 10,
        }
        values.update(overrides)
        return RemoteContextVariantConfig(**values)

    def test_remote_bundle_is_adapted_to_existing_memory_contract(self):
        observed = {}

        def urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            observed["timeout"] = timeout
            return _FakeResponse(
                {
                    "summary": "患者喜欢钓鱼。",
                    "recent_messages": [
                        {"role": "user", "content": "我喜欢钓鱼"},
                    ],
                    "facts": [{"kind": "hobby", "value": "钓鱼"}],
                    "discussed_topics": ["兴趣爱好(钓鱼)"],
                    "asked_questions": ["平时喜欢做什么？"],
                    "next_task_suggestion": {"task_id": "orientation_time_year"},
                    "estimated_tokens": 88,
                    "version": "junior-2026-07-14",
                }
            )

        memory = RemoteContextMemoryTool(
            baseline_memory=_BaselineMemory(),
            config=self._config(),
            session_id="api_test",
            patient_profile={"age": 70},
            urlopen=urlopen,
        )
        memory.update_async([], {"candidates": ["orientation_time_year"]})
        context = memory.get_context(
            [{"role": "assistant", "content": "平时喜欢做什么？"}],
            {"task_done": {"persona_collect_1"}, "memory_words": ["苹果"]},
        )

        self.assertIn("患者喜欢钓鱼", context["summary"])
        self.assertIn("关键事实", context["summary"])
        self.assertIn("已完成任务", context["summary"])
        self.assertEqual(context["recent"][0]["content"], "我喜欢钓鱼")
        self.assertEqual(memory.get_discussed_topics(), ["兴趣爱好(钓鱼)"])
        self.assertEqual(
            memory.get_and_clear_suggestion()["task_id"], "orientation_time_year"
        )
        self.assertEqual(observed["body"]["session_id"], "api_test")
        self.assertEqual(observed["body"]["patient"]["age"], 70)
        self.assertEqual(observed["timeout"], 0.5)
        self.assertEqual(memory.get_diagnostics()["status"], "ok")

    def test_remote_failure_falls_back_and_opens_circuit(self):
        calls = {"count": 0}

        def failing_urlopen(_request, timeout):
            calls["count"] += 1
            raise TimeoutError("candidate timeout")

        now = {"value": 100.0}
        memory = RemoteContextMemoryTool(
            baseline_memory=_BaselineMemory(),
            config=self._config(),
            session_id="api_test",
            urlopen=failing_urlopen,
            monotonic=lambda: now["value"],
        )
        history = [{"role": "user", "content": "你好"}]

        first = memory.get_context(history, {})
        memory.update_async(history, {"candidates": ["x"]})
        second = memory.get_context(history, {})
        memory.update_async(history, {"candidates": ["y"]})
        third = memory.get_context(history, {})

        self.assertEqual(first["summary"], "baseline summary")
        self.assertEqual(second["summary"], "baseline summary")
        self.assertEqual(third["summary"], "baseline summary")
        self.assertEqual(calls["count"], 2)
        self.assertEqual(memory.get_diagnostics()["status"], "circuit_open")

    def test_identical_context_uses_cache(self):
        calls = {"count": 0}

        def urlopen(_request, timeout):
            calls["count"] += 1
            return _FakeResponse({"summary": "cached", "recent_messages": []})

        memory = RemoteContextMemoryTool(
            baseline_memory=_BaselineMemory(),
            config=self._config(),
            session_id="api_test",
            urlopen=urlopen,
        )
        memory.update_async([], {"candidates": ["orientation_time_year"]})
        memory.get_context([], {})
        memory.get_context([], {})
        self.assertEqual(calls["count"], 1)
        self.assertEqual(memory.get_diagnostics()["status"], "cache_hit")

    def test_persistent_background_is_prepended_and_sent_to_candidate(self):
        observed = {}

        def urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse({"summary": "候选摘要。", "recent_messages": []})

        memory = RemoteContextMemoryTool(
            baseline_memory=_BaselineMemory(),
            config=self._config(),
            session_id="api_test",
            urlopen=urlopen,
        )
        memory.set_persistent_background("【患者背景】上次得分27分，爱钓鱼。")
        context = memory.get_context([{"role": "assistant", "content": "您好"}], {})

        # 记忆卡拼在摘要最前面
        self.assertTrue(context["summary"].startswith("【患者背景】上次得分27分，爱钓鱼。"))
        self.assertIn("候选摘要。", context["summary"])
        # 记忆卡通过协议透传给候选服务
        self.assertEqual(observed["body"]["patient_memory"], "【患者背景】上次得分27分，爱钓鱼。")
        self.assertEqual(observed["body"]["schema_version"], "1.1")

    def test_persistent_background_survives_remote_fallback(self):
        def failing_urlopen(_request, timeout):
            raise TimeoutError("candidate timeout")

        memory = RemoteContextMemoryTool(
            baseline_memory=_BaselineMemory(),
            config=self._config(),
            session_id="api_test",
            urlopen=failing_urlopen,
        )
        memory.set_persistent_background("【患者背景】记忆卡内容。")
        context = memory.get_context([], {})
        # 候选失败回退 baseline 时，记忆卡仍由 baseline 侧承载（setter 已透传）
        self.assertTrue(memory.get_diagnostics()["fallback"])
        self.assertIn("记忆卡内容", context["summary"])


class ConversationMemoryWindowTests(unittest.TestCase):
    def test_context_keeps_four_bounded_messages_and_current_turn_background(self):
        memory = ConversationMemoryTool()
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index) * 600}
            for index in range(10)
        ]
        memory.set_turn_background("上一轮旧事")
        memory.set_turn_background("本轮相关旧事")

        context = memory.get_context(history)

        self.assertEqual(len(context["recent"]), 4)
        self.assertEqual(context["recent"][0]["content"], "6" * 500)
        self.assertTrue(all(len(item["content"]) <= 500 for item in context["recent"]))
        self.assertIn("本轮相关旧事", context["summary"])
        self.assertNotIn("上一轮旧事", context["summary"])

        memory.reset()
        self.assertNotIn("本轮相关旧事", memory.get_context([])["summary"])


if __name__ == "__main__":
    unittest.main()
