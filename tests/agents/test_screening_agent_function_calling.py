"""Characterization tests for the default function-calling screening agent.

The real tool constructors may load models or configure API clients.  Tests
replace that bootstrap boundary, then exercise the agent's existing session,
routing, scoring, and comfort-mode behavior without network or model access.
"""

import time
import unittest
from unittest.mock import Mock, patch

from src.agents.screening.tool_gateway import ScreeningToolGateway
from src.agents.screening_agent_function_calling import (
    ADScreeningAgentFunctionCalling,
)


class _FakeMemoryTool:
    def __init__(self):
        self.reset_count = 0
        self.suggestion = None
        self.updates = []

    def reset(self):
        self.reset_count += 1

    def get_and_clear_suggestion(self):
        suggestion = self.suggestion
        self.suggestion = None
        return suggestion

    def update_async(self, chat_history, task_context=None):
        self.updates.append((list(chat_history), task_context))


def _make_agent():
    pooled_llm = object()
    with (
        patch.object(ScreeningToolGateway, "_init_tools"),
        patch.object(ADScreeningAgentFunctionCalling, "_log_summary_card"),
        patch.object(
            ADScreeningAgentFunctionCalling,
            "_create_interrupt_llm",
            return_value=(pooled_llm, "test"),
        ),
    ):
        agent = ADScreeningAgentFunctionCalling(use_local=True)

    agent.tool_gateway.memory_tool = _FakeMemoryTool()
    return agent, pooled_llm


class AgentSessionStateTests(unittest.TestCase):
    def test_initialization_establishes_isolated_session_state(self):
        first, pooled_llm = _make_agent()
        second, _ = _make_agent()

        self.assertIs(first.llm, pooled_llm)
        self.assertIsNone(first.state.session_id)
        self.assertEqual(first.state._task_done, set())
        self.assertEqual(first.state._task_attempts, {})
        self.assertEqual(first.state._asked_questions, [])
        self.assertEqual(first.state._consent_granted_groups, set())
        self.assertEqual(
            first.state.session_data,
            {"memory_words": None, "calculation_config": None},
        )

        first.state._task_done.add("persona_collect_1")
        first.state.session_data["memory_words"] = ["苹果", "桌子", "硬币"]
        self.assertEqual(second.state._task_done, set())
        self.assertIsNone(second.state.session_data["memory_words"])

    def test_session_reset_clears_all_routing_state_and_memory(self):
        agent, _ = _make_agent()
        agent.state._last_task_id = "orientation_time_year"
        agent.state._task_done = {"persona_collect_1", "orientation_time_year"}
        agent.state._task_attempts = {"orientation_time_year:unable": 2}
        agent.state._asked_questions = ["旧问题"]
        agent.state._used_chat_topics = ["旧话题"]
        agent.state._pending_consent_task_id = "registration_3words"
        agent.state.session_data["memory_words"] = ["苹果", "桌子", "硬币"]
        agent.state.session_data["calculation_config"] = {"start": 100}

        agent.session_lifecycle.reset("session-new")

        self.assertEqual(agent.state._active_session_id, "session-new")
        self.assertEqual(agent.state._last_generated_question, "请开始评估")
        self.assertEqual(agent.state._task_done, set(agent.catalog.visual_action_tasks))
        self.assertEqual(agent.state._task_attempts, {})
        self.assertEqual(agent.state._asked_questions, [])
        self.assertEqual(agent.state._used_chat_topics, [])
        self.assertIsNone(agent.state._pending_consent_task_id)
        self.assertIsNone(agent.state.session_data["memory_words"])
        self.assertIsNone(agent.state.session_data["calculation_config"])
        self.assertEqual(agent.tool_gateway.memory_tool.reset_count, 1)
        self.assertEqual(agent.state.dimension_index, 0)
        self.assertEqual(agent.state.current_dimension["id"], "orientation")


class AgentTaskRoutingTests(unittest.TestCase):
    def setUp(self):
        self.agent, _ = _make_agent()
        self.agent.session_lifecycle.reset("routing-session")

    def test_candidate_pool_enforces_stage_and_recall_prerequisites(self):
        fresh_candidates = self.agent.task_planner._get_valid_next_task_candidates(now=1000.0)

        self.assertIn("persona_collect_1", fresh_candidates)
        self.assertIn("orientation_time_year", fresh_candidates)
        self.assertNotIn("registration_3words", fresh_candidates)
        self.assertNotIn("recall_3words", fresh_candidates)
        self.assertNotIn("attention_calc_life_math", fresh_candidates)

        basic_prerequisites = {
            "persona_collect_1",
            "persona_collect_2",
            "orientation_time_year",
            "orientation_time_season",
            "orientation_time_month_date",
            "orientation_time_weekday",
            "orientation_place_province_city",
            "orientation_place_district",
            "orientation_place_location_floor",
        }
        self.agent.state._task_done = set(basic_prerequisites)
        after_basics = self.agent.task_planner._get_valid_next_task_candidates(now=1000.0)
        self.assertIn("registration_3words", after_basics)
        self.assertNotIn("recall_3words", after_basics)

        self.agent.state._task_done.add("registration_3words")
        self.agent.state._registration_ts = 1000.0
        self.assertNotIn(
            "recall_3words",
            self.agent.task_planner._get_valid_next_task_candidates(now=1119.9),
        )
        self.assertIn(
            "recall_3words",
            self.agent.task_planner._get_valid_next_task_candidates(now=1120.0),
        )

    def test_memory_suggestion_wins_when_it_is_a_valid_candidate(self):
        self.agent.tool_gateway.memory_tool.suggestion = {
            "task_id": "orientation_time_year",
            "from_topic": "天气",
            "to_topic": "年份",
            "anchor_fact": "患者刚提到今年天气热",
            "bridge_hint": "从今年的天气聊到当前年份",
        }
        self.agent.task_planner._llm_select_task = Mock(return_value="persona_collect_1")

        selected = self.agent.task_planner._select_next_task()

        self.assertEqual(selected, "orientation_time_year")
        self.assertEqual(
            self.agent.state._last_bridge_hint,
            "从今年的天气聊到当前年份",
        )
        self.assertEqual(self.agent.state._last_bridge_topic, "年份")
        self.assertTrue(self.agent.state._current_turn_topic_set)
        self.agent.task_planner._llm_select_task.assert_not_called()

    def test_cognitive_task_inserts_a_bounded_buffer_turn(self):
        self.agent.state._last_task_id = "orientation_time_year"
        self.agent.state._task_done.add("orientation_time_year")

        selected = self.agent.task_planner._select_next_task()

        self.assertEqual(selected, "buffer_chat")
        self.assertEqual(self.agent.state._consecutive_buffer_count, 1)

    def test_waiting_for_delayed_recall_routes_to_buffer_chat(self):
        self.agent.state._task_done = set(self.agent.catalog.required_tasks) - {"recall_3words"}
        self.agent.state._registration_ts = time.time()
        self.agent.state._last_task_id = None

        self.assertEqual(self.agent.task_planner._select_next_task(), "buffer_chat")


class AgentAnswerHandlingTests(unittest.TestCase):
    def setUp(self):
        self.agent, _ = _make_agent()

    def test_invalid_answer_classification_and_attempt_tracking(self):
        self.assertEqual(
            self.agent.conversation_policy._invalid_answer_kind(
                "请再说一遍",
                {
                    "wants_repeat": True,
                    "has_substantive_answer": False,
                    "is_resistant": False,
                },
            ),
            "repeat_request",
        )
        self.assertEqual(self.agent.conversation_policy._invalid_answer_kind("我不想回答"), "refusal")
        self.assertEqual(self.agent.conversation_policy._invalid_answer_kind("想不起来"), "unable")
        self.assertIsNone(self.agent.conversation_policy._invalid_answer_kind("今年是二零二六年"))

        task_id = "orientation_time_year"
        self.assertEqual(self.agent.conversation_policy._register_invalid_attempt(task_id, "unable"), 1)
        self.assertEqual(self.agent.conversation_policy._register_invalid_attempt(task_id, "unable"), 2)
        self.assertEqual(
            self.agent.conversation_policy._register_invalid_attempt("buffer_chat", "unable"),
            0,
        )
        self.agent.conversation_policy._clear_invalid_attempts(task_id)
        self.assertEqual(self.agent.state._task_attempts, {})

    def test_rule_based_evaluation_handles_calculation_time_and_recall(self):
        calculation = self.agent.answer_evaluator._try_rule_based_evaluation(
            "attention_calc_life_math",
            "100减7是93、86、79、72、65",
            None,
        )
        self.assertTrue(calculation["is_correct"])
        self.assertEqual(calculation["raw_score"], 5)
        self.assertEqual(calculation["raw_max_score"], 5)
        self.assertEqual(calculation["quality_level"], "excellent")

        year = self.agent.answer_evaluator._try_rule_based_evaluation(
            "orientation_time_year",
            "现在是二零二六年",
            "2026年",
        )
        self.assertTrue(year["is_correct"])
        self.assertEqual(year["confidence"], 1.0)

        self.agent.state.session_data["memory_words"] = ["苹果", "桌子", "硬币"]
        recall = self.agent.answer_evaluator._try_rule_based_evaluation(
            "recall_3words",
            "我记得苹果和硬币",
            None,
        )
        self.assertTrue(recall["is_correct"])
        self.assertEqual(recall["quality_level"], "good")
        self.assertIn("2/3", recall["evaluation_detail"])

    def test_mmse_score_mapping_and_risk_boundaries_are_stable(self):
        self.assertEqual(
            self.agent.answer_evaluator._convert_quality_to_mmse_score(
                "orientation", "excellent", "正常", max_score_override=5
            ),
            {"score": 5, "max_score": 5},
        )
        self.assertEqual(
            self.agent.answer_evaluator._convert_quality_to_mmse_score(
                "orientation", "fair", "中度异常", max_score_override=5
            ),
            {"score": 2, "max_score": 5},
        )
        self.assertEqual(self.agent.answer_evaluator._calculate_alzheimers_risk(24)["severity"], "normal")
        self.assertEqual(self.agent.answer_evaluator._calculate_alzheimers_risk(18)["severity"], "mild")
        self.assertEqual(self.agent.answer_evaluator._calculate_alzheimers_risk(10)["severity"], "moderate")
        self.assertEqual(self.agent.answer_evaluator._calculate_alzheimers_risk(9)["severity"], "severe")


class AgentComfortFlowTests(unittest.TestCase):
    def setUp(self):
        self.agent, _ = _make_agent()
        self.agent.session_lifecycle.reset("comfort-session")

    def test_goodbye_finishes_only_after_all_core_tasks_are_complete(self):
        core_tasks = {
            task
            for task in self.agent.catalog.required_tasks
            if task not in self.agent.catalog.buffer_tasks
        }
        self.agent.state._task_done = set(core_tasks)
        self.assertTrue(self.agent.conversation_policy._check_mmse_complete("comfort-session"))
        self.agent.state._task_done.remove("orientation_time_year")
        self.assertFalse(self.agent.conversation_policy._check_mmse_complete("comfort-session"))

    def test_completed_comfort_session_can_return_a_farewell(self):
        self.agent.state.is_in_comfort_mode = True
        self.agent.state._last_task_id = "buffer_chat"
        self.agent.conversation_policy._check_mmse_complete = Mock(return_value=True)

        result = self.agent.process_turn(
            "那今天就到这儿吧，再见",
            session_id="comfort-session",
            patient_profile={"name": "张阿姨"},
            chat_history=[],
            current_emotion="neutral",
        )

        self.assertTrue(result["is_goodbye"])
        self.assertTrue(result["is_comfort_mode"])
        self.assertIn("张阿姨", result["output"])
        self.assertEqual(self.agent.state._last_generated_question, result["output"])
        self.assertEqual(len(self.agent.tool_gateway.memory_tool.updates), 1)

    def test_substantive_comfort_reply_resumes_the_interrupted_task(self):
        self.agent.state.is_in_comfort_mode = True
        self.agent.state._comfort_entry_category = "refusal"
        self.agent.state._comfort_interrupted_task_id = "orientation_time_year"
        self.agent.state._last_task_id = "buffer_chat"
        expected_result = {
            "output": "我们接着聊聊今年是哪一年。",
            "response": "我们接着聊聊今年是哪一年。",
        }
        self.agent.question_generator._generate_assessment_question = Mock(return_value=expected_result)

        result = self.agent.process_turn(
            "我刚才有点紧张，现在好多了",
            session_id="comfort-session",
            patient_profile={"name": "张阿姨"},
            chat_history=[],
            current_emotion="neutral",
        )

        self.assertEqual(result, expected_result)
        self.assertFalse(self.agent.state.is_in_comfort_mode)
        self.assertIsNone(self.agent.state._comfort_interrupted_task_id)
        self.agent.question_generator._generate_assessment_question.assert_called_once()
        self.assertEqual(
            self.agent.question_generator._generate_assessment_question.call_args.kwargs["task_id"],
            "orientation_time_year",
        )
        self.assertEqual(len(self.agent.tool_gateway.memory_tool.updates), 1)


if __name__ == "__main__":
    unittest.main()
