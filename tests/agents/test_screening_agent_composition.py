"""Architecture tests for the composed screening-agent services."""

import asyncio
from unittest.mock import Mock, patch

from src.agents.screening.background_analysis import (
    ScreeningBackgroundAnalysis,
)
from src.agents.screening.answer_evaluation import ScreeningAnswerEvaluation
from src.agents.screening.conversation_policy import (
    ScreeningConversationPolicy,
)
from src.agents.screening.question_generation import (
    ScreeningQuestionGeneration,
)
from src.agents.screening.task_planning import ScreeningTaskPlanning
from src.agents.screening.tool_gateway import ScreeningToolGateway
from src.agents.screening.turn_pipeline import TurnPipeline
from src.agents.screening_agent_function_calling import (
    ADScreeningAgentFunctionCalling,
)


class _FakeMemoryTool:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _RecordingTool:
    def __init__(self):
        self.calls = []

    def _run(self, **kwargs):
        self.calls.append(kwargs)
        return "{}"


class _AsyncResponse:
    def __init__(self, content):
        self.content = content


class _BackgroundLLM:
    def __init__(self, before_return=None):
        self.before_return = before_return

    async def ainvoke(self, _messages):
        if self.before_return is not None:
            self.before_return()
        return _AsyncResponse('["orientation_time_year"]')


def _make_agent():
    with (
        patch.object(ScreeningToolGateway, "_init_tools"),
        patch.object(ADScreeningAgentFunctionCalling, "_log_summary_card"),
        patch.object(
            ADScreeningAgentFunctionCalling,
            "_create_interrupt_llm",
            return_value=(object(), "test"),
        ),
    ):
        agent = ADScreeningAgentFunctionCalling(use_local=True)
    agent.tool_gateway.memory_tool = _FakeMemoryTool()
    return agent


def test_agent_facade_has_no_behavior_mixins_or_state_magic():
    agent = _make_agent()
    assert type(agent).__mro__ == (
        ADScreeningAgentFunctionCalling,
        object,
    )
    assert "__getattr__" not in type(agent).__dict__
    assert "__setattr__" not in type(agent).__dict__


def test_pipeline_and_domain_collaborators_share_explicit_state():
    agent = _make_agent()

    assert isinstance(agent.turn_pipeline, TurnPipeline)
    assert isinstance(agent.conversation_policy, ScreeningConversationPolicy)
    assert isinstance(agent.answer_evaluator, ScreeningAnswerEvaluation)
    assert isinstance(agent.question_generator, ScreeningQuestionGeneration)
    assert ScreeningToolGateway not in type(agent).__mro__
    assert ScreeningTaskPlanning not in type(agent).__mro__
    assert ScreeningBackgroundAnalysis not in type(agent).__mro__
    assert agent.tool_gateway.state is agent.state
    assert agent.task_planner.state is agent.state
    assert agent.background_analysis.state is agent.state
    assert agent.conversation_policy.state is agent.state
    assert agent.answer_evaluator.state is agent.state
    assert agent.question_generator.state is agent.state
    assert agent.turn_pipeline.state is agent.state
    assert agent.turn_pipeline.tool_gateway is agent.tool_gateway
    assert agent.turn_pipeline.question_classifier is agent.question_classifier
    assert agent.turn_pipeline.session_lifecycle is agent.session_lifecycle
    assert agent.turn_pipeline.task_planner is agent.task_planner
    assert agent.turn_pipeline.policy is agent.conversation_policy
    assert agent.turn_pipeline.answer_evaluator is agent.answer_evaluator
    assert agent.turn_pipeline.question_generator is agent.question_generator
    assert agent.turn_pipeline.background_analysis is agent.background_analysis
    assert agent.tool_gateway.catalog is agent.catalog
    assert agent.task_planner.catalog is agent.catalog
    assert agent.background_analysis.catalog is agent.catalog
    assert agent.task_planner.use_local is True


def test_collaborators_are_mutated_through_their_explicit_interfaces():
    agent = _make_agent()
    replacement_memory = _FakeMemoryTool()
    selected_task = Mock(return_value="orientation_time_year")

    agent.tool_gateway.memory_tool = replacement_memory
    agent.task_planner._llm_select_task = selected_task

    assert agent.tool_gateway.memory_tool is replacement_memory
    assert agent.task_planner._llm_select_task is selected_task
    assert agent.task_planner._llm_select_task() == "orientation_time_year"


def test_session_lifecycle_resets_classifier_and_memory_state():
    agent = _make_agent()
    executor = Mock()
    agent.state._classification_executor = executor
    agent.state._pending_classification_task = object()
    agent.state._last_classification_result = "orientation_time_year"
    agent.state._available_candidates = ["orientation_time_year"]

    agent.session_lifecycle.reset("session-reset")

    executor.shutdown.assert_called_once_with(wait=False)
    assert agent.state.session_id == "session-reset"
    assert agent.state._classification_executor is None
    assert agent.state._pending_classification_task is None
    assert agent.state._last_classification_result is None
    assert agent.state._available_candidates == []
    assert agent.tool_gateway.memory_tool.reset_count == 1


def test_background_scoring_uses_the_bound_session_id():
    agent = _make_agent()
    score_tool = _RecordingTool()
    mmse_tool = _RecordingTool()
    agent.tool_gateway.score_tool = score_tool
    agent.tool_gateway.mmse_tool = mmse_tool
    agent.state.session_id = "session-real"
    agent.background_analysis._llm_factory = lambda _model: _BackgroundLLM()

    asyncio.run(
        agent.background_analysis.analyze(
            "",
            [
                {
                    "role": "user",
                    "content": "现在是二零二六年。",
                }
            ],
        )
    )

    assert score_tool.calls[0]["session_id"] == "session-real"
    assert mmse_tool.calls[0]["session_id"] == "session-real"


def test_background_result_is_discarded_after_session_switch():
    agent = _make_agent()
    score_tool = _RecordingTool()
    mmse_tool = _RecordingTool()
    agent.tool_gateway.score_tool = score_tool
    agent.tool_gateway.mmse_tool = mmse_tool
    agent.state.session_id = "session-old"
    agent.background_analysis._llm_factory = lambda _model: _BackgroundLLM(
        before_return=lambda: setattr(
            agent.state,
            "session_id",
            "session-new",
        )
    )

    asyncio.run(
        agent.background_analysis.analyze(
            "",
            [
                {
                    "role": "user",
                    "content": "现在是二零二六年。",
                }
            ],
        )
    )

    assert "orientation_time_year" not in agent.state._task_done
    assert score_tool.calls == []
    assert mmse_tool.calls == []
