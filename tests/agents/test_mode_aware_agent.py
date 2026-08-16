import pytest

from src.agents.mode_aware_agent import ModeAwareAgent


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def process_turn(self, *_args, **_kwargs):
        return {"output": self.name}


def _factory(name: str, created: list[str]):
    def build(*_args, **_kwargs):
        created.append(name)
        return _FakeAgent(name)

    return build


def test_mode_aware_agent_refuses_mid_session_switch_but_resets_for_new_session():
    created: list[str] = []
    agent = ModeAwareAgent(
        cognitive_factory=_factory("cognitive", created),
        wellbeing_factory=_factory("wellbeing", created),
        mode="wellbeing",
    )

    assert agent.process_turn("你好")["output"] == "wellbeing"
    with pytest.raises(RuntimeError):
        agent.set_mode("cognitive_screening")

    agent.reset_for_session("cognitive_screening")
    assert agent.process_turn("开始专项")["output"] == "cognitive"

    agent.reset_for_session("wellbeing")
    assert agent.process_turn("回到陪伴")["output"] == "wellbeing"
    assert created == ["wellbeing", "cognitive", "wellbeing"]
