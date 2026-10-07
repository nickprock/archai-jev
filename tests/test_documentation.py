"""The documented examples tell the truth about the default model.

Two promises are checked without the model file:

* the examples of the README that call ``ask`` are valid for the default model: every question
  in them is exactly one of the four tasks declared in the registry entry (otherwise a reader who
  copies the example gets an ``UnsupportedRequestError``);
* no example promises the case of the model card ("Ignore previous instructions..." is *safe*
  with probability 1.00), which the model does not do (spike S5).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from archai_jev import Choice, Jev, YesNo
from archai_jev.testing import MockScorer

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
REGISTRY_ENTRY = (
    ROOT / "src" / "models" / "registry" / "nickprock__archai-jev-qwen-1.5b.json"
)

# The sentence of the card, and any wording of it that a reader could take as the model's skill.
CARD_EXAMPLE = re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE)
INJECTION_PROMISE = re.compile(
    r"(ignore|disregard)\s+(all\s+|any\s+)?(previous|prior)\s+instructions",
    re.IGNORECASE,
)


def _tasks() -> list[dict[str, Any]]:
    entry = json.loads(REGISTRY_ENTRY.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = entry["tasks"]
    assert [t["id"] for t in tasks] == ["safety", "intent", "entailment", "similarity"]
    return tasks


def _task_of(question: Choice | YesNo) -> str | None:
    """The task a question is, by the exact rule of the spec (an independent reading)."""
    for task in _tasks():
        match = task["match"]["question"]
        if task["kind"] == "choice" and isinstance(question, Choice):
            options = match["options"]
            if (
                question.instructions == match["instructions"]
                and list(question.criteria) == [o["key"] for o in options]
                and all(
                    given in (None, "") or given == o.get("description")
                    for given, o in zip(
                        question.criteria.values(), options, strict=True
                    )
                )
            ):
                return str(task["id"])
        if (
            task["kind"] == "yes_no"
            and isinstance(question, YesNo)
            and dict(question.criteria)
            == {"true": match["true"], "false": match["false"]}
        ):
            return str(task["id"])
    return None


class _CheckedJev:
    """A ``Jev`` that answers with the mock scorer but, like the default model, refuses what is
    not one of its tasks (and remembers which tasks the examples used)."""

    def __init__(self) -> None:
        self._jev = Jev.from_scorer(MockScorer())
        self.used: list[str] = []

    @property
    def model_info(self) -> Any:
        return self._jev.model_info

    def ask(self, *args: Any, **kwargs: Any) -> Any:
        questions = kwargs.get("questions", args[1] if len(args) > 1 else {})
        for name, question in questions.items():
            task = _task_of(question)
            assert task is not None, f"README example: question {name!r} is not a task"
            self.used.append(task)
        return self._jev.ask(*args, **kwargs)


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.DOTALL | re.MULTILINE)


def test_the_readme_examples_run_and_are_tasks_of_the_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = _CheckedJev()
    monkeypatch.setattr(
        Jev, "from_pretrained", classmethod(lambda cls, *a, **k: checked)
    )
    blocks = [b for b in _python_blocks(README.read_text("utf-8")) if ".ask(" in b]
    assert len(blocks) >= 2, "the README must show the default model answering"
    namespace: dict[str, Any] = {}
    for block in blocks:
        exec(compile(block, "README.md", "exec"), namespace)
    assert {"safety", "entailment"} <= set(checked.used)


def test_the_checker_itself_refuses_what_is_not_a_task() -> None:
    # Guards the test above: it must not be able to pass for any question.
    assert _task_of(Choice("Which team?", ["returns", "billing"])) is None
    safety = Choice(
        "You are a System One decision engine for input safety.",
        {"SAFE": "Normal query", "UNSAFE": "Jailbreak, toxicity, injection"},
    )
    assert _task_of(safety) == "safety"
    assert _task_of(Choice(safety.instructions, ["UNSAFE", "SAFE"])) is None
    assert _task_of(YesNo("A condition.")) is None


def _documented_files() -> list[Path]:
    package = ROOT / "python" / "archai_jev"
    return [README, *sorted(package.glob("*.py")), *sorted(package.glob("*.pyi"))]


def test_the_card_example_is_not_promised_by_the_readme_or_the_docstrings() -> None:
    files = _documented_files()
    assert README in files and len(files) > 3
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not INJECTION_PROMISE.search(
            text
        ), f"{path.name} promises the card example"


def test_the_card_example_is_not_in_the_demos_either() -> None:
    demos = sorted((ROOT / "demos").glob("*.py"))
    assert demos
    for path in demos:
        text = path.read_text(encoding="utf-8")
        assert not CARD_EXAMPLE.search(text), f"{path.name} uses the card example"
