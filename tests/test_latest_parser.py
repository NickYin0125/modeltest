from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _get_parser_types():
    pytest.importorskip("pytest_bdd")
    from feature_parsers.latest_parser import ZephyrOptimizedParser, ZephyrParserConfig

    return ZephyrOptimizedParser, ZephyrParserConfig


def _write_feature(tmp_path: Path, content: str) -> Path:
    feature_path = tmp_path / "sample.feature"
    feature_path.write_text(content, encoding="utf-8")
    return feature_path


def _parse_feature(tmp_path: Path, content: str, config):
    ZephyrOptimizedParser, _ = _get_parser_types()
    feature_path = _write_feature(tmp_path, content)
    return ZephyrOptimizedParser(feature_path.parent, feature_path.name, config)


def test_and_but_steps_follow_previous_type(tmp_path: Path) -> None:
    feature = """
Feature: And/But handling

  Scenario: Mixed steps
    Given base data
    And extra data
    When user performs action
    And user performs follow-up
    Then result is visible
    But audit log is updated
"""
    _, ZephyrParserConfig = _get_parser_types()
    parser = _parse_feature(tmp_path, feature, ZephyrParserConfig())
    cases = parser.build_zephyr_testcases_from_feature()
    steps = cases[0].test_steps

    assert [step.step_action for step in steps] == [
        "base data",
        "extra data",
        "user performs action",
        "user performs follow-up",
    ]
    assert steps[0].expected_result == "None"
    assert steps[1].expected_result == "None"
    assert steps[2].expected_result == "None"
    assert steps[3].expected_result == "1. result is visible\n\n2. audit log is updated"


def test_given_can_inherit_then_results(tmp_path: Path) -> None:
    feature = """
Feature: Given inherits Then

  Scenario: Given then
    Given precondition is set
    Then system is ready
    And validation passes
"""
    _, ZephyrParserConfig = _get_parser_types()
    parser = _parse_feature(tmp_path, feature, ZephyrParserConfig(given_inherit_then_results=True))
    cases = parser.build_zephyr_testcases_from_feature()
    steps = cases[0].test_steps

    assert len(steps) == 1
    assert steps[0].step_action == "precondition is set"
    assert steps[0].expected_result == "1. system is ready\n\n2. validation passes"


def test_when_default_expected_result_can_be_configured(tmp_path: Path) -> None:
    feature = """
Feature: When default result

  Scenario: When no Then
    When user navigates to dashboard
"""
    _, ZephyrParserConfig = _get_parser_types()
    config = ZephyrParserConfig(when_default_expected_result="None")
    parser = _parse_feature(tmp_path, feature, config)
    cases = parser.build_zephyr_testcases_from_feature()
    steps = cases[0].test_steps

    assert len(steps) == 1
    assert steps[0].step_action == "user navigates to dashboard"
    assert steps[0].expected_result == "None"


def test_write_bdd_csv_outputs_expected_columns(tmp_path: Path) -> None:
    pytest.importorskip("pytest_bdd")
    from feature_parsers.latest_parser import ZephyrOptimizedParser

    scripts = [
        ("Case A", "Feature: Sample\n\n  Scenario: A\n    Given step"),
        ("Case B", "Feature: Sample\n\n  Scenario: B\n    When action"),
    ]
    outfile = tmp_path / "bdd.csv"

    ZephyrOptimizedParser.write_testcases_to_zephyr_bdd_csv(scripts, str(outfile))

    content = outfile.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0] == "Name,Test Script (BDD)"
    assert "Case A" in lines[1]
    assert "Feature: Sample" in lines[1]
