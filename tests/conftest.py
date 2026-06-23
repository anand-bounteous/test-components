"""pytest plugin for the realworld test module.

Each JSON file in tests/realworld/scenarios/ becomes an independent
parametrized test case.  Three CLI flags control discovery and filtering:

  --rw-fixture=<glob>   run only fixtures whose stem matches the glob
                        (e.g. ``acme_jan_2025`` or ``acme_*``)
  --rw-tag=<tag>        run only fixtures that carry the tag in
                        scenario_tags (repeat for AND semantics)
  --rw-list             list all discovered fixtures (id, description,
                        tags), then exit

These flags are independent from the ``--scenario`` / ``--module`` flags
registered by tests/feature/conftest.py.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest
import yaml

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
STRESS_DIR = Path(__file__).resolve().parent / "scenarios_stress"


# ---------------------------------------------------------------------------
# CLI hooks
# ---------------------------------------------------------------------------


_HELP_TEXT = """\
Real-world scenario test module (tests/realworld/)
===================================================
One JSON file in tests/realworld/scenarios/ = one independent test case.
The full detect_salary_candidates pipeline runs end-to-end per fixture.
Evidence JSON is written to .plan/reviews/realworld-outputs/<stem>.json.

Flags
-----
  --rw-help                   Show this help and exit (does not override pytest -h)
  --rw-fixture=GLOB           Run fixtures whose stem matches GLOB (repeat = OR)
                              e.g. --rw-fixture=acme_* --rw-fixture=revolut_*
  --rw-tag=TAG                Run fixtures carrying TAG in scenario_tags (repeat = AND)
                              e.g. --rw-tag=bacs --rw-tag=monthly
  --rw-list                   List all discovered fixtures (id / tags / description), then exit

Standard pytest flags that compose well with the above
-------------------------------------------------------
  -k EXPR                     Substring / expression filter on fixture stems
  -v                          Verbose: show each test ID as it runs
  -s                          Disable capture — lets log records print live
  --log-cli-level=LEVEL       Stream log records (INFO = summaries, DEBUG = full audit trace)
  -m realworld                Select only realworld-marked tests
  -m "not realworld"          CI default: skip realworld tests

Standalone runner (no pytest needed)
-------------------------------------
  python -m tests.realworld.runner --help
  python -m tests.realworld.runner                      # run all
  python -m tests.realworld.runner acme_jan_2025        # by stem
  python -m tests.realworld.runner "acme_*"             # by glob
  python -m tests.realworld.runner --tag bacs           # by tag
  python -m tests.realworld.runner --debug acme_jan_2025  # full audit trace
  python -m tests.realworld.runner --list               # list fixtures

Fixture schema quick reference
-------------------------------
  analysis_id            (required) unique identifier
  scenario_description   human-readable purpose
  scenario_tags          list of lowercase tags used by --rw-tag
  validation_mode        "lenient" (default) or "strict"
  transactions[]         id / date / description / amount / direction
  expected               optional assertions block:
    candidate_count_min / candidate_count_max
    top_candidate_type / top_confidence_band
    must_contain_transaction_ids
    must_not_classify_as_probable_salary
    expected_warnings
"""


def pytest_addoption(parser):
    group = parser.getgroup("realworld", "Real-world scenario tests")
    group.addoption(
        "--rw-help",
        action="store_true",
        default=False,
        help="Show realworld test module help (flags, fixture schema, runner usage) and exit.",
    )
    group.addoption(
        "--rw-fixture",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Run only realworld fixtures whose stem matches GLOB "
            "(e.g. acme_jan_2025 or acme_*).  Repeat to include multiple globs."
        ),
    )
    group.addoption(
        "--rw-tag",
        action="append",
        default=[],
        metavar="TAG",
        help=(
            "Run only realworld fixtures that carry TAG in scenario_tags.  "
            "Repeat to require multiple tags (AND semantics)."
        ),
    )
    group.addoption(
        "--rw-list",
        action="store_true",
        default=False,
        help="List all discovered realworld fixtures, then exit.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "realworld: real-world scenario fixture test (one JSON file = one test)",
    )
    if config.getoption("--rw-help", default=False):
        print(_HELP_TEXT)
        pytest.exit("", returncode=0)
    if config.getoption("--rw-list", default=False):
        paths = sorted(SCENARIOS_DIR.glob("*.yaml"))
        if not paths:
            print("\nNo realworld fixtures found in", SCENARIOS_DIR)
        else:
            print(f"\nRealworld fixtures ({len(paths)} total):")
            for p in paths:
                try:
                    data = yaml.safe_load(p.read_text()) or {}
                    desc = data.get("scenario_description", "(no description)")
                    tags = data.get("scenario_tags", [])
                except Exception:
                    desc = "(unreadable)"
                    tags = []
                print(f"  {p.stem:40s}  {', '.join(tags) or '(no tags)':30s}  {desc}")
        pytest.exit("", returncode=0)


# ---------------------------------------------------------------------------
# Fixture discovery helpers
# ---------------------------------------------------------------------------


def _all_scenario_paths() -> list[Path]:
    paths = list(SCENARIOS_DIR.glob("*.yaml"))
    if STRESS_DIR.exists():
        paths.extend(STRESS_DIR.glob("*.yaml"))
    return sorted(paths)


def _load_scenario(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _filter_paths(paths: list[Path], config) -> list[Path]:
    fixture_globs: list[str] = config.getoption("--rw-fixture", default=[])
    required_tags: list[str] = config.getoption("--rw-tag", default=[])

    result: list[Path] = []
    for p in paths:
        if fixture_globs:
            if not any(fnmatch.fnmatch(p.stem, g) for g in fixture_globs):
                continue
        if required_tags:
            try:
                tags = set(_load_scenario(p).get("scenario_tags", []))
            except Exception:
                tags = set()
            if not all(t in tags for t in required_tags):
                continue
        result.append(p)
    return result


def _scenario_ids(paths: list[Path]) -> list[str]:
    return [p.stem for p in paths]


# ---------------------------------------------------------------------------
# Shared fixture: one Path per parametrize id
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    if "rw_scenario_path" in metafunc.fixturenames:
        paths = _all_scenario_paths()
        paths = _filter_paths(paths, metafunc.config)
        metafunc.parametrize(
            "rw_scenario_path",
            paths,
            ids=_scenario_ids(paths),
        )


# ---------------------------------------------------------------------------
# Terminal summary — grouped PASSED / FAILED / ERROR + stats bar
# ---------------------------------------------------------------------------

_RW_MODULE = "test_realworld"


def _rw_description(nodeid: str) -> str:
    """Extract the scenario stem from a nodeid like test_realworld.py[Reaworld_101_...]."""
    bracket = nodeid.find("[")
    if bracket != -1:
        return nodeid[bracket + 1 : -1]
    return nodeid


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    stats = terminalreporter.stats

    passed_reports = [r for r in stats.get("passed", []) if _RW_MODULE in r.nodeid]
    failed_reports = [r for r in stats.get("failed", []) if _RW_MODULE in r.nodeid]
    error_reports  = [r for r in stats.get("error",  []) if _RW_MODULE in r.nodeid]

    total = len(passed_reports) + len(failed_reports) + len(error_reports)
    if total == 0:
        return

    sep = "─" * 70
    thick = "═" * 70
    tw = terminalreporter

    tw.write_sep("=", "realworld scenario summary", bold=True)

    # ── PASSED ──
    tw.write_line(f"\n── PASSED ({len(passed_reports)}) {sep[:max(0, 52 - len(str(len(passed_reports))))]}")
    for r in passed_reports:
        stem = _rw_description(r.nodeid)
        # load description from yaml for extra context
        try:
            for base in (SCENARIOS_DIR, STRESS_DIR):
                p = base / (stem + ".yaml")
                if p.exists():
                    data = yaml.safe_load(p.read_text()) or {}
                    desc = data.get("scenario_description", "")[:60]
                    break
            else:
                desc = ""
        except Exception:
            desc = ""
        tw.write_line(f"  {stem:<50s}  {desc}")

    # ── FAILED ──
    tw.write_line(f"\n── FAILED ({len(failed_reports)}) {sep[:max(0, 52 - len(str(len(failed_reports))))]}")
    for r in failed_reports:
        stem = _rw_description(r.nodeid)
        try:
            for base in (SCENARIOS_DIR, STRESS_DIR):
                p = base / (stem + ".yaml")
                if p.exists():
                    data = yaml.safe_load(p.read_text()) or {}
                    desc = data.get("scenario_description", "")[:60]
                    break
            else:
                desc = ""
        except Exception:
            desc = ""
        tw.write_line(f"  {stem:<50s}  {desc}", red=True)
        # first non-empty line of longrepr as the failure detail
        detail = ""
        if r.longrepr:
            for line in str(r.longrepr).splitlines():
                line = line.strip()
                if line and not line.startswith("E   AssertionError"):
                    detail = line.lstrip("E").strip()
                    break
            if not detail:
                detail = str(r.longrepr).splitlines()[-1].strip()
        if detail:
            tw.write_line(f"  {' ' * 52}↳ {detail[:80]}", red=True)

    # ── ERRORS ──
    if error_reports:
        tw.write_line(f"\n── ERRORS ({len(error_reports)}) {sep[:max(0, 51 - len(str(len(error_reports))))]}")
        for r in error_reports:
            tw.write_line(f"  {_rw_description(r.nodeid)}", yellow=True)

    # ════ stats bar ════
    tw.write_line(f"\n{thick}")
    parts = [f"{len(passed_reports)} passed"]
    if failed_reports:
        parts.append(f"{len(failed_reports)} failed")
    if error_reports:
        parts.append(f"{len(error_reports)} error{'s' if len(error_reports) != 1 else ''}")
    parts.append(f"{total} total")
    tw.write_line("  " + ",  ".join(parts))
    tw.write_line(thick + "\n")
