"""Write eval reports: full JSON for machines, a Markdown summary for humans and PRs."""

from __future__ import annotations

from pathlib import Path

from task_agent.evals.runner import EvalReport, Metrics, Outcome


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def _num(x: float | None, fmt: str = "{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


def _row(name: str, m: Metrics) -> str:
    attempted = m.total - m.skipped
    return (
        f"| {name} | {m.passed}/{attempted} | {_pct(m.success_rate)} | {m.errors} | {m.skipped} "
        f"| {_num(m.cost_usd_mean, '${:.4f}')} | {_num(m.steps_mean, '{:.1f}')} "
        f"| {_num(m.wall_s_mean, '{:.1f}s')} |"
    )


def to_markdown(report: EvalReport) -> str:
    lines = [
        f"# Eval run `{report.run_id}`" + (f" — {report.label}" if report.label else ""),
        "",
        f"- runner: `{report.runner}` · agent {report.agent_version}",
        f"- tool groups available: {', '.join(report.available_groups) or 'none'}",
        f"- total cost: ${report.overall.cost_usd_total:.4f}"
        f" (+ ${report.judge_cost_usd:.4f} judging)",
        "",
        "| category | passed | success | errors | skipped | cost/task | steps | wall |",
        "|---|---|---|---|---|---|---|---|",
        *(_row(c, m) for c, m in report.by_category.items()),
        _row("**overall**", report.overall),
        "",
    ]
    problems = [r for r in report.results if r.outcome in (Outcome.FAILED, Outcome.ERROR)]
    if problems:
        lines += ["## Failures", "", "| task | outcome | answer | detail |", "|---|---|---|---|"]
        for r in problems:
            answer = (r.answer or "").replace("\n", " ").replace("|", "\\|")[:80]
            detail = r.detail.replace("\n", " ").replace("|", "\\|")[:120]
            lines.append(f"| `{r.task_id}` | {r.outcome.value} | {answer} | {detail} |")
        lines.append("")
    return "\n".join(lines)


def write_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{report.run_id}.json"
    md_path = out_dir / f"{report.run_id}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path
