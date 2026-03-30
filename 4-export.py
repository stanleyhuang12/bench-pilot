"""
4-export.py — Phase 4: Compile all benchmark data into a results.docx report.

Usage:
    python 4-export.py
    python 4-export.py --config path/to/config.json
    python 4-export.py --results-root results --benchmark emotional-dependency
    python 4-export.py --output report.docx
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Twips
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from config import load_config, get_model_name


class C:
    ACCENT = RGBColor(0x2E, 0x75, 0xB6)
    ACCENT_DARK = RGBColor(0x1F, 0x54, 0x96)
    PASS = RGBColor(0x1E, 0x84, 0x49)
    FAIL = RGBColor(0xC0, 0x39, 0x2B)
    HEADER_BG = "2E75B6"
    ROW_ALT = "EBF3FB"
    WHITE = "FFFFFF"
    MID_GREY = RGBColor(0x66, 0x66, 0x66)


# ---------------------------------------------------------------------------
# Low-level XML helpers
# ---------------------------------------------------------------------------

def _set_cell_bg(cell, hex_color: str) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _set_cell_margins(cell, top=80, bottom=80, left=120, right=120) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.append(mar)


def _add_horizontal_rule(doc: Document, color: str = "2E75B6", size: int = 6) -> None:
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pb = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    pb.append(bottom)
    pPr.append(pb)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)


def _add_page_break(doc: Document) -> None:
    doc.add_paragraph().add_run().add_break()


def _heading(doc: Document, text: str, level: int = 1) -> None:
    colors = {1: C.ACCENT_DARK, 2: C.ACCENT, 3: C.MID_GREY}
    sizes = {1: 20, 2: 16, 3: 13}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(sizes.get(level, 12))
    run.font.color.rgb = colors.get(level, C.ACCENT)


def _body(doc: Document, text: str, italic: bool = False, color: RGBColor | None = None) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.size = Pt(10)
    run.italic = italic
    if color:
        run.font.color.rgb = color


def _label_value(doc: Document, label: str, value: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    r1 = p.add_run(f"{label}: ")
    r1.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = C.MID_GREY
    r2 = p.add_run(value)
    r2.font.size = Pt(10)


def _make_table(doc: Document, col_widths_inches: list[float]):
    col_widths_dxa = [int(w * 1440) for w in col_widths_inches]
    table = doc.add_table(rows=0, cols=len(col_widths_dxa))
    table.style = "Table Grid"
    return table, col_widths_dxa


def _header_row(table, col_widths_dxa: list[int], labels: list[str]) -> None:
    row = table.add_row()
    for cell, label, w in zip(row.cells, labels, col_widths_dxa):
        _set_cell_bg(cell, C.HEADER_BG)
        _set_cell_margins(cell)
        cell.width = Twips(w)
        run = cell.paragraphs[0].add_run(label)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def _data_row(
    table,
    col_widths_dxa: list[int],
    values: list[str],
    alt: bool = False,
    bold_first: bool = False,
    verdict_col: int | None = None,  # column index coloured green/red by "yes"/"no"
) -> None:
    row = table.add_row()
    bg = C.ROW_ALT if alt else C.WHITE
    for i, (cell, val, w) in enumerate(zip(row.cells, values, col_widths_dxa)):
        _set_cell_bg(cell, bg)
        _set_cell_margins(cell)
        cell.width = Twips(w)
        run = cell.paragraphs[0].add_run(str(val))
        run.font.size = Pt(9)
        if i == 0 and bold_first:
            run.bold = True
        if verdict_col is not None and i == verdict_col:
            low = str(val).lower()
            run.font.color.rgb = C.PASS if low == "yes" else C.FAIL
            run.bold = True


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _resolve_benchmarks(results_root: str, test_path: str, benchmark: str | None) -> list[str]:
    entries = os.listdir(results_root)
    if benchmark:
        if benchmark not in entries:
            raise FileNotFoundError(f"Benchmark '{benchmark}' not found in {results_root}")
        return [os.path.join(results_root, benchmark, test_path)]
    return [os.path.join(results_root, b, test_path) for b in entries]


def _load_benchmark(bench: str, conv_dir_name: str, results_filename: str) -> dict:
    bench_dir = os.path.dirname(bench)
    bench_name = os.path.basename(bench_dir)
    conv_path = os.path.join(bench_dir, conv_dir_name)
    results_path = os.path.join(bench_dir, results_filename)

    for path, label in [
        (bench, "generate.py"),
        (conv_path, "simulate.py"),
        (results_path, "evaluate.py"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} — run {label} first.")

    with open(bench) as f:
        test_data = json.load(f)

    conversations = {}
    for fname in sorted(os.listdir(conv_path)):
        if fname.endswith(".json"):
            with open(os.path.join(conv_path, fname)) as f:
                conv = json.load(f)
                sid = conv.get("scenario_id") or fname.replace(".json", "")
                conversations[sid] = conv

    with open(results_path) as f:
        results = json.load(f)

    return {
        "name": bench_name,
        "test_data": test_data,
        "conversations": conversations,
        "results": results,
    }


def _load_goal_content(config: dict) -> str:
    goal_path = config.get("paths", {}).get("goal_prompt", "")
    if goal_path and os.path.exists(goal_path):
        with open(goal_path) as f:
            return f.read().strip()
    return "(goal file not found)"


def _num_iterations(bench: dict) -> int:
    """Infer iteration count from the length of the results list in any detail record."""
    for d in bench["results"].get("details", []):
        r = d.get("results", [])
        if isinstance(r, list) and r:
            return len(r)
    return 1


# ---------------------------------------------------------------------------
# Detail record accessors
#
# Each detail record (from evaluate.py) has:
#   "results":       ["yes"|"no", ...]   — one per iteration
#   "justifications": [str, ...]          — one per iteration
# ---------------------------------------------------------------------------

def _detail_result(detail: dict, sample: int = 0) -> str:
    results = detail.get("results", [])
    if isinstance(results, list) and sample < len(results):
        return str(results[sample]).lower()
    if isinstance(results, str):
        return results.lower()
    return "—"


def _detail_justification(detail: dict, sample: int = 0) -> str:
    justs = detail.get("justifications", [])
    if isinstance(justs, list) and sample < len(justs):
        return justs[sample]
    if isinstance(justs, str):
        return justs
    return detail.get("justification", "")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _write_cover(doc, bench, config, user_model, target_model) -> None:
    test_data = bench["test_data"]
    n_iter = _num_iterations(bench)

    doc.add_paragraph()
    _add_horizontal_rule(doc, color="2E75B6", size=12)

    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(8)
    title_p.paragraph_format.space_after = Pt(4)
    run = title_p.add_run(
        test_data.get("benchmark_name", bench["name"].replace("-", " ").title())
    )
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = C.ACCENT_DARK

    sub_p = doc.add_paragraph()
    sub_r = sub_p.add_run("AI Conversation Evaluation Report")
    sub_r.font.size = Pt(13)
    sub_r.font.color.rgb = C.MID_GREY
    sub_r.italic = True

    _add_horizontal_rule(doc)
    doc.add_paragraph()

    _label_value(doc, "Description",            test_data.get("description", "—"))
    _label_value(doc, "User model",             user_model)
    _label_value(doc, "Target model",           target_model)
    _label_value(doc, "Generated",              datetime.now().strftime("%B %d, %Y"))
    _label_value(doc, "Scenarios",              str(len(bench["conversations"])))
    _label_value(doc, "Metrics",               str(len(test_data.get("metrics", []))))
    _label_value(doc, "Iterations per scenario", str(n_iter))

    doc.add_paragraph()


def _write_model_and_prompts(doc, config, goal_content) -> None:
    _heading(doc, "Model Configuration & Prompts", level=1)
    _add_horizontal_rule(doc)

    # Models table
    _heading(doc, "Models", level=2)
    models = config.get("models", {})
    table, widths = _make_table(doc, [1.5, 3.5, 4.0])
    _header_row(table, widths, ["Role", "Provider / Model", "Notes"])
    for i, (role, label) in enumerate(
        {"generator": "Generator", "user": "User Simulator", "target": "Target"}.items()
    ):
        m = models.get(role, {})
        _data_row(
            table, widths,
            [label, f"{m.get('provider', '—')} / {m.get('model', '—')}", m.get("notes", "")],
            alt=i % 2 == 1,
        )
    doc.add_paragraph()

    # Generation parameters
    _heading(doc, "Generation Parameters", level=2)
    gen = config.get("generation", {})
    paths = config.get("paths", {})
    table2, widths2 = _make_table(doc, [3.0, 6.36])
    _header_row(table2, widths2, ["Parameter", "Value"])
    for i, (k, v) in enumerate([
        ("Scenarios per batch",    str(gen.get("num_scenarios", "—"))),
        ("Turns per conversation", str(gen.get("turns_per_conversation", "—"))),
        ("Goal prompt file",       paths.get("goal_prompt", "—")),
        ("Test file",              paths.get("test_file", "—")),
        ("Conversations dir",      paths.get("conversations_dir", "—")),
        ("Results file",           paths.get("results_file", "—")),
    ]):
        _data_row(table2, widths2, [k, v], alt=i % 2 == 1)
    doc.add_paragraph()


    # Goal prompt file
    _heading(doc, "Goal Prompt (goal.json / goal.md)", level=2)
    _body(
        doc,
        "Raw content of the goal prompt file used to generate scenarios and metrics.",
        italic=True,
        color=C.MID_GREY,
    )
    p2 = doc.add_paragraph()
    p2.paragraph_format.left_indent = Inches(0.2)
    p2.paragraph_format.space_before = Pt(2)
    p2.paragraph_format.space_after = Pt(6)
    run2 = p2.add_run(goal_content)
    run2.font.name = "Courier New"
    run2.font.size = Pt(8)
    run2.font.color.rgb = C.MID_GREY
    doc.add_paragraph()


def _write_metrics_overview(doc, bench) -> None:
    metrics = bench["test_data"].get("metrics", [])
    _heading(doc, "Evaluated Metrics", level=1)
    _add_horizontal_rule(doc)

    table, widths = _make_table(doc, [1.2, 1.8, 4.36, 2.0])
    _header_row(table, widths, ["ID", "Name", "Description", "Applies To"])
    for i, m in enumerate(metrics):
        applies = m.get("applies_to", "all")
        applies_str = "All scenarios" if applies == "all" else ", ".join(applies)
        _data_row(
            table, widths,
            [m.get("id", ""), m.get("name", ""), m.get("description", ""), applies_str],
            alt=i % 2 == 1,
        )
    doc.add_paragraph()


def _write_aggregate(doc, bench) -> None:
    """
    Uses pre-computed fields from results.json:
      summary.yes_rate / .yes / .total
      by_metric[mid].yes_rate / .yes / .valid / .percent_agreement
      by_scenario[sid].yes_rate / .yes / .valid / .percent_agreement
    """
    results = bench["results"]
    s = bench["results"]["summary"]

    _heading(doc, "Aggregate Results", level=1)
    _add_horizontal_rule(doc)

    # Progress bar
    pct = s.get("yes_rate", 0.0)
    bar_p = doc.add_paragraph()
    bar_p.paragraph_format.space_after = Pt(6)
    filled = round(pct * 20)
    bar_r = bar_p.add_run("█" * filled)
    bar_r.font.color.rgb = C.PASS
    bar_r.font.size = Pt(14)
    empty_r = bar_p.add_run("█" * (20 - filled))
    empty_r.font.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    empty_r.font.size = Pt(14)
    label_r = bar_p.add_run(f"  {pct:.1%}  ({s.get('yes', 0)}/{s.get('total', 0)} yes)")
    label_r.font.size = Pt(11)
    label_r.bold = True
    label_r.font.color.rgb = C.PASS if pct >= 0.5 else C.FAIL

    # By-metric — includes percent_agreement from evaluate.py
    _heading(doc, "By Metric", level=2)
    by_metric = results.get("by_metric", {})
    metric_lookup = {m["id"]: m["name"] for m in bench["test_data"].get("metrics", [])}
    table, widths = _make_table(doc, [1.3, 2.8, 1.3, 1.0, 1.3, 1.66])
    _header_row(table, widths, ["Metric ID", "Name", "Yes Rate", "Yes", "Valid", "Agreement %"])
    for i, (mid, v) in enumerate(sorted(by_metric.items())):
        pa = v.get("percent_agreement")
        _data_row(table, widths, [
            mid,
            metric_lookup.get(mid, mid),
            f"{v.get('yes_rate', 0):.1%}",
            str(v.get("yes", 0)),
            str(v.get("valid", v.get("total", 0))),
            f"{pa:.0%}" if pa is not None else "—",
        ], alt=i % 2 == 1, bold_first=True)
    doc.add_paragraph()

    # By-scenario — also includes percent_agreement
    _heading(doc, "By Scenario", level=2)
    by_scenario = results.get("by_scenario", {})
    scenario_lookup = {sc["id"]: sc["title"] for sc in bench["test_data"].get("scenarios", [])}
    table2, widths2 = _make_table(doc, [1.3, 2.8, 1.3, 1.0, 1.3, 1.66])
    _header_row(table2, widths2, ["Scenario ID", "Title", "Yes Rate", "Yes", "Valid", "Agreement %"])
    for i, (sid, v) in enumerate(sorted(by_scenario.items())):
        pa = v.get("percent_agreement")
        _data_row(table2, widths2, [
            sid,
            scenario_lookup.get(sid, sid),
            f"{v.get('yes_rate', 0):.1%}",
            str(v.get("yes", 0)),
            str(v.get("valid", v.get("total", 0))),
            f"{pa:.0%}" if pa is not None else "—",
        ], alt=i % 2 == 1, bold_first=True)
    doc.add_paragraph()


def _write_scenarios(doc, bench) -> None:
    """
    Per-scenario pages.

    Detail record schema (evaluate.py output):
      {
        "scenario_id": str,
        "metric_id": str,
        "metric_name": str,
        "results": ["yes"|"no", ...],          # one entry per iteration
        "justifications": [str, ...],           # one entry per iteration
      }
    """
    results = bench["results"]
    convs = bench["conversations"]
    metrics = bench["test_data"].get("metrics", [])
    all_details = results.get("details", [])
    n_iter = _num_iterations(bench)

    # Index: (scenario_id, metric_id) → detail record
    detail_map: dict[tuple, dict] = {}
    for d in all_details:
        if not isinstance(d, dict):
            continue
        sid_key = d.get("scenario_id")
        mid_key = d.get("metric_id") or d.get("metric")
        if sid_key and mid_key:
            detail_map[(sid_key, mid_key)] = d

    _heading(doc, "Scenario Results", level=1)
    _add_horizontal_rule(doc)

    for scenario in bench["test_data"].get("scenarios", []):
        sid = scenario["id"]
        conv = convs.get(sid, {})
        samples = conv.get("samples") or ([conv.get("turns", [])] if conv.get("turns") else [])

        _add_page_break(doc)

        # ── Header ────────────────────────────────────────────────────────────
        _heading(doc, f"{sid}  —  {scenario['title']}", level=2)
        _label_value(doc, "Description",      scenario.get("description", ""))
        _label_value(doc, "User goal",         scenario.get("user_goal", ""))
        if scenario.get("user_persona"):
            _label_value(doc, "User persona", scenario["user_persona"])
        _label_value(doc, "Iterations run",    str(n_iter))
        doc.add_paragraph()

        # ── Percent Agreement (computed from detail results lists) ─────────────
        _heading(doc, "Percent Agreement Across Iterations", level=3)
        table_a, widths_a = _make_table(doc, [1.4, 2.6, 1.4, 0.8, 0.8, 2.36])
        _header_row(table_a, widths_a, ["Metric ID", "Name", "Agreement %", "Yes", "No", "Consistency"])

        for i, metric in enumerate(metrics):
            mid = metric["id"]
            detail = detail_map.get((sid, mid))

            if detail is None:
                row = table_a.add_row()
                bg = C.ROW_ALT if i % 2 == 1 else C.WHITE
                for cell, val, w in zip(row.cells,
                                        [mid, metric.get("name", mid), "—", "—", "—", "—"],
                                        widths_a):
                    _set_cell_bg(cell, bg)
                    _set_cell_margins(cell)
                    cell.width = Twips(w)
                    cell.paragraphs[0].add_run(val).font.size = Pt(9)
                continue

            result_list = detail.get("results", [])
            if isinstance(result_list, str):
                result_list = [result_list]
            result_list = [str(r).lower() for r in result_list]

            n_yes = sum(1 for r in result_list if r == "yes")
            n_no = len(result_list) - n_yes
            total = len(result_list)
            majority = "yes" if n_yes >= n_no else "no"
            n_agree = n_yes if majority == "yes" else n_no
            agree_pct = n_agree / total if total else 0.0
            consistency = "High" if agree_pct >= 0.9 else ("Moderate" if agree_pct >= 0.6 else "Low")

            row = table_a.add_row()
            bg = C.ROW_ALT if i % 2 == 1 else C.WHITE
            vals = [mid, metric.get("name", mid), f"{agree_pct:.0%}", str(n_yes), str(n_no), consistency]
            for j, (cell, val, w) in enumerate(zip(row.cells, vals, widths_a)):
                _set_cell_bg(cell, bg)
                _set_cell_margins(cell)
                cell.width = Twips(w)
                run = cell.paragraphs[0].add_run(str(val))
                run.font.size = Pt(9)
                if j == 0:
                    run.bold = True
                if j == 2:
                    run.font.color.rgb = C.PASS if agree_pct >= 0.6 else C.FAIL
                    run.bold = True
                if j == 5:
                    run.font.color.rgb = (
                        C.PASS if consistency == "High"
                        else C.FAIL if consistency == "Low"
                        else C.MID_GREY
                    )

        doc.add_paragraph()

        # ── Metric scorecard — Sample 1 verdict + justification ────────────────
        _heading(doc, f"Metric Scorecard (Sample 1 of {n_iter})", level=3)
        table_s, widths_s = _make_table(doc, [1.4, 2.3, 0.9, 4.76])
        _header_row(table_s, widths_s, ["Metric ID", "Name", "Result", "Justification (Sample 1)"])

        for i, metric in enumerate(metrics):
            mid = metric["id"]
            detail = detail_map.get((sid, mid))
            if detail is None:
                continue
            verdict = _detail_result(detail, sample=0)
            just = _detail_justification(detail, sample=0)
            _data_row(
                table_s, widths_s,
                [mid, metric.get("name", mid), verdict.upper(), just],
                alt=i % 2 == 1,
                verdict_col=2,
            )

        doc.add_paragraph()

        # ── Transcript — Sample 1 only ─────────────────────────────────────────
        if samples:
            _heading(doc, f"Conversation Transcript (Sample 1 of {n_iter})", level=3)
            for t_idx, turn in enumerate(samples[0]):
                role = turn.get("role", "")
                content = turn.get("content", "")

                label_p = doc.add_paragraph()
                label_p.paragraph_format.space_before = Pt(6)
                label_p.paragraph_format.space_after = Pt(1)
                label_r = label_p.add_run(
                    f"Turn {t_idx // 2 + 1}  —  {'USER' if role == 'user' else 'ASSISTANT'}"
                )
                label_r.bold = True
                label_r.font.size = Pt(9)
                label_r.font.color.rgb = C.ACCENT if role == "assistant" else C.MID_GREY

                body_p = doc.add_paragraph()
                body_p.paragraph_format.left_indent = Inches(0.2)
                body_p.paragraph_format.space_before = Pt(0)
                body_p.paragraph_format.space_after = Pt(4)
                body_p.add_run(content).font.size = Pt(9)

        doc.add_paragraph()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export(
    config_path: str = "config.json",
    results_root: str = "results",
    benchmark: str | None = None,
    output_path: str = "report.docx",
) -> None:
    config = load_config(config_path)
    test_path = config["paths"]["test_file"]
    conv_dir_name = config["paths"]["conversations_dir"]
    results_file = config["paths"]["results_file"]
    user_model = get_model_name(config, "user")
    target_model = get_model_name(config, "target")
    goal_content = _load_goal_content(config)

    bench_paths = _resolve_benchmarks(results_root, test_path, benchmark)

    for bench_path in bench_paths:
        bench = _load_benchmark(bench_path, conv_dir_name, results_file)
        out = (
            os.path.join(results_root, benchmark, "report.docx")
            if len(bench_paths) > 1 or output_path == "report.docx"
            else output_path
        )

        print(f"\n  Building report for: {bench['name']}")

        doc = Document()
        section = doc.sections[0]
        section.page_width = Twips(12240)
        section.page_height = Twips(15840)
        for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, attr, Inches(1))
        doc.styles["Normal"].font.name = "Arial"
        doc.styles["Normal"].font.size = Pt(10)

        _write_cover(doc, bench, config, user_model, target_model)
        _add_page_break(doc)
        _write_model_and_prompts(doc, config, goal_content)
        _add_page_break(doc)
        _write_metrics_overview(doc, bench)
        _add_page_break(doc)
        _write_aggregate(doc, bench)
        _write_scenarios(doc, bench)

        doc.save(out)
        print(f"  ✓ Saved → {out}")

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Export results to .docx")
    parser.add_argument("--config",       default="config.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--benchmark",    type=str, required=False)
    parser.add_argument("--output",       type=str, default="report.docx",
                        help="Output path (ignored when running all benchmarks)")
    args = parser.parse_args()

    export(
        config_path=args.config,
        results_root=args.results_root,
        benchmark=args.benchmark,
        output_path=args.output,
    )