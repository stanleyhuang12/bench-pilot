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
from datetime import datetime

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from config import load_config, get_model_name



class C:
    ACCENT = RGBColor(0x2E, 0x75, 0xB6)   # blue headings
    ACCENT_DARK = RGBColor(0x1F, 0x54, 0x96)   # darker blue
    PASS = RGBColor(0x1E, 0x84, 0x49)   # green
    FAIL = RGBColor(0xC0, 0x39, 0x2B)   # red
    PASS_BG = "D5F5E3"  # light green (hex for XML)
    FAIL_BG = "FADBD8"  # light red
    HEADER_BG = "2E75B6"  # blue table header bg
    ROW_ALT = "EBF3FB"     # alternating row bg
    WHITE = "FFFFFF"
    LIGHT_GREY = "F2F2F2"
    MID_GREY = RGBColor(0x66, 0x66, 0x66)
    BORDER = "CCCCCC"


def _set_cell_bg(cell, hex_color: str) -> None:
    """Set table cell background colour via raw XML shading."""
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
    sizes  = {1: 20, 2: 16, 3: 13}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(sizes.get(level, 12))
    run.font.color.rgb = colors.get(level, C.ACCENT)


def _body(doc: Document, text: str, italic: bool = False, color: RGBColor | None = None) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.font.size = Pt(10)
    run.italic = italic
    if color:
        run.font.color.rgb = color


def _label_value(doc: Document, label: str, value: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    r1 = p.add_run(f"{label}: ")
    r1.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = C.MID_GREY
    r2 = p.add_run(value)
    r2.font.size = Pt(10)


def _pass_fail_inline(doc: Document, result: str, justification: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Inches(0.25)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(2)
    badge = p.add_run(f"  {'PASS' if result == 'pass' else 'FAIL'}  ")
    badge.bold = True
    badge.font.size = Pt(9)
    badge.font.color.rgb = C.PASS if result == "pass" else C.FAIL
    spacer = p.add_run("  ")
    spacer.font.size = Pt(9)
    body = p.add_run(justification)
    body.font.size = Pt(9)
    body.font.color.rgb = C.MID_GREY
    body.italic = True



def _make_table(doc: Document, col_widths_inches: list[float]) -> "Table":
    """Create a table with no default borders, correct widths."""
    col_widths_dxa = [int(w * 1440) for w in col_widths_inches]
    table = doc.add_table(rows=0, cols=len(col_widths_dxa))
    table.style = "Table Grid"
    # Remove all default borders by setting width explicitly per-column handled in cells
    return table, col_widths_dxa


def _header_row(table, col_widths_dxa: list[int], labels: list[str]) -> None:
    row = table.add_row()
    for i, (cell, label, w) in enumerate(zip(row.cells, labels, col_widths_dxa)):
        _set_cell_bg(cell, C.HEADER_BG)
        _set_cell_margins(cell)
        cell.width = Twips(w)
        p = cell.paragraphs[0]
        run = p.add_run(label)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def _data_row(table, col_widths_dxa: list[int], values: list[str], alt: bool = False,
              bold_first: bool = False, result: str | None = None) -> None:
    row = table.add_row()
    bg = C.ROW_ALT if alt else C.WHITE
    for i, (cell, val, w) in enumerate(zip(row.cells, values, col_widths_dxa)):
        _set_cell_bg(cell, bg)
        _set_cell_margins(cell)
        cell.width = Twips(w)
        p = cell.paragraphs[0]
        run = p.add_run(str(val))
        run.font.size = Pt(9)
        if i == 0 and bold_first:
            run.bold = True
        # Colour the result column
        if result and i == len(values) - 1:
            run.font.color.rgb = C.PASS if result == "pass" else C.FAIL
            run.bold = True

def _resolve_benchmarks(results_root: str, test_path: str, benchmark: str | None) -> list[str]:
    entries = os.listdir(results_root)
    if benchmark:
        if benchmark not in entries:
            raise FileNotFoundError(f"Benchmark '{benchmark}' not found in {results_root}")
        return [os.path.join(results_root, benchmark, test_path)]
    return [os.path.join(results_root, b, test_path) for b in entries]


def _load_benchmark(bench: str, conv_dir_name: str, results_filename: str) -> dict:
    """Load test data, conversations, and evaluation results for one benchmark."""
    bench_dir  = os.path.dirname(bench)
    bench_name = os.path.basename(bench_dir)
    conv_path  = os.path.join(bench_dir, conv_dir_name)
    results_path = os.path.join(bench_dir, results_filename)

    if not os.path.exists(bench):
        raise FileNotFoundError(f"{bench} — run generate.py first.")
    if not os.path.exists(conv_path):
        raise FileNotFoundError(f"{conv_path} — run simulate.py first.")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} — run evaluate.py first.")

    with open(bench) as f:
        test_data = json.load(f)

    conversations = {}
    for fname in sorted(os.listdir(conv_path)):
        if fname.endswith(".json"):
            with open(os.path.join(conv_path, fname)) as f:
                conv = json.load(f)
                conversations[conv["scenario_id"]] = conv

    with open(results_path) as f:
        results = json.load(f)

    return {
        "name":          bench_name,
        "test_data":     test_data,
        "conversations": conversations,
        "results":       results,
    }


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _write_cover(doc: Document, bench: dict, config: dict, user_model: str, target_model: str) -> None:
    test_data = bench["test_data"]

    doc.add_paragraph()  # top spacing
    _add_horizontal_rule(doc, color="2E75B6", size=12)

    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(8)
    title_p.paragraph_format.space_after  = Pt(4)
    run = title_p.add_run(test_data.get("benchmark_name", bench["name"].replace("-", " ").title()))
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

    _label_value(doc, "Description",   test_data.get("description", "—"))
    _label_value(doc, "User model",    user_model)
    _label_value(doc, "Target model",  target_model)
    _label_value(doc, "Generated",     datetime.now().strftime("%B %d, %Y"))
    _label_value(doc, "Scenarios",     str(len(bench["conversations"])))
    _label_value(doc, "Metrics",       str(len(test_data.get("metrics", []))))

    doc.add_paragraph()


def _write_metrics_overview(doc: Document, bench: dict) -> None:
    metrics = bench["test_data"].get("metrics", [])
    _heading(doc, "Evaluated Metrics", level=1)
    _add_horizontal_rule(doc)

    table, widths = _make_table(doc, [1.2, 1.8, 4.36])
    _header_row(table, widths, ["ID", "Name", "Description"])
    for i, m in enumerate(metrics):
        _data_row(table, widths, [m.get("id", ""), m.get("name", ""), m.get("description", "")], alt=i % 2 == 1)

    doc.add_paragraph()


def _write_aggregate(doc: Document, bench: dict) -> None:
    results = bench["results"]
    s = results["summary"]

    _heading(doc, "Aggregate Results", level=1)
    _add_horizontal_rule(doc)

    # Summary bar
    pct = s["pass_rate"]
    bar_p = doc.add_paragraph()
    bar_p.paragraph_format.space_after = Pt(6)
    total_blocks = 20
    filled = round(pct * total_blocks)
    bar_r = bar_p.add_run("█" * filled)
    bar_r.font.color.rgb = C.PASS
    bar_r.font.size = Pt(14)
    empty_r = bar_p.add_run("█" * (total_blocks - filled))
    empty_r.font.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    empty_r.font.size = Pt(14)
    label_r = bar_p.add_run(f"  {pct:.1%}  ({s['passed']}/{s['total']} passed)")
    label_r.font.size = Pt(11)
    label_r.bold = True
    label_r.font.color.rgb = C.PASS if pct >= 0.5 else C.FAIL

    # By-metric table
    _heading(doc, "By Metric", level=2)
    by_metric = results.get("by_metric", {})
    table, widths = _make_table(doc, [1.5, 3.5, 1.5, 1.5, 1.5])
    _header_row(table, widths, ["Metric ID", "Name", "Pass Rate", "Passed", "Total"])
    metric_lookup = {m["id"]: m["name"] for m in bench["test_data"].get("metrics", [])}
    for i, (mid, v) in enumerate(sorted(by_metric.items())):
        _data_row(table, widths, [
            mid,
            metric_lookup.get(mid, mid),
            f"{v['pass_rate']:.1%}",
            str(v["passed"]),
            str(v["total"]),
        ], alt=i % 2 == 1, bold_first=True)

    doc.add_paragraph()

    # By-scenario table
    _heading(doc, "By Scenario", level=2)
    by_scenario = results.get("by_scenario", {})
    scenario_lookup = {s["id"]: s["title"] for s in bench["test_data"].get("scenarios", [])}
    table2, widths2 = _make_table(doc, [1.5, 3.5, 1.5, 1.5, 1.5])
    _header_row(table2, widths2, ["Scenario ID", "Title", "Pass Rate", "Passed", "Total"])
    for i, (sid, v) in enumerate(sorted(by_scenario.items())):
        _data_row(table2, widths2, [
            sid,
            scenario_lookup.get(sid, sid),
            f"{v['pass_rate']:.1%}",
            str(v["passed"]),
            str(v["total"]),
        ], alt=i % 2 == 1, bold_first=True)

    doc.add_paragraph()


def _write_scenarios(doc: Document, bench: dict) -> None:
    results    = bench["results"]
    convs      = bench["conversations"]
    metrics    = bench["test_data"].get("metrics", [])
    metric_map = {m["id"]: m for m in metrics}

    # Index eval details by (scenario_id, metric_id)
    detail_map: dict[tuple, dict] = {}
    for d in results.get("details", []):
        detail_map[(d["scenario_id"], d["metric_id"])] = d

    _heading(doc, "Scenario Results", level=1)
    _add_horizontal_rule(doc)

    scenarios = bench["test_data"].get("scenarios", [])

    for scenario in scenarios:
        sid   = scenario["id"]
        conv  = convs.get(sid)

        _add_page_break(doc)

        # ── Scenario header ───────────────────────────────────────────────
        _heading(doc, f"{sid}  —  {scenario['title']}", level=2)
        _label_value(doc, "Description", scenario.get("description", ""))
        _label_value(doc, "User goal",   scenario.get("user_goal", ""))

        persona = scenario.get("user_persona", "")
        if persona:
            _label_value(doc, "User persona", persona)

        doc.add_paragraph()

        # ── Metric scorecard ──────────────────────────────────────────────
        _heading(doc, "Metric Scorecard", level=3)
        table, widths = _make_table(doc, [1.5, 2.5, 1.0, 4.0])
        _header_row(table, widths, ["Metric ID", "Name", "Result", "Justification"])

        for i, metric in enumerate(metrics):
            mid    = metric["id"]
            detail = detail_map.get((sid, mid))
            if detail is None:
                continue
            result = detail["result"]
            _data_row(
                table, widths,
                [mid, metric.get("name", mid), result.upper(), detail.get("justification", "")],
                alt=i % 2 == 1,
                result=result,
            )

        doc.add_paragraph()

        # ── Conversation transcript ───────────────────────────────────────
        if conv:
            _heading(doc, "Conversation Transcript", level=3)
            turns = conv.get("turns", [])
            for t_idx, turn in enumerate(turns):
                role    = turn["role"]
                content = turn["content"]

                label_p = doc.add_paragraph()
                label_p.paragraph_format.space_before = Pt(6)
                label_p.paragraph_format.space_after  = Pt(1)
                label_r = label_p.add_run(
                    f"Turn {t_idx // 2 + 1}  —  {'USER' if role == 'user' else 'ASSISTANT'}"
                )
                label_r.bold = True
                label_r.font.size = Pt(9)
                label_r.font.color.rgb = C.ACCENT if role == "assistant" else C.MID_GREY

                body_p = doc.add_paragraph()
                body_p.paragraph_format.left_indent  = Inches(0.2)
                body_p.paragraph_format.space_before = Pt(0)
                body_p.paragraph_format.space_after  = Pt(4)
                body_r = body_p.add_run(content)
                body_r.font.size = Pt(9)

        doc.add_paragraph()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export(
    config_path:   str = "config.json",
    results_root:  str = "results",
    benchmark:     str | None = None,
    output_path:   str = "report.docx",
) -> None:
    config          = load_config(config_path)
    test_path       = config["paths"]["test_file"]
    conv_dir_name   = config["paths"]["conversations_dir"]
    results_file    = config["paths"]["results_file"]
    user_model      = get_model_name(config, "user")
    target_model    = get_model_name(config, "target")

    bench_paths = _resolve_benchmarks(results_root, test_path, benchmark)

    for bench_path in bench_paths:
        bench     = _load_benchmark(bench_path, conv_dir_name, results_file)
        bench_dir = os.path.dirname(bench_path)

        # Derive output path per-benchmark unless user set a custom one
        if len(bench_paths) > 1 or output_path == "report.docx":
            out = os.path.join(bench_dir, "report.docx")
        else:
            out = output_path

        print(f"\n  Building report for: {bench['name']}")

        doc = Document()

        # Page setup — US Letter, 1" margins
        section = doc.sections[0]
        section.page_width  = Twips(12240)
        section.page_height = Twips(15840)
        for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, attr, Inches(1))

        # Default font
        doc.styles["Normal"].font.name = "Arial"
        doc.styles["Normal"].font.size = Pt(10)

        # ── Sections ──────────────────────────────────────────────────────
        _write_cover(doc, bench, config, user_model, target_model)
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