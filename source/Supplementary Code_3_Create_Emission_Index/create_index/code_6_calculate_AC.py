# %%
import os
import glob
import re
import time
from pathlib import Path

import sys
if sys.platform == "win32" and sys.version_info[:2] == (3, 12):
    _runtime_base = Path(__file__).resolve().parent if "__file__" in globals() else Path(r"D:\emi_association\3_construct_index_old\create_index")
    for _runtime_dir in [_runtime_base / ".economic_plot_dependencies", _runtime_base / ".code_10_1_dependencies"]:
        if _runtime_dir.is_dir():
            sys.path.insert(0, str(_runtime_dir))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

START_TIME = time.perf_counter()
BASE_DIR = Path(__file__).resolve().parent
BETA_PATH = BASE_DIR / (
    "code_5_results/Targeted_CEI_Association_Results.csv"
)
HIST_PANEL_PATH = BASE_DIR / "data_raw/grid_level_emission.csv"
HIST_COUNTY_PATH = BASE_DIR / (
    "code_4_results/2_merged_yearly_data_new.csv"
)
PRIORITY_PATH = BASE_DIR / (
    "code_3_results/3_Final_Top50pct_Categorized_Targets.csv"
)
DOSE_RESPONSE_LOOKUP_PATH = BASE_DIR / (
    "code_5_results/Dose_Response_CEI_IQR_Results.csv"
)

FUT_PRED_DIR = BASE_DIR / "data_raw"
SHP_PATH = FUT_PRED_DIR / "shp版/中国_县_面.shp"
OUTPUT_DIR = BASE_DIR / "code_6_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SSPS = ["ssp126", "ssp245", "ssp370"]
SSP_COLORS = {
    "ssp126": "#2ca02c",
    "ssp245": "#1f77b4",
    "ssp370": "#ff7f0e",
}
SSP_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
}

EXCLUDED_DISEASES = {""}

MAKE_FIGURES = True

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": True,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}缺少列：{missing}")


def integer_key(series, name):
    """统一整数键，避免县域ID或年份的类型不一致。"""
    values = pd.to_numeric(series, errors="raise")
    arr = values.to_numpy(dtype=float)

    if (
        not np.isfinite(arr).all()
        or not np.equal(arr, np.floor(arr)).all()
    ):
        raise ValueError(f"{name}包含缺失值或非整数")

    return values.astype("int64")


def check_finite(df, columns, name):
    for col in columns:
        values = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}：{col}存在无效值")
        df[col] = values


def resolve_prediction_column(columns, pollutant):
    """优先精确匹配，禁止从多个候选列中任意取最后一列。"""
    for col in [
        pollutant,
        f"{pollutant}_pred",
        f"{pollutant}_Total_pred",
        f"{pollutant}_prediction",
    ]:
        if col in columns:
            return col

    candidates = [
        c for c in columns
        if c.startswith(f"{pollutant}_")
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{pollutant}无法唯一匹配预测列：{candidates}"
        )
    return candidates[0]


def load_population_skeleton(ssp):
    """沿用SSP文件中的县域人口，不改变人口情景。"""
    files = sorted(
        glob.glob(str(FUT_PRED_DIR / f"*_{ssp}.csv"))
    )
    required = ["unique_id", "time", "population"]

    source = None
    for file in files:
        header = pd.read_csv(file, nrows=0)
        if set(required).issubset(header.columns):
            source = file
            break

    if source is None:
        raise FileNotFoundError(
            f"{ssp}没有包含{required}的人口骨架文件"
        )

    df = pd.read_csv(source, usecols=required)
    df["unique_id"] = integer_key(df["unique_id"], "人口县域ID")
    df["time"] = integer_key(df["time"], "人口年份")
    check_finite(df, ["population"], source)
    df = df.drop_duplicates().copy()

    if df.duplicated(["unique_id", "time"]).any():
        raise ValueError(
            f"{source}中同一县域、年份对应多个不同人口值"
        )
    if (df["population"] < 0).any():
        raise ValueError(f"{source}存在负人口")

    print(f" -> {ssp}人口文件：{Path(source).name}")
    return df


def save_figure(fig, stem):
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


_CEI_ALIGNMENT_SOURCE = '#!/usr/bin/env python3\n"""Audit multi-panel figure alignment from rendered layout geometry.\n\nThe core auditor consumes a small backend-neutral JSON manifest. Matplotlib\nfigures can call :func:`require_matplotlib_panel_alignment` directly after the\nfinal layout draw. R/patchwork figures use ``panel_alignment.R`` to export the\nsame manifest and then invoke this CLI.\n\nThis gate checks the plot-area rectangles that readers perceive as panels. It\ndoes not infer scientific equivalence: comparable row and column groups come\nfrom Matplotlib SubplotSpec metadata, patchwork/gtable metadata, or explicit\ngroups supplied by the plotting script. Horizontal rows of three or four equal\ngrid spans must also have equal final physical widths. Asymmetric hero panels,\ninsets and colorbars must be excluded or exempted with a recorded reason rather\nthan silently weakening the tolerance.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport html\nimport json\nimport math\nimport os\nimport re\nimport tempfile\nfrom collections import defaultdict\nfrom collections.abc import Iterable, Mapping, Sequence\nfrom pathlib import Path\nfrom typing import Any\n\n\nSCHEMA_VERSION = 1\nDEFAULT_TOLERANCE_PT = 1.5\nDEFAULT_GUTTER_TOLERANCE_PT = 1.5\n\n\nclass PanelAlignmentError(RuntimeError):\n    """Raised when a required in-memory alignment gate does not pass."""\n\n\ndef _atomic_write_text(path: Path, content: str) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    descriptor, temporary_name = tempfile.mkstemp(\n        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent\n    )\n    temporary = Path(temporary_name)\n    try:\n        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:\n            handle.write(content)\n        temporary.replace(path)\n    except Exception:\n        temporary.unlink(missing_ok=True)\n        raise\n\n\ndef _bbox(value: Any) -> tuple[float, float, float, float]:\n    if isinstance(value, Mapping):\n        values = [value.get(key) for key in ("left", "bottom", "right", "top")]\n    else:\n        values = list(value) if isinstance(value, Sequence) else []\n    if len(values) != 4:\n        raise ValueError("panel bbox_pt must contain left, bottom, right and top")\n    bbox = tuple(float(item) for item in values)\n    if not all(math.isfinite(item) for item in bbox):\n        raise ValueError("panel bbox_pt values must be finite")\n    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:\n        raise ValueError("panel bbox_pt must have positive width and height")\n    return bbox\n\n\ndef _group_rows(raw: Any, prefix: str) -> list[dict[str, Any]]:\n    groups: list[dict[str, Any]] = []\n    if raw is None:\n        return groups\n    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):\n        raise ValueError(f"{prefix}_groups must be a list")\n    for index, item in enumerate(raw, 1):\n        if isinstance(item, Mapping):\n            panels = item.get("panels", [])\n            group_id = str(item.get("id") or f"{prefix}-{index}")\n        else:\n            panels = item\n            group_id = f"{prefix}-{index}"\n        if not isinstance(panels, Sequence) or isinstance(panels, (str, bytes)):\n            raise ValueError(f"{prefix} group {group_id} must list panel ids")\n        panel_ids = [str(panel) for panel in panels]\n        if len(panel_ids) < 2:\n            raise ValueError(f"{prefix} group {group_id} needs at least two panels")\n        if len(panel_ids) != len(set(panel_ids)):\n            raise ValueError(f"{prefix} group {group_id} repeats a panel id")\n        groups.append({"id": group_id, "panels": panel_ids})\n    return groups\n\n\ndef _inferred_groups(panels: Sequence[dict[str, Any]], orientation: str) -> list[dict[str, Any]]:\n    grouped: dict[tuple[Any, ...], list[str]] = defaultdict(list)\n    for panel in panels:\n        if orientation == "row":\n            fields = (panel.get("grid_id"), panel.get("row_start"), panel.get("row_stop"))\n        else:\n            fields = (panel.get("grid_id"), panel.get("col_start"), panel.get("col_stop"))\n        if any(value is None for value in fields):\n            continue\n        grouped[fields].append(str(panel["id"]))\n    output: list[dict[str, Any]] = []\n    for index, (_key, panel_ids) in enumerate(grouped.items(), 1):\n        if len(panel_ids) >= 2:\n            output.append({"id": f"inferred-{orientation}-{index}", "panels": panel_ids})\n    return output\n\n\ndef _inferred_boundary_groups(panels: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:\n    """Infer shared grid boundaries across panels with unequal spans.\n\n    This covers layouts such as two stacked panels beside one panel that spans\n    both rows. Equal-span panels are handled by the ordinary row/column groups,\n    so boundary groups are emitted only when at least two different spans meet\n    at the same grid boundary.\n    """\n    specifications = (\n        ("top", "row_start", "row_start", "row_stop"),\n        ("bottom", "row_stop", "row_start", "row_stop"),\n        ("left", "col_start", "col_start", "col_stop"),\n        ("right", "col_stop", "col_start", "col_stop"),\n    )\n    output: list[dict[str, Any]] = []\n    for edge, boundary_field, span_start, span_stop in specifications:\n        grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)\n        for panel in panels:\n            grid_id = panel.get("grid_id")\n            boundary = panel.get(boundary_field)\n            start = panel.get(span_start)\n            stop = panel.get(span_stop)\n            if any(value is None for value in (grid_id, boundary, start, stop)):\n                continue\n            grouped[(grid_id, boundary)].append(panel)\n        edge_index = 0\n        for (_grid_id, _boundary), members in grouped.items():\n            spans = {(panel.get(span_start), panel.get(span_stop)) for panel in members}\n            if len(members) < 2 or len(spans) < 2:\n                continue\n            edge_index += 1\n            output.append(\n                {\n                    "id": f"inferred-shared-{edge}-{edge_index}",\n                    "edge": edge,\n                    "panels": [str(panel["id"]) for panel in members],\n                }\n            )\n    return output\n\n\ndef _validated_layout(manifest: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:\n    errors: list[str] = []\n    if manifest.get("schema_version") != SCHEMA_VERSION:\n        errors.append(f"schema_version must be {SCHEMA_VERSION}")\n\n    figure = manifest.get("figure")\n    if not isinstance(figure, Mapping):\n        errors.append("figure must be an object with width_pt and height_pt")\n        figure = {}\n    try:\n        width_pt = float(figure.get("width_pt"))\n        height_pt = float(figure.get("height_pt"))\n        if not math.isfinite(width_pt) or not math.isfinite(height_pt) or width_pt <= 0 or height_pt <= 0:\n            raise ValueError\n    except (TypeError, ValueError):\n        errors.append("figure width_pt and height_pt must be positive finite numbers")\n        width_pt = height_pt = 0.0\n\n    raw_panels = manifest.get("panels")\n    panels: list[dict[str, Any]] = []\n    if not isinstance(raw_panels, Sequence) or isinstance(raw_panels, (str, bytes)):\n        errors.append("panels must be a list")\n        raw_panels = []\n    for index, raw_panel in enumerate(raw_panels, 1):\n        if not isinstance(raw_panel, Mapping):\n            errors.append(f"panel {index} must be an object")\n            continue\n        panel_id = str(raw_panel.get("id") or "").strip()\n        if not panel_id:\n            errors.append(f"panel {index} is missing id")\n            continue\n        try:\n            bbox = _bbox(raw_panel.get("bbox_pt"))\n        except (TypeError, ValueError) as exc:\n            errors.append(f"panel {panel_id}: {exc}")\n            continue\n        if bbox[0] < -0.01 or bbox[1] < -0.01 or bbox[2] > width_pt + 0.01 or bbox[3] > height_pt + 0.01:\n            errors.append(f"panel {panel_id} bbox_pt extends beyond the declared figure")\n        panel = dict(raw_panel)\n        panel["id"] = panel_id\n        panel["bbox_pt"] = list(bbox)\n        anchor = raw_panel.get("panel_label_anchor_pt")\n        if anchor is not None:\n            try:\n                anchor_values = [float(value) for value in anchor]\n                if len(anchor_values) != 2 or not all(math.isfinite(value) for value in anchor_values):\n                    raise ValueError\n                panel["panel_label_anchor_pt"] = anchor_values\n            except (TypeError, ValueError):\n                errors.append(f"panel {panel_id} has invalid panel_label_anchor_pt")\n        panels.append(panel)\n\n    panel_ids = [panel["id"] for panel in panels]\n    if len(panel_ids) != len(set(panel_ids)):\n        errors.append("panel ids must be unique")\n    if not panels:\n        errors.append("at least one panel rectangle is required")\n\n    try:\n        row_groups = _group_rows(manifest.get("row_groups"), "row")\n        column_groups = _group_rows(manifest.get("column_groups"), "column")\n    except ValueError as exc:\n        errors.append(str(exc))\n        row_groups = column_groups = []\n    if not row_groups:\n        row_groups = _inferred_groups(panels, "row")\n    if not column_groups:\n        column_groups = _inferred_groups(panels, "column")\n    boundary_groups = _inferred_boundary_groups(panels)\n\n    known_ids = set(panel_ids)\n    for group in [*row_groups, *column_groups]:\n        missing = [panel for panel in group["panels"] if panel not in known_ids]\n        if missing:\n            errors.append(f"group {group[\'id\']} references unknown panels: {\', \'.join(missing)}")\n\n    exemptions: list[dict[str, Any]] = []\n    raw_exemptions = manifest.get("exemptions", [])\n    if not isinstance(raw_exemptions, Sequence) or isinstance(raw_exemptions, (str, bytes)):\n        errors.append("exemptions must be a list")\n        raw_exemptions = []\n    for index, exemption in enumerate(raw_exemptions, 1):\n        if not isinstance(exemption, Mapping):\n            errors.append(f"exemption {index} must be an object")\n            continue\n        exemption_panels = exemption.get("panels", [])\n        checks = exemption.get("checks", [])\n        reason = str(exemption.get("reason") or "").strip()\n        if isinstance(exemption_panels, str):\n            exemption_panels = [exemption_panels]\n        if isinstance(checks, str):\n            checks = [checks]\n        if not exemption_panels or any(str(panel) not in known_ids for panel in exemption_panels):\n            errors.append(f"exemption {index} must reference known panels")\n        allowed_checks = {\n            "row",\n            "column",\n            "panel-width",\n            "horizontal-gutter",\n            "vertical-gutter",\n            "panel-label",\n            "all",\n        }\n        check_names = [str(check) for check in checks]\n        if not check_names or any(check not in allowed_checks for check in check_names):\n            errors.append(f"exemption {index} has invalid checks")\n        if not reason:\n            errors.append(f"exemption {index} needs a non-empty reason")\n        exemptions.append(\n            {"panels": [str(panel) for panel in exemption_panels], "checks": check_names, "reason": reason}\n        )\n\n    if errors:\n        return None, errors\n    return {\n        "schema_version": SCHEMA_VERSION,\n        "backend": str(manifest.get("backend") or "unknown"),\n        "figure": {"width_pt": width_pt, "height_pt": height_pt},\n        "panels": panels,\n        "row_groups": row_groups,\n        "column_groups": column_groups,\n        "boundary_groups": boundary_groups,\n        "exemptions": exemptions,\n    }, []\n\n\ndef _is_exempt(layout: Mapping[str, Any], panel_id: str, check: str) -> bool:\n    for exemption in layout.get("exemptions", []):\n        if panel_id in exemption["panels"] and (check in exemption["checks"] or "all" in exemption["checks"]):\n            return True\n    return False\n\n\ndef _metric_spread(values: Mapping[str, float]) -> float:\n    return max(values.values()) - min(values.values()) if values else 0.0\n\n\ndef _finding(\n    kind: str,\n    group: Mapping[str, Any],\n    panels: Sequence[str],\n    tolerance_pt: float,\n    metrics: Mapping[str, Mapping[str, float]],\n    message: str,\n    severity: str = "FAIL",\n) -> dict[str, Any]:\n    return {\n        "severity": severity,\n        "kind": kind,\n        "group": group["id"],\n        "panels": list(panels),\n        "message": message,\n        "tolerance_pt": tolerance_pt,\n        "metric_spreads_pt": {\n            metric: round(_metric_spread(values), 6) for metric, values in metrics.items()\n        },\n        "values_pt": {\n            metric: {panel: round(value, 6) for panel, value in values.items()}\n            for metric, values in metrics.items()\n        },\n    }\n\n\ndef audit_layout_manifest(\n    manifest: Mapping[str, Any],\n    *,\n    tolerance_pt: float = DEFAULT_TOLERANCE_PT,\n    gutter_tolerance_pt: float = DEFAULT_GUTTER_TOLERANCE_PT,\n    require_panel_labels: bool = False,\n) -> dict[str, Any]:\n    """Audit a backend-neutral panel-layout manifest."""\n    if tolerance_pt < 0 or gutter_tolerance_pt < 0:\n        raise ValueError("alignment tolerances must be non-negative")\n    layout, errors = _validated_layout(manifest)\n    if layout is None:\n        return {\n            "schema_version": SCHEMA_VERSION,\n            "auditable": False,\n            "verdict": "NOT AUDITABLE",\n            "summary": {"fail": 0, "warn": 0, "comparisons": 0, "exemptions": 0},\n            "errors": errors,\n            "findings": [],\n        }\n\n    if len(layout["panels"]) == 1:\n        return {\n            "schema_version": SCHEMA_VERSION,\n            "applicable": False,\n            "auditable": False,\n            "verdict": "NOT APPLICABLE",\n            "backend": layout["backend"],\n            "summary": {"fail": 0, "warn": 0, "comparisons": 0, "exemptions": 0},\n            "tolerances": {\n                "alignment_pt": tolerance_pt,\n                "gutter_pt": gutter_tolerance_pt,\n            },\n            "errors": [],\n            "findings": [],\n            "layout": layout,\n        }\n\n    panel_map = {panel["id"]: panel for panel in layout["panels"]}\n    findings: list[dict[str, Any]] = []\n    comparisons = 0\n\n    for orientation, group_key, check_name in (\n        ("row", "row_groups", "row"),\n        ("column", "column_groups", "column"),\n    ):\n        for group in layout[group_key]:\n            group_panel_ids = list(group["panels"])\n            panel_ids = [\n                panel_id for panel_id in group_panel_ids if not _is_exempt(layout, panel_id, check_name)\n            ]\n            if len(panel_ids) < 2:\n                continue\n            comparisons += 1\n            boxes = {panel_id: _bbox(panel_map[panel_id]["bbox_pt"]) for panel_id in panel_ids}\n            if orientation == "row":\n                metrics = {\n                    "top": {panel: box[3] for panel, box in boxes.items()},\n                    "bottom": {panel: box[1] for panel, box in boxes.items()},\n                    "height": {panel: box[3] - box[1] for panel, box in boxes.items()},\n                }\n                kind = "row-axes-misalignment"\n                message = "Panels intended for one row do not share top/bottom edges and height"\n            else:\n                metrics = {\n                    "left": {panel: box[0] for panel, box in boxes.items()},\n                    "right": {panel: box[2] for panel, box in boxes.items()},\n                    "width": {panel: box[2] - box[0] for panel, box in boxes.items()},\n                }\n                kind = "column-axes-misalignment"\n                message = "Panels intended for one column do not share left/right edges and width"\n            if max(_metric_spread(values) for values in metrics.values()) > tolerance_pt:\n                findings.append(_finding(kind, group, panel_ids, tolerance_pt, metrics, message))\n\n            if orientation == "row" and len(group_panel_ids) >= 3:\n                width_candidates = [\n                    panel_id\n                    for panel_id in group_panel_ids\n                    if not _is_exempt(layout, panel_id, "row")\n                    and not _is_exempt(layout, panel_id, "panel-width")\n                ]\n                by_column_span: dict[Any, list[str]] = defaultdict(list)\n                for panel_id in width_candidates:\n                    panel = panel_map[panel_id]\n                    try:\n                        column_span = float(panel.get("col_stop")) - float(panel.get("col_start"))\n                        if not math.isfinite(column_span) or column_span <= 0:\n                            raise ValueError\n                        span_key: Any = round(column_span, 9)\n                    except (TypeError, ValueError):\n                        span_key = "unspecified"\n                    by_column_span[span_key].append(panel_id)\n                for width_index, width_ids in enumerate(by_column_span.values(), 1):\n                    if len(width_ids) < 2:\n                        continue\n                    comparisons += 1\n                    width_metrics = {\n                        "width": {\n                            panel_id: _bbox(panel_map[panel_id]["bbox_pt"])[2]\n                            - _bbox(panel_map[panel_id]["bbox_pt"])[0]\n                            for panel_id in width_ids\n                        }\n                    }\n                    if _metric_spread(width_metrics["width"]) > tolerance_pt:\n                        findings.append(\n                            _finding(\n                                "horizontal-panel-width-misalignment",\n                                {"id": f"{group[\'id\']}-equal-span-{width_index}"},\n                                width_ids,\n                                tolerance_pt,\n                                width_metrics,\n                                "Three-or-more-panel row contains unequal final plot-area widths for equal grid spans",\n                            )\n                        )\n\n            anchors = {\n                panel_id: panel_map[panel_id].get("panel_label_anchor_pt") for panel_id in panel_ids\n            }\n            present_anchors = {panel: anchor for panel, anchor in anchors.items() if anchor is not None}\n            missing_labels = [panel for panel, anchor in anchors.items() if anchor is None]\n            if require_panel_labels and missing_labels:\n                findings.append(\n                    {\n                        "severity": "WARN",\n                        "kind": "panel-label-not-auditable",\n                        "group": group["id"],\n                        "panels": missing_labels,\n                        "message": "Comparable panels are missing detectable top-left panel-label anchors",\n                    }\n                )\n            if len(present_anchors) >= 2:\n                label_check = "panel-label"\n                label_ids = [\n                    panel for panel in panel_ids if panel in present_anchors and not _is_exempt(layout, panel, label_check)\n                ]\n                if len(label_ids) >= 2:\n                    coordinate = 1 if orientation == "row" else 0\n                    metric_name = "label-y" if orientation == "row" else "label-x"\n                    metrics = {\n                        metric_name: {panel: float(present_anchors[panel][coordinate]) for panel in label_ids}\n                    }\n                    if _metric_spread(metrics[metric_name]) > tolerance_pt:\n                        findings.append(\n                            _finding(\n                                f"{orientation}-panel-label-misalignment",\n                                group,\n                                label_ids,\n                                tolerance_pt,\n                                metrics,\n                                "Panel-label anchors are not aligned within their comparable group",\n                            )\n                        )\n\n            gutter_check = "horizontal-gutter" if orientation == "row" else "vertical-gutter"\n            gutter_ids = [panel for panel in panel_ids if not _is_exempt(layout, panel, gutter_check)]\n            if len(gutter_ids) >= 2:\n                if orientation == "row":\n                    ordered = sorted(gutter_ids, key=lambda panel: boxes[panel][0])\n                    gaps = {\n                        f"{first}->{second}": boxes[second][0] - boxes[first][2]\n                        for first, second in zip(ordered, ordered[1:])\n                    }\n                else:\n                    ordered = sorted(gutter_ids, key=lambda panel: boxes[panel][3], reverse=True)\n                    gaps = {\n                        f"{first}->{second}": boxes[first][1] - boxes[second][3]\n                        for first, second in zip(ordered, ordered[1:])\n                    }\n                if gaps and min(gaps.values()) < -tolerance_pt:\n                    findings.append(\n                        _finding(\n                            "panel-axes-overlap",\n                            group,\n                            gutter_ids,\n                            tolerance_pt,\n                            {"gutter": gaps},\n                            "Panel plot-area rectangles overlap",\n                        )\n                    )\n                elif len(gaps) >= 2 and _metric_spread(gaps) > gutter_tolerance_pt:\n                    findings.append(\n                        _finding(\n                            f"{gutter_check}-misalignment",\n                            group,\n                            gutter_ids,\n                            gutter_tolerance_pt,\n                            {"gutter": gaps},\n                            "Comparable inter-panel gutters are not uniform",\n                        )\n                    )\n\n    edge_coordinates = {"left": 0, "bottom": 1, "right": 2, "top": 3}\n    for group in layout["boundary_groups"]:\n        edge = str(group["edge"])\n        check_name = "row" if edge in {"top", "bottom"} else "column"\n        panel_ids = [\n            panel_id\n            for panel_id in group["panels"]\n            if not _is_exempt(layout, panel_id, check_name)\n        ]\n        if len(panel_ids) < 2:\n            continue\n        comparisons += 1\n        coordinate = edge_coordinates[edge]\n        metrics = {\n            edge: {\n                panel_id: _bbox(panel_map[panel_id]["bbox_pt"])[coordinate]\n                for panel_id in panel_ids\n            }\n        }\n        if _metric_spread(metrics[edge]) > tolerance_pt:\n            findings.append(\n                _finding(\n                    f"shared-{edge}-edge-misalignment",\n                    group,\n                    panel_ids,\n                    tolerance_pt,\n                    metrics,\n                    f"Panels meeting the same grid {edge} boundary are not aligned",\n                )\n            )\n\n        if edge not in {"top", "left"}:\n            continue\n        anchors = {\n            panel_id: panel_map[panel_id].get("panel_label_anchor_pt")\n            for panel_id in panel_ids\n        }\n        missing_labels = [panel_id for panel_id, anchor in anchors.items() if anchor is None]\n        if require_panel_labels and missing_labels:\n            findings.append(\n                {\n                    "severity": "WARN",\n                    "kind": "panel-label-not-auditable",\n                    "group": group["id"],\n                    "panels": missing_labels,\n                    "message": "Comparable panels are missing detectable top-left panel-label anchors",\n                }\n            )\n        label_ids = [\n            panel_id\n            for panel_id, anchor in anchors.items()\n            if anchor is not None and not _is_exempt(layout, panel_id, "panel-label")\n        ]\n        if len(label_ids) < 2:\n            continue\n        label_coordinate = 1 if edge == "top" else 0\n        metric_name = "label-y" if edge == "top" else "label-x"\n        label_metrics = {\n            metric_name: {\n                panel_id: float(anchors[panel_id][label_coordinate])\n                for panel_id in label_ids\n            }\n        }\n        if _metric_spread(label_metrics[metric_name]) > tolerance_pt:\n            findings.append(\n                _finding(\n                    f"shared-{edge}-panel-label-misalignment",\n                    group,\n                    label_ids,\n                    tolerance_pt,\n                    label_metrics,\n                    "Panel-label anchors are not aligned at the shared grid boundary",\n                )\n            )\n    fail_count = sum(finding["severity"] == "FAIL" for finding in findings)\n    warn_count = sum(finding["severity"] == "WARN" for finding in findings)\n    auditable = comparisons > 0\n    verdict = "FIX BEFORE DELIVERY" if fail_count else "REVIEW REQUIRED" if warn_count else "PASS"\n    if not auditable:\n        verdict = "NOT AUDITABLE"\n    return {\n        "schema_version": SCHEMA_VERSION,\n        "applicable": True,\n        "auditable": auditable,\n        "verdict": verdict,\n        "backend": layout["backend"],\n        "summary": {\n            "fail": fail_count,\n            "warn": warn_count,\n            "comparisons": comparisons,\n            "exemptions": len(layout["exemptions"]),\n        },\n        "tolerances": {\n            "alignment_pt": tolerance_pt,\n            "gutter_pt": gutter_tolerance_pt,\n        },\n        "layout": layout,\n        "findings": findings,\n        **(\n            {"errors": ["No comparable row, column or shared-boundary groups were declared or inferred"]}\n            if not auditable\n            else {}\n        ),\n    }\n\n\ndef exit_code(report: Mapping[str, Any], strict: bool = False) -> int:\n    if report.get("verdict") == "NOT APPLICABLE":\n        return 0\n    if not report.get("auditable") or report.get("verdict") == "NOT AUDITABLE":\n        return 2\n    if int(report.get("summary", {}).get("fail", 0)):\n        return 1\n    if strict and int(report.get("summary", {}).get("warn", 0)):\n        return 1\n    return 0\n\n\ndef render_text(report: Mapping[str, Any], strict: bool = False) -> str:\n    summary = report.get("summary", {})\n    lines = [\n        f"Panel alignment: {report.get(\'verdict\', \'UNKNOWN\')}",\n        f"  comparisons: {summary.get(\'comparisons\', 0)}",\n        f"  fail: {summary.get(\'fail\', 0)}",\n        f"  warn: {summary.get(\'warn\', 0)}",\n        f"  exemptions: {summary.get(\'exemptions\', 0)}",\n    ]\n    for error in report.get("errors", []):\n        lines.append(f"  ERROR: {error}")\n    for finding in report.get("findings", []):\n        lines.append(\n            f"  {finding[\'severity\']} {finding[\'kind\']} [{finding.get(\'group\', \'n/a\')}]: "\n            f"{finding[\'message\']} ({\', \'.join(finding.get(\'panels\', []))})"\n        )\n    if strict and summary.get("warn", 0):\n        lines.append("  strict mode: WARN findings block delivery")\n    return "\\n".join(lines)\n\n\ndef write_json_report(report: Mapping[str, Any], path: str | Path) -> None:\n    _atomic_write_text(Path(path), json.dumps(report, indent=2, ensure_ascii=False) + "\\n")\n\n\ndef write_overlay_svg(report: Mapping[str, Any], path: str | Path) -> None:\n    layout = report.get("layout")\n    if not isinstance(layout, Mapping):\n        raise ValueError("an auditable layout is required for the diagnostic SVG")\n    width = float(layout["figure"]["width_pt"])\n    height = float(layout["figure"]["height_pt"])\n    failed_panels = {\n        panel for finding in report.get("findings", []) if finding.get("severity") == "FAIL"\n        for panel in finding.get("panels", [])\n    }\n    rows = [\n        f\'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">\',\n        \'<rect width="100%" height="100%" fill="white"/>\',\n    ]\n    for panel in layout["panels"]:\n        left, bottom, right, top = _bbox(panel["bbox_pt"])\n        y = height - top\n        color = "#d7191c" if panel["id"] in failed_panels else "#2c7bb6"\n        rows.append(\n            f\'<rect x="{left:.3f}" y="{y:.3f}" width="{right-left:.3f}" height="{top-bottom:.3f}" \'\n            f\'fill="none" stroke="{color}" stroke-width="1.2"/>\'\n        )\n        rows.append(\n            f\'<text x="{left+2:.3f}" y="{y+10:.3f}" font-family="Arial, sans-serif" font-size="8" \'\n            f\'fill="{color}">{html.escape(str(panel["id"]))}</text>\'\n        )\n    rows.append(\n        f\'<text x="4" y="{height-5:.3f}" font-family="Arial, sans-serif" font-size="7" fill="#333">\'\n        f\'{report.get("verdict", "UNKNOWN")}: {report.get("summary", {}).get("fail", 0)} fail, \'\n        f\'{report.get("summary", {}).get("warn", 0)} warn</text>\'\n    )\n    rows.append("</svg>")\n    _atomic_write_text(Path(path), "\\n".join(rows) + "\\n")\n\n\ndef _alphabetic_id(index: int) -> str:\n    output = ""\n    value = index + 1\n    while value:\n        value, remainder = divmod(value - 1, 26)\n        output = chr(ord("a") + remainder) + output\n    return output\n\n\ndef _font_is_bold(weight: Any) -> bool:\n    if isinstance(weight, (int, float)):\n        return float(weight) >= 600\n    return str(weight).lower() in {"bold", "semibold", "demibold", "heavy", "black"}\n\n\ndef _matplotlib_panel_label_anchor(ax: Any) -> tuple[str, list[float]] | None:\n    for artist in ax.texts:\n        label = artist.get_text().strip()\n        if not re.fullmatch(r"[a-z]", label) or not _font_is_bold(artist.get_fontweight()):\n            continue\n        display = artist.get_transform().transform(artist.get_position())\n        axes_xy = ax.transAxes.inverted().transform(display)\n        if not (-0.35 <= axes_xy[0] <= 0.2 and 0.8 <= axes_xy[1] <= 1.35):\n            continue\n        inches = ax.figure.dpi_scale_trans.inverted().transform(display)\n        return label, [float(inches[0] * 72), float(inches[1] * 72)]\n    return None\n\n\ndef matplotlib_layout_manifest(\n    fig: Any,\n    *,\n    axes: Sequence[Any] | None = None,\n    panel_ids: Mapping[Any, str] | Sequence[str] | None = None,\n    row_groups: Sequence[Any] | None = None,\n    column_groups: Sequence[Any] | None = None,\n    exclude_axes: Iterable[Any] = (),\n    exemptions: Sequence[Mapping[str, Any]] = (),\n) -> dict[str, Any]:\n    """Measure final Matplotlib axes rectangles in physical points."""\n    fig.canvas.draw()\n    width_in, height_in = (float(value) for value in fig.get_size_inches())\n    width_pt, height_pt = width_in * 72, height_in * 72\n    excluded = {id(axis) for axis in exclude_axes}\n    candidates = list(axes) if axes is not None else list(fig.axes)\n    selected: list[Any] = []\n    seen_subplot_cells: set[tuple[int, int, int]] = set()\n    for axis in candidates:\n        if id(axis) in excluded or not axis.get_visible():\n            continue\n        axis_label = str(axis.get_label() or "")\n        if axis_label.startswith("<colorbar"):\n            continue\n        try:\n            subplot_spec = axis.get_subplotspec()\n        except AttributeError:\n            subplot_spec = None\n        if subplot_spec is not None and panel_ids is None:\n            subplot_key = (\n                id(subplot_spec.get_gridspec()),\n                int(subplot_spec.num1),\n                int(subplot_spec.num2),\n            )\n            if subplot_key in seen_subplot_cells:\n                continue\n            seen_subplot_cells.add(subplot_key)\n        selected.append(axis)\n\n    if isinstance(panel_ids, Mapping):\n        ids = [str(panel_ids.get(axis) or "") for axis in selected]\n    elif panel_ids is not None:\n        ids = [str(value) for value in panel_ids]\n        if len(ids) != len(selected):\n            raise ValueError("panel_ids length must match the selected Matplotlib axes")\n    else:\n        ids = []\n        for index, axis in enumerate(selected):\n            candidate = str(axis.get_gid() or axis.get_label() or "").strip()\n            ids.append(candidate if candidate and not candidate.startswith("<") else _alphabetic_id(index))\n\n    grid_ids: dict[int, str] = {}\n    panels: list[dict[str, Any]] = []\n    for axis, panel_id in zip(selected, ids):\n        position = axis.get_position(original=False)\n        panel: dict[str, Any] = {\n            "id": panel_id,\n            "bbox_pt": [\n                float(position.x0 * width_pt),\n                float(position.y0 * height_pt),\n                float(position.x1 * width_pt),\n                float(position.y1 * height_pt),\n            ],\n        }\n        try:\n            subplot_spec = axis.get_subplotspec()\n        except AttributeError:\n            subplot_spec = None\n        if subplot_spec is not None:\n            grid_spec = subplot_spec.get_gridspec()\n            grid_key = id(grid_spec)\n            if grid_key not in grid_ids:\n                grid_ids[grid_key] = f"matplotlib-grid-{len(grid_ids) + 1}"\n            panel.update(\n                {\n                    "grid_id": grid_ids[grid_key],\n                    "row_start": int(subplot_spec.rowspan.start),\n                    "row_stop": int(subplot_spec.rowspan.stop),\n                    "col_start": int(subplot_spec.colspan.start),\n                    "col_stop": int(subplot_spec.colspan.stop),\n                }\n            )\n        label_anchor = _matplotlib_panel_label_anchor(axis)\n        if label_anchor is not None:\n            panel["panel_label"] = label_anchor[0]\n            panel["panel_label_anchor_pt"] = label_anchor[1]\n        panels.append(panel)\n\n    manifest: dict[str, Any] = {\n        "schema_version": SCHEMA_VERSION,\n        "backend": "python-matplotlib",\n        "figure": {"width_pt": width_pt, "height_pt": height_pt},\n        "panels": panels,\n        "exemptions": [dict(exemption) for exemption in exemptions],\n    }\n    if row_groups is not None:\n        manifest["row_groups"] = list(row_groups)\n    if column_groups is not None:\n        manifest["column_groups"] = list(column_groups)\n    return manifest\n\n\ndef require_matplotlib_panel_alignment(\n    fig: Any,\n    *,\n    json_out: str | Path | None = None,\n    overlay_svg: str | Path | None = None,\n    tolerance_pt: float = DEFAULT_TOLERANCE_PT,\n    gutter_tolerance_pt: float = DEFAULT_GUTTER_TOLERANCE_PT,\n    require_panel_labels: bool = False,\n    strict: bool = False,\n    **manifest_options: Any,\n) -> dict[str, Any]:\n    """Measure and block delivery when a Matplotlib multi-panel layout is misaligned."""\n    manifest = matplotlib_layout_manifest(fig, **manifest_options)\n    report = audit_layout_manifest(\n        manifest,\n        tolerance_pt=tolerance_pt,\n        gutter_tolerance_pt=gutter_tolerance_pt,\n        require_panel_labels=require_panel_labels,\n    )\n    if json_out is not None:\n        write_json_report(report, json_out)\n    if overlay_svg is not None and report.get("layout"):\n        write_overlay_svg(report, overlay_svg)\n    code = exit_code(report, strict=strict)\n    if code:\n        raise PanelAlignmentError(render_text(report, strict=strict))\n    return report\n\n\ndef run_self_tests() -> None:\n    aligned = {\n        "schema_version": 1,\n        "backend": "self-test",\n        "figure": {"width_pt": 300, "height_pt": 200},\n        "panels": [\n            {"id": "a", "bbox_pt": [20, 110, 130, 180], "grid_id": "g", "row_start": 0, "row_stop": 1, "col_start": 0, "col_stop": 1},\n            {"id": "b", "bbox_pt": [170, 110, 280, 180], "grid_id": "g", "row_start": 0, "row_stop": 1, "col_start": 1, "col_stop": 2},\n            {"id": "c", "bbox_pt": [20, 20, 130, 90], "grid_id": "g", "row_start": 1, "row_stop": 2, "col_start": 0, "col_stop": 1},\n            {"id": "d", "bbox_pt": [170, 20, 280, 90], "grid_id": "g", "row_start": 1, "row_stop": 2, "col_start": 1, "col_stop": 2},\n        ],\n    }\n    if exit_code(audit_layout_manifest(aligned)) != 0:\n        raise AssertionError("aligned self-test layout did not pass")\n    shifted = json.loads(json.dumps(aligned))\n    shifted["panels"][1]["bbox_pt"][1] -= 5\n    if exit_code(audit_layout_manifest(shifted)) != 1:\n        raise AssertionError("shifted self-test layout did not fail")\n    unequal_widths = {\n        "schema_version": 1,\n        "backend": "self-test",\n        "figure": {"width_pt": 320, "height_pt": 100},\n        "panels": [\n            {"id": "a", "bbox_pt": [10, 20, 70, 80], "grid_id": "g", "row_start": 0, "row_stop": 1, "col_start": 0, "col_stop": 1},\n            {"id": "b", "bbox_pt": [90, 20, 170, 80], "grid_id": "g", "row_start": 0, "row_stop": 1, "col_start": 1, "col_stop": 2},\n            {"id": "c", "bbox_pt": [190, 20, 290, 80], "grid_id": "g", "row_start": 0, "row_stop": 1, "col_start": 2, "col_stop": 3},\n        ],\n    }\n    unequal_report = audit_layout_manifest(unequal_widths)\n    if exit_code(unequal_report) != 1 or not any(\n        finding.get("kind") == "horizontal-panel-width-misalignment"\n        for finding in unequal_report.get("findings", [])\n    ):\n        raise AssertionError("unequal-width self-test row did not fail")\n\n\ndef build_parser() -> argparse.ArgumentParser:\n    parser = argparse.ArgumentParser(\n        description="Audit rendered multi-panel axes alignment from a JSON layout manifest."\n    )\n    parser.add_argument("layout", nargs="?", help="backend-neutral panel-layout JSON")\n    parser.add_argument("--json", action="store_true", help="print the report as JSON")\n    parser.add_argument("--json-out", help="write the report atomically as JSON")\n    parser.add_argument("--overlay-svg", help="write a QA-only SVG of measured panel rectangles")\n    parser.add_argument("--tolerance-pt", type=float, default=DEFAULT_TOLERANCE_PT)\n    parser.add_argument("--gutter-tolerance-pt", type=float, default=DEFAULT_GUTTER_TOLERANCE_PT)\n    parser.add_argument("--require-panel-labels", action="store_true")\n    parser.add_argument("--strict", action="store_true", help="make WARN findings blocking")\n    parser.add_argument("--self-test", action="store_true", help="run dependency-free core tests")\n    return parser\n\n\ndef main(argv: Sequence[str] | None = None) -> int:\n    parser = build_parser()\n    args = parser.parse_args(argv)\n    if args.self_test:\n        run_self_tests()\n        print("Panel alignment self-test: PASS")\n        return 0\n    if not args.layout:\n        parser.error("layout JSON is required unless --self-test is used")\n    try:\n        manifest = json.loads(Path(args.layout).read_text(encoding="utf-8"))\n        report = audit_layout_manifest(\n            manifest,\n            tolerance_pt=args.tolerance_pt,\n            gutter_tolerance_pt=args.gutter_tolerance_pt,\n            require_panel_labels=args.require_panel_labels,\n        )\n    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:\n        report = {\n            "schema_version": SCHEMA_VERSION,\n            "auditable": False,\n            "verdict": "NOT AUDITABLE",\n            "summary": {"fail": 0, "warn": 0, "comparisons": 0, "exemptions": 0},\n            "errors": [str(exc)],\n            "findings": [],\n        }\n    if args.json_out:\n        write_json_report(report, args.json_out)\n    if args.overlay_svg and report.get("layout"):\n        write_overlay_svg(report, args.overlay_svg)\n    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else render_text(report, strict=args.strict))\n    return exit_code(report, strict=args.strict)\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n\r\n'


def plot_cei_spatial_from_results(results_dir, county_path, future_year=2050):
    """Plot unscaled county CEI; read existing outputs without refitting models."""
    import json
    import types
    import matplotlib
    if "ipykernel" not in sys.modules:
        plt.switch_backend("Agg")
    started = time.perf_counter()
    out = Path(results_dir)
    files = sorted(out.glob("Future_CEI_AC_*.csv"))
    if not files:
        raise FileNotFoundError(f"No existing county CEI results: {out}")
    rows = []
    for file in files:
        data = pd.read_csv(file, usecols=["unique_id", "time", "Base_CEI", "Future_CEI", "Disease", "SSP"])
        if data.duplicated(["unique_id", "time", "Disease", "SSP"]).any():
            raise ValueError(f"Duplicate CEI county/year keys: {file.name}")
        rows.append(data)
    data = pd.concat(rows, ignore_index=True)
    if not np.isfinite(data[["Base_CEI", "Future_CEI"]].to_numpy()).all():
        raise ValueError("Nonfinite CEI values cannot be mapped as zero.")
    order = ["Circulation", "Digestive", "Respiration", "Blood", "Endocrine",
             "Skin", "Congenital", "Cancer", "Neural"]
    diseases = [d for d in order if d in set(data.Disease)]
    diseases += sorted(set(data.Disease) - set(diseases))
    # The fixed 2023 CEI is repeated across future years and SSPs.
    for (dis, uid), group in data.groupby(["Disease", "unique_id"]):
        if not np.allclose(group.Base_CEI, group.Base_CEI.iloc[0], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Inconsistent fixed 2023 CEI: {dis}/{uid}")
    baseline = data.drop_duplicates(["Disease", "unique_id"])[["Disease", "unique_id", "Base_CEI"]].rename(columns={"Base_CEI": "CEI"})
    baseline["Year"], baseline["Scenario"] = 2023, "Historical"
    future = data.loc[data.time == future_year, ["Disease", "unique_id", "Future_CEI", "SSP"]].rename(columns={"Future_CEI": "CEI", "SSP": "Scenario"})
    future["Year"] = future_year
    source = pd.concat([baseline, future], ignore_index=True)
    if source.empty:
        raise ValueError("No CEI observations available to map.")
    counties = gpd.read_file(county_path)
    if counties.crs is None:
        raise ValueError("County boundary CRS is undefined.")
    counties["unique_id"] = counties.index.astype("int64")
    counties = counties.to_crs("EPSG:4326")
    # Display-only simplification below the map pixel size; no CEI or county IDs change.
    counties.geometry = counties.geometry.simplify(0.002, preserve_topology=True)
    unknown = set(source.unique_id) - set(counties.unique_id)
    if unknown:
        raise ValueError(f"Unknown county shapefile indices: {sorted(unknown)[:10]}")
    provinces_path = Path(county_path).parent / "中国_省_面.shp"
    provinces = gpd.read_file(provinces_path).to_crs(counties.crs) if provinces_path.exists() else None
    context_path = Path(county_path).parent.parent / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
    context = gpd.read_file(context_path).to_crs(counties.crs) if context_path.exists() else None
    outline = counties.dissolve().boundary
    alignment = types.ModuleType("_cei_panel_alignment")
    exec(_CEI_ALIGNMENT_SOURCE, alignment.__dict__)
    # Same original-index linear color range for each disease across all four plates.
    scale = source.groupby("Disease").CEI.max().reindex(diseases).clip(lower=1e-12)
    minimum = source.groupby("Disease").CEI.min().reindex(diseases).clip(upper=0.0)
    def cei_norm(dis):
        lo, hi = float(minimum.loc[dis]), float(scale.loc[dis])
        return mcolors.TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi) if lo < 0 else mcolors.Normalize(0, hi)
    source["Colorbar_min"] = source.Disease.map(minimum)
    source["Colorbar_max"] = source.Disease.map(scale)
    source.to_csv(out/"03_CEI_spatial_source_data.csv", index=False, encoding="utf-8-sig")
    counts = []
    palettes = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cei_blue_red", ["#417CA1", "#A6C8D8", "#F5F3EF", "#DDAA9B", "#A44149"])
    settings = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                "font.size": 7, "axes.titlesize": 8, "pdf.fonttype": 42, "svg.fonttype": "none"}
    plates = [("Historical", 2023)] + [(sc, future_year) for sc in ["ssp126", "ssp245", "ssp370"]]
    with plt.rc_context(settings):
        for scenario, year in plates:
            selected = source.loc[(source.Scenario == scenario) & (source.Year == year)]
            if set(selected.Disease) != set(diseases):
                raise ValueError(f"Missing disease CEI for {scenario}/{year}")
            nrows = int(np.ceil(len(diseases)/3))
            fig, all_axes = plt.subplots(nrows, 3, figsize=(7.2, 1.85*nrows+.35), squeeze=False)
            fig.subplots_adjust(left=.025, right=.945, bottom=.055, top=.94, wspace=.34, hspace=.30)
            axes = []
            for ax, dis in zip(all_axes.ravel(), diseases):
                axes.append(ax)
                values = selected.loc[selected.Disease == dis, ["unique_id", "CEI"]]
                if values.unique_id.duplicated().any():
                    raise ValueError("Duplicate map county values.")
                mapped = counties.merge(values, on="unique_id", how="left", validate="one_to_one")
                norm = cei_norm(dis)
                if context is not None:
                    context.plot(ax=ax, facecolor="#FAFAFA", edgecolor="#C6C9CB", linewidth=.25, aspect=None, rasterized=True)
                mapped.plot(ax=ax, column="CEI", cmap=palettes, norm=norm,
                            edgecolor="#D4D9DD", linewidth=.10, aspect=None,
                            rasterized=True, missing_kwds={"color": "#E9E9E9", "edgecolor": "#D4D9DD"})
                if provinces is not None:
                    provinces.boundary.plot(ax=ax, color="#ADB4B9", linewidth=.28, aspect=None, rasterized=True)
                outline.plot(ax=ax, color="#989FA4", linewidth=.35, aspect=None, rasterized=True)
                ax.set(xlim=(73, 136), ylim=(17, 55), title=dis)
                ax.set_aspect(1.2, adjustable="box")
                ax.set_axis_off()
                counts.append({"Year": year, "Scenario": scenario, "Disease": dis,
                               "N_counties_with_CEI": len(values), "N_counties_grey": int(mapped.CEI.isna().sum()),
                               "Minimum_CEI": values.CEI.min(), "Maximum_CEI": values.CEI.max(),
                               "Colorbar_min": minimum.loc[dis], "Colorbar_max": scale.loc[dis]})
            for unused in all_axes.ravel()[len(diseases):]:
                unused.set_visible(False)
            # Add colorbars after axes rectangles have reached their final sizes.
            fig.canvas.draw()
            for ax, dis in zip(axes, diseases):
                box = ax.get_position()
                cax = fig.add_axes([box.x1+.006, box.y0+.03*box.height, .0065, .94*box.height])
                maximum = float(scale.loc[dis])
                sm = plt.cm.ScalarMappable(norm=cei_norm(dis), cmap=palettes)
                ticks = [float(minimum.loc[dis]), 0, maximum] if minimum.loc[dis] < 0 else [0, maximum]
                cb = fig.colorbar(sm, cax=cax, ticks=ticks)
                cb.ax.set_yticklabels([f"{v:.3g}" for v in ticks])
                cb.ax.tick_params(labelsize=5.5, length=1.5, width=.4, pad=1)
                cb.set_label("CEI", fontsize=6, rotation=270, labelpad=5)
                cb.outline.set_linewidth(.4)
            label = "2023 reference" if scenario == "Historical" else f"{SSP_LABELS[scenario]} | {year}"
            fig.text(.5, .985, label, ha="center", va="top", fontsize=9)
            fig.text(.5, .012, "Grey: no county estimate; disease-specific color ranges shared across years and scenarios",
                     ha="center", va="bottom", fontsize=6)
            name = f"03_CEI_Spatial_{year}_{scenario}"
            fig.canvas.draw()
            alignment.require_matplotlib_panel_alignment(fig, axes=axes,
                json_out=out/(name+".alignment.json"), strict=True,
                tolerance_pt=1.5, gutter_tolerance_pt=1.5)
            for ext in ["png", "pdf"]:
                fig.savefig(out/(name+"."+ext), dpi=400, bbox_inches="tight", pad_inches=.05, facecolor="white")
            fig.savefig(out/(name+".svg"), dpi=400, bbox_inches="tight", pad_inches=.05, facecolor="white")
            plt.close(fig)
            print(f"Saved {name}; elapsed {time.perf_counter()-started:.1f}s")
    pd.DataFrame(counts).to_csv(out/"03_CEI_spatial_coverage_and_scales.csv", index=False, encoding="utf-8-sig")
    (out/"03_CEI_spatial_description.md").write_text(
        "# County-level disease-specific CEI spatial distributions\n\n"
        "The 2023 fixed reference and 2050 SSP1-2.6, SSP2-4.5 and SSP3-7.0 CEI distributions "
        "are shown for every disease available in code_6.3_results. Values are the original "
        "county-level Base_CEI and Future_CEI, not IQR-scaled values or hospitalization changes. "
        "The fixed 2023 reference is deduplicated across future years and scenarios after a consistency check. "
        "Each disease uses the same minimum/zero/maximum color mapping across the four plates. Negative CEI values are retained. When both signs are present, a diverging scale is linear on each side of zero; "
        "different diseases have different scales. No percentile clipping, spatial smoothing or recalculation "
        "of the index is applied. Grey indicates counties without an estimate in the processed health panel. "
        "County/province boundaries are shown where the local source files provide coverage. The 2023 map "
        "covers counties retained in the future health results, rather than every historical inventory county. "
        "County geometry is simplified only for display with a 0.002-degree topology-preserving tolerance; no values or county IDs change. Map fills and boundaries are rasterized at 400 dpi while labels remain editable. Source data and county counts are provided in the accompanying CSV files.\n\n"
        "To draw these maps only: python code_6.3.py --cei-maps-only\n", encoding="utf-8")



def plot_cei_grid_scatter(future_year=2050):
    """Reconstruct original weighted grid CEI, without county aggregation."""
    import json
    import types
    from matplotlib.ticker import FuncFormatter
    started = time.perf_counter()
    if "ipykernel" not in sys.modules:
        plt.switch_backend("Agg")
    beta = pd.read_csv(BETA_PATH)
    diseases = beta.loc[beta.Significance == "Significant Harm", "Disease"].tolist()
    diseases = [d for d in diseases if d not in EXCLUDED_DISEASES]
    order = ["Circulation", "Digestive", "Respiration", "Blood", "Endocrine",
             "Skin", "Congenital", "Cancer", "Neural"]
    diseases = [d for d in order if d in diseases] + sorted(set(diseases)-set(order))
    if not diseases:
        raise ValueError("No eligible disease models")
    priority = pd.read_csv(PRIORITY_PATH)
    weights_by_disease = {}
    for disease in diseases:
        weights = {}
        for _, row in priority.loc[priority.Disease_Name == disease].iterrows():
            if pd.isna(row.Exposure):
                raise ValueError(f"{disease}: missing exposure")
            weight = float(row.get("Priority_Index", 1.0))
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"{disease}: invalid weight")
            for exp in re.split(r",\s*(?=X_)", str(row.Exposure).strip()):
                exp = exp.strip()
                if not exp:
                    continue
                exp = exp if exp.startswith("X_") else "X_"+exp
                weights[exp] = max(weights.get(exp, 0.), weight)
        if not weights:
            raise ValueError(f"{disease}: no weights")
        weights_by_disease[disease] = weights
    pollutants = sorted({e[2:] for weights in weights_by_disease.values() for e in weights})
    print(">>> Grid CEI scatter: loading historical standardization and original coordinates...")
    hist = pd.read_csv(HIST_PANEL_PATH, usecols=["lon", "lat", "time"]+pollutants)
    hist["time"] = integer_key(hist.time, "Historical years")
    hist = hist.loc[hist.time.between(2017, 2023)].copy()
    scaling = {pol: (float(hist[pol].mean()), float(hist[pol].std(ddof=1))) for pol in pollutants}
    for pol, (mean, sd) in scaling.items():
        if not np.isfinite([mean, sd]).all() or sd <= 0:
            raise ValueError(f"{pol}: invalid historical scaling")
    baseline = hist.loc[hist.time == 2023].copy()
    if baseline.empty:
        raise ValueError("No 2023 historical grid")
    header = pd.read_csv(FUT_PRED_DIR/"pred_results.csv", nrows=0).columns
    columns = {pol: resolve_prediction_column(header, pol) for pol in pollutants}
    future = pd.read_csv(FUT_PRED_DIR/"pred_results.csv",
                         usecols=list(dict.fromkeys(["lon", "lat", "time", "scenario"]+list(columns.values()))))
    future["time"] = integer_key(future.time, "Future years")
    future = future.loc[(future.time == future_year) & future.scenario.isin(SSPS)].copy()
    rows = []
    for scenario, year, frame, mapping in [
        ("Historical", 2023, baseline, {pol: pol for pol in pollutants})
    ] + [(sc, future_year, future.loc[future.scenario == sc], columns) for sc in SSPS]:
        if frame.empty or frame.duplicated(["lon", "lat", "time"]).any():
            raise ValueError(f"{scenario}: empty or duplicate grid coordinates")
        check_finite(frame, ["lon", "lat"]+list(mapping.values()), scenario)
        for disease in diseases:
            points = frame[["lon", "lat"]].copy()
            points["CEI"] = 0.
            for exposure, weight in weights_by_disease[disease].items():
                pol = exposure[2:]
                mean, sd = scaling[pol]
                points["CEI"] += ((frame[mapping[pol]]-mean)/sd)*weight
            points["Disease"], points["Scenario"], points["Year"] = disease, scenario, year
            rows.append(points)
    source = pd.concat(rows, ignore_index=True)
    if not np.isfinite(source[["lon", "lat", "CEI"]]).all().all():
        raise ValueError("Nonfinite grid CEI")
    source.to_csv(OUTPUT_DIR/"04_CEI_grid_scatter_source_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Pollutant": pol, "Historical_mean": mean, "Historical_sample_SD": sd,
                  "Reference_start": 2017, "Reference_end": 2023}
                  for pol, (mean, sd) in scaling.items()]).to_csv(
                      OUTPUT_DIR/"04_CEI_grid_standardization.csv", index=False, encoding="utf-8-sig")
    counties = gpd.read_file(SHP_PATH)
    if counties.crs is None:
        raise ValueError("Boundary CRS is undefined")
    counties = counties.to_crs("EPSG:4326")
    outline = counties.dissolve().boundary
    province_path = Path(SHP_PATH).parent/"中国_省_面.shp"
    provinces = gpd.read_file(province_path).to_crs(counties.crs) if province_path.exists() else None
    context_path = Path(SHP_PATH).parent.parent/"ne_110m_admin_0_countries"/"ne_110m_admin_0_countries.shp"
    context = gpd.read_file(context_path).to_crs(counties.crs) if context_path.exists() else None
    alignment = types.ModuleType("_cei_scatter_alignment")
    exec(_CEI_ALIGNMENT_SOURCE, alignment.__dict__)
    sequential = mcolors.LinearSegmentedColormap.from_list("cei_sequential", ["#DAEEF1", "#85BFC7", "#408AA7", "#576C93", "#B65457"])
    diverging = mcolors.LinearSegmentedColormap.from_list("cei_signed", ["#447DA5", "#9BC8D8", "#F5F3EF", "#DCA293", "#A74049"])
    limits = {d: (min(0., float(source.loc[source.Disease == d, "CEI"].min())),
                  max(0., float(source.loc[source.Disease == d, "CEI"].max()))) for d in diseases}
    counts = []
    settings = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                "font.size": 7, "axes.titlesize": 8, "pdf.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(settings):
        for scenario, year in [("Historical", 2023)]+[(sc, future_year) for sc in SSPS]:
            nrows = int(np.ceil(len(diseases)/3))
            fig, aa = plt.subplots(nrows, 3, figsize=(7.2, nrows*1.9+.15), squeeze=False)
            fig.subplots_adjust(left=.025, right=.945, bottom=.045, top=.95, wspace=.34, hspace=.34)
            axes = []
            artists = []
            for i, (ax, disease) in enumerate(zip(aa.ravel(), diseases)):
                sub = source.loc[(source.Disease == disease) & (source.Scenario == scenario)]
                lo, hi = limits[disease]
                if lo == hi:
                    lo, hi = lo-1e-12, hi+1e-12
                norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0., vmax=hi) if lo < 0 < hi else mcolors.Normalize(lo, hi)
                cmap = diverging if lo < 0 < hi else sequential
                if context is not None:
                    context.plot(ax=ax, facecolor="#FBFBFB", edgecolor="#C4C9CE", linewidth=.25, aspect=None, rasterized=True)
                ax.scatter(sub.lon, sub.lat, c=sub.CEI, s=1.6, marker="o", cmap=cmap,vmin=np.percentile(sub.CEI, 0.1), vmax=np.percentile(sub.CEI, 99.9),
                           norm=norm, edgecolors="none", alpha=1, rasterized=True, zorder=3)
                if provinces is not None:
                    provinces.boundary.plot(ax=ax, color="#B1B8BD", linewidth=.28, aspect=None, rasterized=True, zorder=4)
                outline.plot(ax=ax, color="#949DA3", linewidth=.35, aspect=None, rasterized=True, zorder=4)
                ax.set(xlim=(72, 136), ylim=(17, 55))
                ax.set_aspect(1/np.cos(np.deg2rad(35)), adjustable="box")
                ax.set_axis_off()
                ax.text(0, 1.03, f"{chr(97+i)}  {disease}", transform=ax.transAxes,
                        fontsize=8, fontweight="normal", ha="left", va="bottom")
                axes.append(ax)
                artists.append((norm, cmap, lo, hi))
                counts.append({"Scenario": scenario, "Year": year, "Disease": disease,
                               "N_grid_points": len(sub), "Minimum_CEI": float(sub.CEI.min()),
                               "Maximum_CEI": float(sub.CEI.max()), "Colorbar_min": lo, "Colorbar_max": hi})
            for ax in aa.ravel()[len(diseases):]:
                ax.set_visible(False)
            fig.canvas.draw()
            for ax, (norm, cmap, lo, hi) in zip(axes, artists):
                box = ax.get_position()
                cbax = fig.add_axes([box.x1+.008, box.y0+.06*box.height, .007, box.height*.88])
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax)
                ticks = [lo, 0., hi] if lo < 0 < hi else [lo, hi]
                cb.set_ticks(ticks)
                cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3g}"))
                cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
                cb.outline.set_linewidth(.35)
                cb.set_label("CEI", fontsize=6, rotation=270, labelpad=6)
            stem = OUTPUT_DIR/f"04_CEI_Grid_Scatter_{year}_{scenario}"
            fig.canvas.draw()
            alignment.require_matplotlib_panel_alignment(fig, axes=axes,
                json_out=str(stem)+".alignment.json", tolerance_pt=1.5,
                gutter_tolerance_pt=1.5, strict=True)
            for ext in ["png", "pdf", "svg"]:
                fig.savefig(str(stem)+"."+ext, dpi=400, bbox_inches="tight", pad_inches=.04)
            if "ipykernel" in sys.modules:
                plt.show()
            plt.close(fig)
            print("  saved", stem.name)
    pd.DataFrame(counts).to_csv(OUTPUT_DIR/"04_CEI_grid_scatter_coverage_and_scales.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR/"04_CEI_grid_scatter_description.md").write_text(
        "# Disease-specific grid CEI spatial distributions\n\n"
        "Original-coordinate scatter maps of the 2023 historical grid and the 2050 SSP1-2.6, SSP2-4.5 and SSP3-7.0 grids. "
        "CEI is the sum of pollutant-sector values standardized using their pooled 2017-2023 historical grid mean and sample standard deviation, multiplied by the original disease-specific Priority Index weights. "
        "The original disease eligibility and repeated-exposure maximum-weight rule are retained. No county aggregation, coordinate jitter, smoothing, interpolation or IQR rescaling is applied. "
        "All finite grid records are plotted. Negative values are retained and indicate below-reference weighted emissions, not protective health effects. "
        "Each disease uses the same color endpoints across historical and future plates; scales differ across diseases. Mixed-sign scales are linear on either side of zero. "
        "Boundaries provide geographic context only. The grid coverage is independent of the county health-analysis sample. "
        "These are emission indices, not personal exposure measurements or hospitalization estimates. Source values, historical standardization parameters and grid counts are supplied as CSV files.\n\n"
        "Run only these figures: python code_6.3.py --cei-scatter-only\n",
        encoding="utf-8")
    print(f"Grid CEI scatter completed in {time.perf_counter()-started:.1f}s")


# Draw grid scatter figures independently of the hospitalization calculation.
if "--cei-scatter-only" in sys.argv:
    plot_cei_grid_scatter(future_year=2050)
    raise SystemExit(0)

# Reuse existing CEI files when only a spatial figure is requested.
if "--cei-maps-only" in sys.argv:
    plot_cei_spatial_from_results(OUTPUT_DIR, SHP_PATH, future_year=2050)
    raise SystemExit(0)

# SciPy is required for risk interpolation, not for CEI-only maps.
from scipy.interpolate import interp1d

# ============================================================
# 1. 模型、疾病及权重
# ============================================================
print(">>> [1/4] 正在加载关联参数、疾病权重与RR曲线...")

df_beta = pd.read_csv(BETA_PATH)
require_columns(
    df_beta,
    ["Disease", "Significance", "LogRR"],
    "关联结果表",
)

sig_models = df_beta.loc[
    df_beta["Significance"] == "Significant Harm"
].copy()

if sig_models["Disease"].duplicated().any():
    raise ValueError("关联结果表存在重复疾病模型")

valid_diseases = [
    d for d in sig_models["Disease"].tolist()
    if d not in EXCLUDED_DISEASES
]
if not valid_diseases:
    raise ValueError("没有可用于计算的疾病模型")

# 无论查表文件是否存在，都定义线性备用系数
beta_dict = dict(zip(
    sig_models["Disease"],
    pd.to_numeric(sig_models["LogRR"], errors="raise"),
))

print(f" -> 本次疾病：{valid_diseases}")

USE_NONLINEAR = DOSE_RESPONSE_LOOKUP_PATH.exists()

if USE_NONLINEAR:
    df_dr_lookup = pd.read_csv(DOSE_RESPONSE_LOOKUP_PATH)

    if "CEI_IQR_scaled" not in df_dr_lookup.columns:
        df_dr_lookup = df_dr_lookup.rename(
            columns={"Exposure_Value": "CEI_IQR_scaled"}
        )

    require_columns(
        df_dr_lookup,
        ["Disease", "CEI_IQR_scaled", "RR"],
        "RR曲线表",
    )
    print(" -> 使用疾病特异非线性RR曲线")
else:
    df_dr_lookup = pd.DataFrame()
    print(" -> 未找到RR曲线表，使用线性模型")

df_priority = pd.read_csv(PRIORITY_PATH)
require_columns(
    df_priority,
    ["Disease_Name", "Exposure"],
    "优先级表",
)

cei_weights = {}

for disease in valid_diseases:
    rows = df_priority.loc[
        df_priority["Disease_Name"] == disease
    ]
    weights = {}

    for _, row in rows.iterrows():
        if pd.isna(row["Exposure"]):
            raise ValueError(f"{disease}存在缺失Exposure")

        # 保留原来的拆分方式，避免拆开化学名称内部的逗号
        exposures = re.split(
            r",\s*(?=X_)", str(row["Exposure"]).strip()
        )
        weight = float(row.get("Priority_Index", 1.0))

        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"{disease}存在无效Priority Index")

        for exposure in exposures:
            exposure = exposure.strip()
            if not exposure:
                continue
            if not exposure.startswith("X_"):
                exposure = f"X_{exposure}"

            # 保留原规则：同一疾病、同一Exposure重复时取最大权重
            weights[exposure] = max(
                weights.get(exposure, 0.0), weight
            )

    if not weights:
        raise ValueError(f"{disease}没有疾病特异权重")
    cei_weights[disease] = weights


# ============================================================
# 2. 历史标准化、IQR、基线CEI及历史合并住院率
# ============================================================
print(">>> [2/4] 正在提取历史标度与基线...")

df_hist_grid = pd.read_csv(HIST_PANEL_PATH)
require_columns(df_hist_grid, ["time"], "历史栅格表")
df_hist_grid["time"] = integer_key(
    df_hist_grid["time"], "历史栅格年份"
)

df_hist_period_grid = df_hist_grid.loc[
    df_hist_grid["time"].between(2017, 2023)
].copy()

if df_hist_period_grid.empty:
    raise ValueError("历史栅格表没有2017–2023年数据")

all_needed_pols = sorted({
    exposure[2:]
    for weights in cei_weights.values()
    for exposure in weights
})

z_params = {}
for pollutant in all_needed_pols:
    require_columns(
        df_hist_period_grid, [pollutant], "历史栅格表"
    )
    values = pd.to_numeric(
        df_hist_period_grid[pollutant], errors="raise"
    )

    # 沿用原代码：历史均值和样本标准差，忽略缺失值
    mean_v = float(values.mean())
    std_v = float(values.std(ddof=1))

    if not np.isfinite(mean_v) or not np.isfinite(std_v) or std_v <= 0:
        raise ValueError(f"{pollutant}历史均值或标准差无效")

    z_params[pollutant] = {"mean": mean_v, "std": std_v}

county_shp = gpd.read_file(SHP_PATH)
require_columns(county_shp, ["gb"], "县域Shapefile")

# 保留原来的Shapefile行索引ID，与mapping.csv对应
county_shp["unique_id"] = county_shp.index.astype("int64")
county_shp["B_county_id"] = integer_key(
    county_shp["gb"].astype(str).str[3:],
    "Shapefile行政区划代码",
)

if county_shp["B_county_id"].duplicated().any():
    raise ValueError(
        "Shapefile存在重复行政区划代码，请检查县域映射"
    )

county_keys = county_shp[
    ["B_county_id", "unique_id"]
].copy()

df_hist_county = pd.read_csv(HIST_COUNTY_PATH)

# 兼容旧表把行政区划代码命名为unique_id的情况
if "B_county_id" not in df_hist_county.columns:
    require_columns(
        df_hist_county, ["unique_id"], "历史县域表"
    )
    df_hist_county = df_hist_county.rename(
        columns={"unique_id": "B_county_id"}
    )
else:
    # unique_id以本次Shapefile映射为准，避免合并后产生后缀
    df_hist_county = df_hist_county.drop(
        columns=["unique_id"], errors="ignore"
    )

require_columns(
    df_hist_county,
    ["B_county_id", "B_year", "C_POP"],
    "历史县域表",
)

df_hist_county["B_county_id"] = integer_key(
    df_hist_county["B_county_id"], "历史行政区划代码"
)
df_hist_county["B_year"] = integer_key(
    df_hist_county["B_year"], "历史县域年份"
)

df_hist_period_county = df_hist_county.loc[
    df_hist_county["B_year"].between(2017, 2023)
].copy()

if df_hist_period_county.duplicated(
    ["B_county_id", "B_year"]
).any():
    raise ValueError("历史县域表存在重复县域×年份记录")

df_hist_period_county = df_hist_period_county.merge(
    county_keys,
    on="B_county_id",
    how="left",
    validate="many_to_one",
)

unmapped = df_hist_period_county.loc[
    df_hist_period_county["unique_id"].isna(),
    "B_county_id",
].nunique()
print(f" -> 历史行政区划代码未匹配县域：{unmapped}")

cei_params = {}

for disease in valid_diseases:
    cei_col = f"CEI_{disease}"
    outcome_col = f"Y_{disease}"
    require_columns(
        df_hist_period_county,
        [cei_col, outcome_col],
        "历史县域表",
    )

    # 关键修正：2023年CEI从完整面板提取，
    # 不要求2023年住院数存在
    df_base = (
        df_hist_period_county.loc[
            df_hist_period_county["B_year"] == 2023,
            ["unique_id", cei_col],
        ]
        .dropna(subset=["unique_id", cei_col])
        .rename(columns={cei_col: "Base_CEI"})
        .copy()
    )
    df_base["unique_id"] = integer_key(
        df_base["unique_id"], "基线县域ID"
    )
    check_finite(df_base, ["Base_CEI"], disease)

    if df_base["unique_id"].duplicated().any():
        raise ValueError(f"{disease}存在重复2023年基线CEI")

    # 保留原来的历史有效记录规则，用于IQR及合并住院率
    df_clean = df_hist_period_county.dropna(
        subset=[cei_col, outcome_col, "C_POP"]
    ).copy()
    df_clean = df_clean.loc[df_clean["C_POP"] > 0].copy()

    check_finite(
        df_clean, [cei_col, outcome_col, "C_POP"], disease
    )
    if (df_clean[outcome_col] < 0).any():
        raise ValueError(f"{disease}存在负住院数")

    iqr = float(
        df_clean[cei_col].quantile(0.75)
        - df_clean[cei_col].quantile(0.25)
    )
    if not np.isfinite(iqr) or iqr <= 0:
        raise ValueError(f"{disease}历史CEI的IQR无效")

    # 如曲线表记录了IQR，检查两套缩放是否一致
    if USE_NONLINEAR and "IQR" in df_dr_lookup.columns:
        lookup_iqr = pd.to_numeric(
            df_dr_lookup.loc[
                df_dr_lookup["Disease"] == disease, "IQR"
            ],
            errors="raise",
        ).dropna().to_numpy()

        if lookup_iqr.size and not np.allclose(
            lookup_iqr, iqr, rtol=1e-6, atol=1e-10
        ):
            raise ValueError(
                f"{disease}：历史计算IQR={iqr}与RR曲线IQR"
                "不一致，请核对输入文件"
            )

    # I0 = 2017–2023有效记录的住院数之和 / 人口之和
    grouped = (
        df_clean.dropna(subset=["unique_id"])
        .groupby("unique_id", as_index=False)[
            [outcome_col, "C_POP"]
        ]
        .sum()
    )
    grouped["unique_id"] = integer_key(
        grouped["unique_id"], "住院率县域ID"
    )
    grouped["Incidence_0"] = (
        grouped[outcome_col] / grouped["C_POP"]
    )

    cei_params[disease] = {
        "iqr": iqr,
        "base_cei_df": df_base,
        "incidence_df": grouped[
            ["unique_id", "Incidence_0"]
        ],
    }

    print(
        f" -> {disease}: IQR={iqr:.6g}；"
        f"2023年CEI县域={len(df_base):,}；"
        f"历史住院率县域={len(grouped):,}"
    )


# ============================================================
# 3. 未来CEI及同一未来人口下的住院变化
# ============================================================
print("\n>>> [3/4] 正在计算未来CEI和模型估计住院变化...")

pred_df = pd.read_csv(FUT_PRED_DIR / "pred_results.csv")
require_columns(
    pred_df, ["scenario", "lat", "lon", "time"], "预测排放表"
)
pred_df["time"] = integer_key(pred_df["time"], "预测年份")
check_finite(pred_df, ["lat", "lon"], "预测排放表")

mapping_df = pd.read_csv(
    FUT_PRED_DIR / "mapping.csv",
    usecols=["lon", "lat", "unique_id"],
).drop_duplicates()
mapping_df["unique_id"] = integer_key(
    mapping_df["unique_id"], "栅格映射县域ID"
)
check_finite(mapping_df, ["lon", "lat"], "栅格映射表")

unknown_ids = set(mapping_df["unique_id"]) - set(
    county_shp["unique_id"]
)
if unknown_ids:
    raise ValueError(
        "mapping.csv包含当前Shapefile中不存在的县域ID"
    )

population_panels = {
    ssp: load_population_skeleton(ssp) for ssp in SSPS
}

# 本次结果在内存中收集，避免合并旧文件
results_by_ssp = {ssp: [] for ssp in SSPS}
summary_rows = []
coverage_rows = []
plot_cei_data = []
plot_ac_data = []
map_data_2050 = []

for disease in valid_diseases:
    print(f"\n{'=' * 60}\n正在处理：{disease}")
    params = cei_params[disease]
    iqr = params["iqr"]

    # 非线性RR查表；保留端点保持规则
    rr_func = None
    lookup_min = lookup_max = np.nan

    if USE_NONLINEAR:
        d_lookup = df_dr_lookup.loc[
            df_dr_lookup["Disease"] == disease,
            ["CEI_IQR_scaled", "RR"],
        ].copy()

        if not d_lookup.empty:
            check_finite(
                d_lookup, ["CEI_IQR_scaled", "RR"], disease
            )
            d_lookup = d_lookup.sort_values("CEI_IQR_scaled")

            if (
                len(d_lookup) < 2
                or d_lookup["CEI_IQR_scaled"].duplicated().any()
                or (d_lookup["RR"] <= 0).any()
            ):
                raise ValueError(f"{disease}的RR曲线无效")

            lookup_min = float(
                d_lookup["CEI_IQR_scaled"].iloc[0]
            )
            lookup_max = float(
                d_lookup["CEI_IQR_scaled"].iloc[-1]
            )

            rr_func = interp1d(
                d_lookup["CEI_IQR_scaled"].to_numpy(),
                d_lookup["RR"].to_numpy(),
                kind="linear",
                bounds_error=False,
                fill_value=(
                    float(d_lookup["RR"].iloc[0]),
                    float(d_lookup["RR"].iloc[-1]),
                ),
                assume_sorted=True,
            )
        else:
            print(" -> 该疾病缺少RR曲线，使用线性备用系数")

    for ssp in SSPS:
        df_fut = population_panels[ssp].copy()
        n_before = len(df_fut)
        n_counties_before = df_fut["unique_id"].nunique()

        df_ssp = pred_df.loc[
            pred_df["scenario"] == ssp
        ].copy()

        if df_ssp.empty:
            raise ValueError(f"{ssp}没有预测排放数据")
        if df_ssp.duplicated(["lat", "lon", "time"]).any():
            raise ValueError(f"{ssp}存在重复栅格×年份")

        df_ssp["Future_CEI_grid"] = 0.0

        for exposure, weight in cei_weights[disease].items():
            pollutant = exposure[2:]
            pred_col = resolve_prediction_column(
                df_ssp.columns, pollutant
            )
            check_finite(df_ssp, [pred_col], f"{disease}, {ssp}")

            z = (
                df_ssp[pred_col] - z_params[pollutant]["mean"]
            ) / z_params[pollutant]["std"]

            df_ssp["Future_CEI_grid"] += z * weight

        # Preserve original grid CEI before county averaging.
        grid_cei_output = df_ssp[["lat", "lon", "time", "Future_CEI_grid"]].copy()
        grid_cei_output["Disease"], grid_cei_output["SSP"] = disease, ssp
        grid_cei_output.to_csv(
            OUTPUT_DIR / f"Future_CEI_grid_{disease}_{ssp}.csv",
            index=False, encoding="utf-8-sig",
        )

        # 栅格CEI均值聚合到县域，保留原来的空间汇总方法
        mapped = df_ssp[
            ["lat", "lon", "time", "Future_CEI_grid"]
        ].merge(
            mapping_df,
            on=["lon", "lat"],
            how="inner",
        )

        county_cei = (
            mapped.groupby(
                ["unique_id", "time"], as_index=False
            )["Future_CEI_grid"]
            .mean()
            .rename(columns={"Future_CEI_grid": "Future_CEI"})
        )

        # 所有匹配均按键进行
        df_fut = df_fut.merge(
            county_cei,
            on=["unique_id", "time"],
            how="left",
            validate="one_to_one",
        )
        df_fut = df_fut.merge(
            params["base_cei_df"],
            on="unique_id",
            how="left",
            validate="many_to_one",
        )
        df_fut = df_fut.merge(
            params["incidence_df"],
            on="unique_id",
            how="left",
            validate="many_to_one",
        )

        required = ["Future_CEI", "Base_CEI", "Incidence_0"]
        missing_counts = df_fut[required].isna().sum()
        df_fut = df_fut.dropna(subset=required).copy()

        if df_fut.empty:
            raise ValueError(f"{disease}, {ssp}没有完整县年记录")

        check_finite(df_fut, required, f"{disease}, {ssp}")
        if (df_fut["Incidence_0"] < 0).any():
            raise ValueError("基线住院率不能为负数")

        print(
            f"  {ssp}: 融合{len(cei_weights[disease])}个组分；"
            f"保留{df_fut['unique_id'].nunique():,}县、"
            f"{len(df_fut):,}行；排除{n_before-len(df_fut):,}行"
        )
        print(f"    各列缺失数：{missing_counts.to_dict()}")

        df_fut["Base_CEI_scaled"] = df_fut["Base_CEI"] / iqr
        df_fut["Future_CEI_scaled"] = df_fut["Future_CEI"] / iqr
        df_fut["Delta_CEI_Scaled"] = (
            df_fut["Future_CEI_scaled"]
            - df_fut["Base_CEI_scaled"]
        )

        base_outside = future_outside = np.nan

        if rr_func is not None:
            for prefix, col in [
                ("Base", "Base_CEI_scaled"),
                ("Future", "Future_CEI_scaled"),
            ]:
                outside = (
                    (df_fut[col] < lookup_min)
                    | (df_fut[col] > lookup_max)
                )
                df_fut[f"{prefix}_CEI_outside_lookup"] = outside

                if prefix == "Base":
                    base_outside = float(outside.mean())
                else:
                    future_outside = float(outside.mean())

                if outside.any():
                    print(
                        f"    {prefix} CEI超出曲线范围："
                        f"{outside.sum():,}行"
                        f"（{outside.mean():.1%}），使用端点RR"
                    )

            df_fut["RR_base"] = rr_func(
                df_fut["Base_CEI_scaled"].to_numpy()
            )
            df_fut["RR_future"] = rr_func(
                df_fut["Future_CEI_scaled"].to_numpy()
            )

            check_finite(
                df_fut, ["RR_base", "RR_future"], disease
            )
            if (
                (df_fut["RR_base"] <= 0)
                | (df_fut["RR_future"] <= 0)
            ).any():
                raise ValueError(f"{disease}存在非正RR")

            # 不将RR截断为最低1
            rr_ratio = (
                df_fut["RR_future"] / df_fut["RR_base"]
            )
            risk_method = "nonlinear_lookup"
        else:
            beta = beta_dict.get(disease)
            if beta is None or not np.isfinite(beta):
                raise ValueError(f"{disease}缺少有效线性系数")

            rr_ratio = np.exp(
                beta * df_fut["Delta_CEI_Scaled"]
            )
            risk_method = "linear"

        if (
            ~np.isfinite(rr_ratio)
            | (rr_ratio <= 0)
        ).any():
            raise ValueError(f"{disease}, {ssp}风险比无效")

        # ----------------------------------------------------
        # 正确的共同基线换算
        #
        # I_future = I0 × RR_future / RR_2023
        #
        # Delta_H = P_future × I0
        #           × (RR_future / RR_2023 - 1)
        #
        # 比较两种CEI时均使用同一未来人口。
        # 历史合并住院率I0近似2023年基线。
        # ----------------------------------------------------
        df_fut["RR_ratio_vs_2023"] = rr_ratio
        df_fut["Relative_change_vs_2023"] = rr_ratio - 1

        df_fut["Incidence"] = (
            df_fut["Incidence_0"] * rr_ratio
        )
        df_fut["Baseline_Cases"] = (
            df_fut["population"] * df_fut["Incidence_0"]
        )
        df_fut["Total_Cases"] = (
            df_fut["population"] * df_fut["Incidence"]
        )

        ac_col = f"AC_{disease}"
        af_col = f"AF_{disease}"

        # AC、AF保留旧列名兼容下游，不再表示传统PAF及归因病例
        df_fut[af_col] = df_fut["Relative_change_vs_2023"]
        df_fut[ac_col] = (
            df_fut["Baseline_Cases"]
            * df_fut["Relative_change_vs_2023"]
        )
        df_fut[f"Delta_H_{disease}"] = df_fut[ac_col]

        check_finite(
            df_fut,
            ["Incidence", "Baseline_Cases", "Total_Cases", ac_col],
            f"{disease}, {ssp}病例计算",
        )

        df_fut["Risk_method"] = risk_method
        df_fut["SSP"] = ssp
        df_fut["Disease"] = disease

        df_fut = df_fut.sort_values(
            ["unique_id", "time"]
        ).reset_index(drop=True)

        df_fut.to_csv(
            OUTPUT_DIR / f"Future_CEI_AC_{disease}_{ssp}.csv",
            index=False,
        )
        results_by_ssp[ssp].append(df_fut)

        # 汇总实际纳入县域，不称为未经外推的全国总量
        yearly = (
            df_fut.groupby("time", as_index=False)
            .agg(
                Future_CEI=("Future_CEI", "mean"),
                Population=("population", "sum"),
                Baseline_Cases=("Baseline_Cases", "sum"),
                Future_Cases=("Total_Cases", "sum"),
                Delta_H=(ac_col, "sum"),
                N_counties=("unique_id", "nunique"),
            )
        )
        yearly["Disease"] = disease
        yearly["SSP"] = ssp
        summary_rows.append(yearly)

        plot_cei_data.append(
            yearly[["time", "Future_CEI", "Disease", "SSP"]]
        )
        plot_ac_data.append(
            yearly[["time", "Delta_H", "Disease", "SSP"]]
            .rename(columns={"Delta_H": ac_col})
        )

        if (df_fut["time"] == 2050).any():
            map_data_2050.append(
                df_fut.loc[df_fut["time"] == 2050].copy()
            )

        coverage_rows.append({
            "Disease": disease,
            "SSP": ssp,
            "Risk_method": risk_method,
            "N_counties_before": n_counties_before,
            "N_counties_after": df_fut["unique_id"].nunique(),
            "N_rows_before": n_before,
            "N_rows_after": len(df_fut),
            "Missing_Future_CEI": int(missing_counts["Future_CEI"]),
            "Missing_Base_CEI": int(missing_counts["Base_CEI"]),
            "Missing_Incidence_0": int(missing_counts["Incidence_0"]),
            "Base_outside_fraction": base_outside,
            "Future_outside_fraction": future_outside,
        })


# ============================================================
# 本次结果立即合并，避免绘图失败后合并表仍为旧版本
# ============================================================
for ssp in SSPS:
    combined = pd.concat(
        results_by_ssp[ssp], ignore_index=True
    )
    combined.to_csv(
        OUTPUT_DIR / f"Combined_Future_CEI_AC_{ssp}.csv",
        index=False,
    )
    print(f" -> 已更新{ssp}合并表：{len(combined):,}行")

df_summary = pd.concat(summary_rows, ignore_index=True)
df_summary.to_csv(
    OUTPUT_DIR / "Summary_CEI_Hospitalization_Changes.csv",
    index=False,
)
pd.DataFrame(coverage_rows).to_csv(
    OUTPUT_DIR / "Calculation_Coverage_and_Boundary_Checks.csv",
    index=False,
)

print(
    "\n计算文件已保存。正AC表示模型估计住院增加，"
    "负AC表示模型估计住院减少。"
)


# ============================================================
# 4. 保留趋势图、2050年地图
# ============================================================
if MAKE_FIGURES:
    print("\n>>> [4/4] 正在绘制趋势图和2050年地图...")

    df_plot_cei = pd.concat(plot_cei_data, ignore_index=True)
    df_plot_ac = pd.concat(plot_ac_data, ignore_index=True)

    for disease in valid_diseases:
        fig, axes = plt.subplots(
            1, 2, figsize=(11, 4), constrained_layout=True
        )
        ac_col = f"AC_{disease}"

        for ssp in SSPS:
            cei = df_plot_cei.loc[
                (df_plot_cei["Disease"] == disease)
                & (df_plot_cei["SSP"] == ssp)
            ].sort_values("time")

            ac = df_plot_ac.loc[
                (df_plot_ac["Disease"] == disease)
                & (df_plot_ac["SSP"] == ssp)
            ].sort_values("time")

            axes[0].plot(
                cei["time"], cei["Future_CEI"],
                color=SSP_COLORS[ssp],
                lw=1.8,
                label=SSP_LABELS[ssp],
            )
            axes[1].plot(
                ac["time"], ac[ac_col],
                color=SSP_COLORS[ssp],
                lw=1.8,
                label=SSP_LABELS[ssp],
            )

        axes[0].set_title(f"a  CEI trajectory: {disease}", loc="left")
        axes[0].set_ylabel("Mean CEI across included counties")

        axes[1].set_title(
            f"b  Modeled hospitalization change: {disease}",
            loc="left",
        )
        axes[1].set_ylabel("Change vs fixed 2023 CEI")
        axes[1].axhline(0, color="#555555", ls="--", lw=0.8)
        axes[1].ticklabel_format(
            axis="y", style="sci", scilimits=(-3, 4)
        )

        for ax in axes:
            ax.set_xlabel("Year")
            ax.grid(
                axis="y", ls=":", color="#D8D8D8", lw=0.6
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend()

        save_figure(fig, f"01_Trends_CEI_AC_{disease}")

    if map_data_2050:
        if county_shp.crs is None:
            raise ValueError("县域Shapefile缺少CRS，无法正确绘图")

        china_map = county_shp.to_crs("EPSG:4326").copy()
        print(" -> 正在生成地图外轮廓...")
        china_boundary = china_map.dissolve()
        df_map_all = pd.concat(map_data_2050, ignore_index=True)

        # 保留跨疾病、跨情景统一的对称色标
        all_map_values = np.concatenate([
            df_map_all.loc[
                df_map_all["Disease"] == d, f"AC_{d}"
            ].dropna().to_numpy()
            for d in valid_diseases
        ])
        vabs = max(float(np.max(np.abs(all_map_values))), 1.0)
        norm = mcolors.SymLogNorm(
            linthresh=10,
            linscale=1,
            vmin=-vabs,
            vmax=vabs,
            base=10,
        )

        for disease in valid_diseases:
            ac_col = f"AC_{disease}"

            for ssp in SSPS:
                df_map = df_map_all.loc[
                    (df_map_all["Disease"] == disease)
                    & (df_map_all["SSP"] == ssp),
                    ["unique_id", ac_col],
                ]

                if df_map["unique_id"].duplicated().any():
                    raise ValueError(
                        f"{disease}, {ssp}地图存在重复县域"
                    )

                gdf_map = china_map.merge(
                    df_map,
                    on="unique_id",
                    how="left",
                    validate="one_to_one",
                )
                data_layer = gdf_map.dropna(subset=[ac_col])

                fig, ax = plt.subplots(figsize=(9, 7))

                china_map.plot(
                    ax=ax, color="#F1F1F1",
                    edgecolor="#BBBBBB", linewidth=0.2,
                )
                data_layer.plot(
                    ax=ax, column=ac_col,
                    cmap="RdBu_r", norm=norm, legend=False,
                )
                china_boundary.boundary.plot(
                    ax=ax, color="#333333", linewidth=0.8,
                )

                ax.set_xlim(73, 136)
                ax.set_ylim(17, 55)
                ax.set_title(
                    f"{disease} | {SSP_LABELS[ssp]} | 2050\n"
                    "Modeled hospitalization change vs fixed 2023 CEI"
                )
                ax.axis("off")

                # 保留南海区域小图
                inset = ax.inset_axes([0.80, 0.02, 0.16, 0.28])
                china_map.plot(
                    ax=inset, color="#F1F1F1",
                    edgecolor="#BBBBBB", linewidth=0.1,
                )
                data_layer.plot(
                    ax=inset, column=ac_col,
                    cmap="RdBu_r", norm=norm, legend=False,
                )
                china_boundary.boundary.plot(
                    ax=inset, color="#333333", linewidth=0.5,
                )
                inset.set_xlim(105, 125)
                inset.set_ylim(2, 25)
                inset.set_xticks([])
                inset.set_yticks([])

                sm = plt.cm.ScalarMappable(
                    norm=norm, cmap="RdBu_r"
                )
                sm.set_array([])
                cbar = fig.colorbar(
                    sm, ax=ax, orientation="horizontal",
                    fraction=0.045, pad=0.055, shrink=0.8,
                )
                cbar.set_label(
                    "Modeled change in hospitalizations "
                    "(blue: decrease; red: increase)"
                )

                # 灰色区域表示没有纳入计算，不代表零变化
                fig.text(
                    0.5, 0.02,
                    "Grey: no estimate; color scale: symmetric logarithmic",
                    ha="center", fontsize=8, color="#555555",
                )
                fig.subplots_adjust(bottom=0.12, top=0.90)

                save_figure(
                    fig,
                    f"02_Spatial_Map_2050_{disease}_{ssp}",
                )

elapsed = time.perf_counter() - START_TIME
print(f"\n完成，用时：{elapsed / 60:.2f}分钟")
print(f"结果目录：{OUTPUT_DIR}")
# Additional county CEI spatial plates (original index scale).
if MAKE_FIGURES:
    plot_cei_spatial_from_results(OUTPUT_DIR, SHP_PATH, future_year=2050)

# Original-coordinate grid scatter plates, without county aggregation.
if MAKE_FIGURES:
    plot_cei_grid_scatter(future_year=2050)
#%%
