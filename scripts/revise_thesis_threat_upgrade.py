#!/usr/bin/env python3
"""Add verified flow and endpoint upgrade results to the revised thesis DOCX.

The input document is never overwritten. LibreOffice refreshes navigation fields in
an isolated profile, so the output package may be normalized during the final save.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import tempfile
import time
from pathlib import Path
from subprocess import DEVNULL, Popen, TimeoutExpired
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

etree: Any = importlib.import_module("lxml.etree")

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "final_Thesis_revised_public_scenarios.docx"
OUTPUT = ROOT / "final_Thesis_revised_threat_upgrade.docx"
FLOW_METRICS = ROOT / "models" / "flow" / "metrics.json"
ENDPOINT_METRICS = ROOT / "models" / "endpoint" / "novelty_metrics.json"
FIXED_FPR_SUMMARY = ROOT / "report" / "detection_coverage_fixed_fpr_summary.json"
INITIAL_PUBLIC_SUMMARY = ROOT / "report" / "public_scenario_evaluation" / "summary.json"
INTERNAL_SUMMARY = ROOT / "models" / "detection_evaluation.json"

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W = f"{{{NS['w']}}}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read verified result {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected an object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(paragraph: Any) -> str:
    return "".join(paragraph.itertext()).strip()


def _style(paragraph: Any) -> str | None:
    style = paragraph.find(f"{W}pPr/{W}pStyle")
    return None if style is None else style.get(f"{W}val")


def _find(root: Any, start: str, *, style: str | None = None) -> Any:
    matches = [
        paragraph
        for paragraph in root.iter(W + "p")
        if _text(paragraph).startswith(start) and (style is None or _style(paragraph) == style)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one paragraph starting {start!r} style={style!r}, found {len(matches)}"
        )
    return matches[0]


def _set_text(paragraph: Any, text: str, *, bold: bool = False) -> None:
    properties = paragraph.find(W + "pPr")
    for child in list(paragraph):
        if child is not properties:
            paragraph.remove(child)
    run = etree.SubElement(paragraph, W + "r")
    if bold:
        run_properties = etree.SubElement(run, W + "rPr")
        etree.SubElement(run_properties, W + "b")
    text_element = etree.SubElement(run, W + "t")
    text_element.set(XML_SPACE, "preserve")
    text_element.text = text


def _clone(template: Any, text: str, *, bold: bool = False) -> Any:
    paragraph = copy.deepcopy(template)
    _set_text(paragraph, text, bold=bold)
    return paragraph


def _insert_before(anchor: Any, elements: list[Any]) -> None:
    parent = anchor.getparent()
    if parent is None:
        raise RuntimeError("Anchor paragraph has no parent")
    index = parent.index(anchor)
    for element in elements:
        parent.insert(index, element)
        index += 1


def _cell(text: str, *, header: bool = False) -> Any:
    cell = etree.Element(W + "tc")
    cell_properties = etree.SubElement(cell, W + "tcPr")
    width = etree.SubElement(cell_properties, W + "tcW")
    width.set(W + "w", "0")
    width.set(W + "type", "auto")
    paragraph = etree.SubElement(cell, W + "p")
    _set_text(paragraph, text, bold=header)
    return cell


def _table(rows: list[list[str]]) -> Any:
    table = etree.Element(W + "tbl")
    properties = etree.SubElement(table, W + "tblPr")
    borders = etree.SubElement(properties, W + "tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = etree.SubElement(borders, W + edge)
        border.set(W + "val", "single")
        border.set(W + "sz", "4")
        border.set(W + "color", "auto")
    for row_index, values in enumerate(rows):
        row = etree.SubElement(table, W + "tr")
        row_properties = etree.SubElement(row, W + "trPr")
        etree.SubElement(row_properties, W + "cantSplit")
        if row_index == 0:
            etree.SubElement(row_properties, W + "tblHeader")
        for value in values:
            row.append(_cell(value, header=row_index == 0))
    return table


def _page_break_before(paragraph: Any) -> None:
    properties = paragraph.find(W + "pPr")
    if properties is None:
        properties = etree.Element(W + "pPr")
        paragraph.insert(0, properties)
    etree.SubElement(properties, W + "pageBreakBefore")


def _index_entry(template: Any, label: str, page: int) -> Any:
    paragraph = copy.deepcopy(template)
    properties = paragraph.find(W + "pPr")
    for child in list(paragraph):
        if child is not properties:
            paragraph.remove(child)
    label_run = etree.SubElement(paragraph, W + "r")
    label_text = etree.SubElement(label_run, W + "t")
    label_text.set(XML_SPACE, "preserve")
    label_text.text = label
    tab_run = etree.SubElement(paragraph, W + "r")
    etree.SubElement(tab_run, W + "tab")
    page_run = etree.SubElement(paragraph, W + "r")
    page_text = etree.SubElement(page_run, W + "t")
    page_text.text = str(page)
    return paragraph


def _insert_after_index(index_entry: Any, elements: list[Any]) -> None:
    container = index_entry
    while container is not None and container.tag != W + "sdt":
        container = container.getparent()
    if container is None:
        container = index_entry
    parent = container.getparent()
    if parent is None:
        raise RuntimeError("Index content control has no parent")
    position = parent.index(container) + 1
    for element in elements:
        parent.insert(position, element)
        position += 1


def _copy_info(info: ZipInfo) -> ZipInfo:
    copied = ZipInfo(info.filename, date_time=info.date_time)
    for name in (
        "compress_type",
        "comment",
        "extra",
        "create_system",
        "create_version",
        "extract_version",
        "flag_bits",
        "volume",
        "internal_attr",
        "external_attr",
    ):
        setattr(copied, name, getattr(info, name))
    return copied


def _append_static_table_index_entries(path: Path) -> None:
    """Add plain-caption entries that LibreOffice cannot discover as SEQ fields."""
    with ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
        members = [(_copy_info(info), archive.read(info.filename)) for info in archive.infolist()]
    root = etree.fromstring(document_xml, etree.XMLParser(remove_blank_text=False))
    templates = [
        paragraph
        for paragraph in root.iter(W + "p")
        if _style(paragraph) == "TableofFigures" and "Current Limitations" in _text(paragraph)
    ]
    if len(templates) != 1:
        raise RuntimeError(f"Expected one final table-index entry, found {len(templates)}")
    template = templates[0]
    _insert_after_index(
        template,
        [
            _index_entry(
                template,
                "ຕາຕະລາງ 4.12 ການປຽບທຽບ Sigma-only ກັບ Full AI Pipeline",
                62,
            ),
            _index_entry(
                template,
                "ຕາຕະລາງ 4.13 ຜົນ Label-blind Public Threat Scenario Evaluation",
                62,
            ),
            _index_entry(
                template,
                "ຕາຕະລາງ 4.14 ຜົນ Domain-specific and Open-set Evaluation",
                64,
            ),
        ],
    )
    revised = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    temporary = path.with_suffix(".index.tmp.docx")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output_archive:
        for info, data in members:
            if info.filename == "word/document.xml":
                data = revised
            output_archive.writestr(info, data)
    temporary.replace(path)


def _mark_fields_for_update(settings_xml: bytes) -> bytes:
    settings = etree.fromstring(settings_xml)
    update = settings.find(W + "updateFields")
    if update is None:
        update = etree.SubElement(settings, W + "updateFields")
    update.set(W + "val", "true")
    return etree.tostring(settings, xml_declaration=True, encoding="UTF-8", standalone=True)


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _refresh_document_indexes(path: Path) -> None:
    """Refresh TOC/list fields through an isolated headless LibreOffice process."""
    uno: Any = importlib.import_module("uno")
    pipe_name = f"thesis_upgrade_{os.getpid()}"
    with tempfile.TemporaryDirectory(prefix="thesis-lo-profile-") as profile:
        process = Popen(
            [
                "libreoffice",
                f"-env:UserInstallation=file://{profile}",
                "--headless",
                f"--accept=pipe,name={pipe_name};urp;StarOffice.ComponentContext",
                "--norestore",
                "--nodefault",
                "--nofirststartwizard",
            ],
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        try:
            local = uno.getComponentContext()
            resolver = local.ServiceManager.createInstanceWithContext(
                "com.sun.star.bridge.UnoUrlResolver", local
            )
            context = None
            for _ in range(100):
                try:
                    context = resolver.resolve(
                        f"uno:pipe,name={pipe_name};urp;StarOffice.ComponentContext"
                    )
                    break
                except Exception:  # noqa: BLE001 - UNO exposes provider-specific exceptions
                    time.sleep(0.1)
            if context is None:
                raise RuntimeError("Could not connect to headless LibreOffice")
            desktop = context.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", context
            )
            hidden = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
            hidden.Name = "Hidden"
            hidden.Value = True
            document = desktop.loadComponentFromURL(
                uno.systemPathToFileUrl(str(path.resolve())), "_blank", 0, (hidden,)
            )
            if document is None:
                raise RuntimeError("LibreOffice could not open the revised thesis")
            indexes = document.getDocumentIndexes()
            for index in range(indexes.getCount()):
                indexes.getByIndex(index).update()
            document.TextFields.refresh()
            document.store()
            document.close(True)
            desktop.terminate()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def revise() -> None:
    for path in (
        SOURCE,
        FLOW_METRICS,
        ENDPOINT_METRICS,
        FIXED_FPR_SUMMARY,
        INITIAL_PUBLIC_SUMMARY,
        INTERNAL_SUMMARY,
    ):
        if not path.exists():
            raise SystemExit(f"Required input is missing: {path}")

    flow = _load(FLOW_METRICS)
    endpoint = _load(ENDPOINT_METRICS)
    fixed_fpr = _load(FIXED_FPR_SUMMARY)
    heldout = fixed_fpr["runs"]["1%"]
    initial_public = _load(INITIAL_PUBLIC_SUMMARY)
    internal = _load(INTERNAL_SUMMARY)
    expected_runs = {
        "0.1%": "571d3cf06807b60ad84510b1",
        "1%": "1235a3bca4ef601a6be71f96",
        "5%": "77ea349dcdc39f1003fb8d52",
    }
    if {target: run.get("run_id") for target, run in fixed_fpr["runs"].items()} != expected_runs:
        raise SystemExit("Unexpected fixed-FPR runs; review results before thesis revision")
    if initial_public.get("run", {}).get("id") != "fc88947605086f72bae5a988":
        raise SystemExit("Unexpected initial public evaluation run")
    label_features_excluded = flow.get("protocol", {}).get("label_features_excluded")
    if not isinstance(label_features_excluded, bool) or not label_features_excluded:
        raise SystemExit("Flow result does not attest label-feature exclusion")
    if internal.get("samples") != 500:
        raise SystemExit("Unexpected internal synthetic evaluation sample count")
    endpoint_protocol = endpoint.get("protocol", {})
    if endpoint_protocol.get("training_positive_count") != 0:
        raise SystemExit("Endpoint novelty result was not trained on benign-only records")

    flow_test = flow["test"]
    endpoint_test = endpoint["test"]
    scenarios = heldout["scenarios"]
    initial_scenarios = initial_public["scenarios"]
    ddos = scenarios["cic_ddos"]
    web = scenarios["cic_web"]
    bot = scenarios["cic_bot"]
    csic = scenarios["csic_http"]
    target_01 = fixed_fpr["runs"]["0.1%"]["scenarios"]
    target_5 = fixed_fpr["runs"]["5%"]["scenarios"]
    initial_ddos = initial_scenarios["cic_ddos"]["full_pipeline"]

    english_abstract = (
        "This project presents the design, implementation, and evaluation of an AI-powered "
        "cybersecurity log-filtering and threat-prioritization prototype operating alongside a "
        f"Security Information and Event Management platform. Internal evaluation on {internal['samples']:,} "
        f"synthetic labelled syslog records produced {_pct(internal['recall'])} recall, "
        f"{_pct(internal['precision'])} precision, {_pct(internal['f1'])} F1-score, and an "
        f"{_pct(internal['false_positive_rate'], 2)} false-positive rate. An initial label-blind "
        "public-data evaluation exposed severe "
        "domain mismatch between text-template models and numeric network-flow telemetry. The "
        "prototype was therefore extended with a canonical 33-feature CICFlowMeter XGBoost model, "
        "a Bot-vs-rest head, validation-calibrated fusion exported to ONNX, structured and redacted "
        "WAF telemetry, and a benign-only BETH endpoint novelty model. On immutable held-out CIC "
        f"partitions, the general flow model achieved {_pct(flow_test['recall'], 2)} recall, "
        f"{_pct(flow_test['precision'], 2)} precision, {_pct(flow_test['f1'], 2)} F1-score, and "
        f"AUROC {flow_test['auroc']:.5f}. At the 1% validation-FPR target, observed public-test DDoS "
        f"recall/FPR was {_pct(ddos['recall'])}/{_pct(ddos['false_positive_rate'])}, Web Attack was "
        f"{_pct(web['recall'])}/{_pct(web['false_positive_rate'])}, and Bot was "
        f"{_pct(bot['recall'])}/{_pct(bot['false_positive_rate'])}; at the 5% validation target, "
        f"Bot recall/FPR was {_pct(target_5['cic_bot']['recall'])}/"
        f"{_pct(target_5['cic_bot']['false_positive_rate'])}. The BETH novelty "
        "model, trained with zero suspicious or "
        f"evil-labelled records, achieved {_pct(endpoint_test['recall'], 2)} recall, "
        f"{_pct(endpoint_test['precision'], 2)} precision, {_pct(endpoint_test['f1'], 2)} F1-score, "
        f"{_pct(endpoint_test['fpr'], 2)} false-positive rate, and AUROC "
        f"{endpoint_test['auroc']:.5f} on the predefined test split. These are bounded "
        "same-capture and open-set-proxy results, not evidence of genuine zero-day detection or "
        "live malware execution. Automated containment remains disabled by default and was "
        "validated only through dry-run and mock-policy tests."
    )
    lao_abstract = (
        "ໂຄງການນີ້ນຳສະເໜີການອອກແບບ, ພັດທະນາ ແລະປະເມີນ AI-powered log filtering "
        "ແລະ threat-prioritization prototype ທີ່ເຮັດວຽກຮ່ວມກັບ SIEM. ການທົດສອບພາຍໃນ "
        f"ດ້ວຍ synthetic syslog {internal['samples']:,} ລາຍການໄດ້ Recall "
        f"{_pct(internal['recall'])}, Precision {_pct(internal['precision'])}, F1-score "
        f"{_pct(internal['f1'])} ແລະ FPR {_pct(internal['false_positive_rate'], 2)}. "
        "ການທົດສອບ public data ຮອບທຳອິດພົບ domain mismatch "
        "ລະຫວ່າງ text-template model ແລະ numeric flow telemetry. ຈຶ່ງໄດ້ເພີ່ມ "
        "CICFlowMeter XGBoost 33 features, Bot-vs-rest head ແລະ validation-calibrated fusion "
        "ທີ່ export ເປັນ ONNX, WAF telemetry ທີ່ປິດບັງ credential ແລະ BETH endpoint "
        "novelty model ທີ່ train ດ້ວຍ benign-only records. General flow model ໃນ immutable "
        f"held-out partitions ໄດ້ Recall {_pct(flow_test['recall'], 2)}, Precision "
        f"{_pct(flow_test['precision'], 2)}, F1 {_pct(flow_test['f1'], 2)} ແລະ AUROC "
        f"{flow_test['auroc']:.5f}. ທີ່ validation-FPR target 1%, public test ໄດ້ DDoS "
        f"Recall/FPR {_pct(ddos['recall'])}/{_pct(ddos['false_positive_rate'])}, Web Attack "
        f"{_pct(web['recall'])}/{_pct(web['false_positive_rate'])}, Bot "
        f"{_pct(bot['recall'])}/{_pct(bot['false_positive_rate'])}; ທີ່ target 5%, Bot "
        f"Recall/FPR ເປັນ {_pct(target_5['cic_bot']['recall'])}/"
        f"{_pct(target_5['cic_bot']['false_positive_rate'])}. "
        f"BETH novelty test ໄດ້ Recall {_pct(endpoint_test['recall'], 2)}, Precision "
        f"{_pct(endpoint_test['precision'], 2)}, F1 {_pct(endpoint_test['f1'], 2)}, FPR "
        f"{_pct(endpoint_test['fpr'], 2)} ແລະ AUROC {endpoint_test['auroc']:.5f}. ຜົນເຫຼົ່ານີ້ "
        "ແມ່ນ same-capture ແລະ open-set proxy ບໍ່ແມ່ນຫຼັກຖານ genuine zero-day ຫຼື "
        "live malware execution. Automated containment ຍັງປິດເປັນຄ່າຕັ້ງຕົ້ນ ແລະ "
        "ທົດສອບສະເພາະ DRY_RUN/mock policy."
    )

    with ZipFile(SOURCE) as archive:
        document_xml = archive.read("word/document.xml")
        settings_xml = archive.read("word/settings.xml")
        members = [(_copy_info(info), archive.read(info.filename)) for info in archive.infolist()]

    root = etree.fromstring(document_xml, etree.XMLParser(remove_blank_text=False))
    _set_text(_find(root, "This project presents"), english_abstract)
    _set_text(_find(root, "ໂຄງການນີ້ນຳສະເໜີ"), lao_abstract)

    body_template = _find(root, "ການພິສູດຜົນໃຊ້ການລາຍງານຕາມຫຼັກຖານ")
    heading_template = _find(root, "ວິທິການພິສູດຜົນ", style="Heading3")
    caption_template = _find(root, "ຕາຕະລາງ 4.12", style="Caption")

    chapter3_anchor = _find(root, "ຂັ້ນຕອນການພັດທະນາລະບົບ", style="Heading2")
    methodology = [
        _clone(heading_template, "ການພັດທະນາ Domain-specific Flow ແລະ Endpoint Models"),
        _clone(
            body_template,
            "Flow models ໃຊ້ 33 canonical numeric CICFlowMeter features ຈາກ DDoS, Bot ແລະ "
            "Web captures. ແຕ່ລະ capture ມີ immutable train, early-stopping validation, "
            "independent calibration ແລະ test ranges. Exact float32 feature duplicates ຈາກ "
            "development partitions ຖືກຕັດອອກຈາກ later partitions ແລະ public evaluation. "
            "Label, attack name, IP, timestamp ແລະ file identity ບໍ່ເປັນ features. Fixed-FPR "
            "thresholds ຖືກເລືອກຈາກ calibration labels ເທົ່ານັ້ນ; test labels ບໍ່ໃຊ້ tune.",
        ),
        _clone(
            body_template,
            "Endpoint open-set proxy ໃຊ້ BETH predefined train/validation/test files. "
            "IsolationForest ຖືກ train ສະເພາະ records ທີ່ sus=0 ແລະ evil=0; labels, host "
            "ແລະ timestamp ບໍ່ເປັນ features. Evaluation positive target ເປັນ derived union "
            "(sus>0 OR evil>0), ໂດຍລາຍງານ evil recall ແຍກຕ່າງຫາກ ເພາະ sus ແລະ evil "
            "ມີ semantics ບໍ່ຄືກັນ. Threshold ຖືກ calibrate ໃຫ້ validation-benign FPR "
            "ເປົ້າໝາຍ 1% ໂດຍບໍ່ໃຊ້ test labels.",
        ),
        _clone(
            body_template,
            "WAF ingestion ຮອງຮັບ structured ModSecurity/generic JSON, percent-decoded URI "
            "ແລະ bounded POST body. JSON/form/text credentials, Authorization, cookies ແລະ "
            "tokens ຖືກ redaction ກ່ອນສ້າງ model-visible text. Automated response ຖືກຈຳກັດ "
            "ໃນ DRY_RUN/mock staging ດ້ວຍ allowlist, execution-time revalidation, kill switch, "
            "rate limit ແລະ duplicate suppression; ບໍ່ມີ production block/quarantine.",
        ),
    ]
    _insert_before(chapter3_anchor, methodology)

    chapter4_anchor = _find(root, "ຂໍ້ຈຳກັດ ແລະ ຄວາມເປັນໄປໄດ້ໃນອະນາຄົດ", style="Heading2")
    result_table = [
        ["Evaluation", "N", "Recall", "Precision", "F1", "FPR/AUROC"],
        [
            "CIC flow held-out block",
            str(flow_test["n"]),
            _pct(flow_test["recall"], 2),
            _pct(flow_test["precision"], 2),
            _pct(flow_test["f1"], 2),
            f"{_pct(flow_test['fp'] / flow_test['benign'], 2)}/{flow_test['auroc']:.5f}",
        ],
        [
            "CIC DDoS (1% validation target)",
            f"{ddos['n_attack']:,}/{ddos['n_benign']:,}",
            _pct(ddos["recall"]),
            _pct(ddos["precision"]),
            _pct(ddos["f1"]),
            f"{_pct(ddos['false_positive_rate'])}/{ddos['auroc']:.3f}",
        ],
        [
            "CIC Web (1% validation target)",
            f"{web['n_attack']:,}/{web['n_benign']:,}",
            _pct(web["recall"]),
            _pct(web["precision"]),
            _pct(web["f1"]),
            f"{_pct(web['false_positive_rate'])}/{web['auroc']:.3f}",
        ],
        [
            "CIC Bot (1% validation target)",
            f"{bot['n_attack']:,}/{bot['n_benign']:,}",
            _pct(bot["recall"]),
            _pct(bot["precision"]),
            _pct(bot["f1"]),
            f"{_pct(bot['false_positive_rate'])}/{bot['auroc']:.3f}",
        ],
        [
            "CSIC HTTP",
            f"{csic['n_attack']:,}/{csic['n_benign']:,}",
            _pct(csic["recall"]),
            _pct(csic["precision"]),
            _pct(csic["f1"]),
            f"{_pct(csic['false_positive_rate'])}/{csic['auroc']:.3f}",
        ],
        [
            "BETH open-set proxy",
            str(endpoint_test["n"]),
            _pct(endpoint_test["recall"], 2),
            _pct(endpoint_test["precision"], 2),
            _pct(endpoint_test["f1"], 2),
            f"{_pct(endpoint_test['fpr'], 2)}/{endpoint_test['auroc']:.5f}",
        ],
    ]
    upgrade_heading = _clone(heading_template, "ຜົນການປັບປຸງ Flow-aware ແລະ Endpoint Open-set Models")
    _page_break_before(upgrade_heading)
    chapter4 = [
        upgrade_heading,
        _clone(
            body_template,
            f"Flow development test ມີ {flow_test['n']:,} records (attack {flow_test['attack']:,}, "
            f"benign {flow_test['benign']:,}) ແລະໄດ້ TP={flow_test['tp']:,}, FP={flow_test['fp']}, "
            f"TN={flow_test['tn']:,}, FN={flow_test['fn']}; Recall={_pct(flow_test['recall'], 2)}, "
            f"Precision={_pct(flow_test['precision'], 2)}, F1={_pct(flow_test['f1'], 2)} ແລະ "
            f"AUROC={flow_test['auroc']:.5f}. ຜົນລວມນີ້ມາຈາກ immutable held-out "
            "partitions ຂອງ DDoS, Bot ແລະ Web captures; performance ແຕ່ລະ capture ແຕກຕ່າງ "
            "ແລະອາດສະທ້ອນ capture-specific regularities.",
        ),
        _clone(caption_template, "ຕາຕະລາງ 4.14 ຜົນ Domain-specific and Open-set Evaluation"),
        _table(result_table),
        _clone(
            body_template,
            f"Attested runs 0.1%={expected_runs['0.1%']}, 1%={heldout['run_id']} ແລະ "
            f"5%={expected_runs['5%']} ໃຊ້ thresholds ຈາກ independent calibration. ທີ່ "
            f"validation target 0.1%, observed DDoS/Web/Bot FPR ເປັນ "
            f"{_pct(target_01['cic_ddos']['false_positive_rate'])}/"
            f"{_pct(target_01['cic_web']['false_positive_rate'])}/"
            f"{_pct(target_01['cic_bot']['false_positive_rate'])}. ທີ່ "
            f"validation-FPR target 1%, run {heldout['run_id']} ໃຊ້ threshold ຈາກ independent "
            f"calibration ທີ່ validation-FPR target 1%. Observed test DDoS Recall/FPR="
            f"{_pct(ddos['recall'])}/{_pct(ddos['false_positive_rate'])}, Web Attack="
            f"{_pct(web['recall'])}/{_pct(web['false_positive_rate'])}, Bot="
            f"{_pct(bot['recall'])}/{_pct(bot['false_positive_rate'])} ແລະ CSIC="
            f"{_pct(csic['recall'])}/{_pct(csic['false_positive_rate'])}. ທີ່ validation target 5%, "
            f"Bot Recall/FPR ເພີ່ມເປັນ {_pct(target_5['cic_bot']['recall'])}/"
            f"{_pct(target_5['cic_bot']['false_positive_rate'])}. Validation target ບໍ່ແມ່ນ "
            "ການຮັບປະກັນ observed test FPR ເມື່ອມີ temporal/capture shift.",
        ),
        _clone(
            body_template,
            f"BETH predefined test split ມີ {endpoint_test['n']:,} records. Benign-only "
            f"IsolationForest ໄດ້ Recall={_pct(endpoint_test['recall'], 2)}, Precision="
            f"{_pct(endpoint_test['precision'], 2)}, F1={_pct(endpoint_test['f1'], 2)}, FPR="
            f"{_pct(endpoint_test['fpr'], 2)}, AUROC={endpoint_test['auroc']:.5f} ແລະ "
            f"evil-labelled recall={_pct(endpoint['test_evil_recall'], 2)}. ເນື່ອງຈາກ target "
            "ເປັນ union ຂອງ BETH sus/evil ແລະ dataset ເປັນ scenario-derived telemetry, "
            "ຜົນນີ້ຖືກອະທິບາຍເປັນ endpoint open-set proxy ເທົ່ານັ້ນ.",
        ),
    ]
    _insert_before(chapter4_anchor, chapter4)

    strengths_anchor = _find(root, "ປະເມີນຈຸດດີ", style="Heading2")
    _insert_before(
        strengths_anchor,
        [
            _clone(
                body_template,
                "ການປັບປຸງ domain-specific model ແກ້ໄຂ failure mode ທີ່ພົບໃນ public "
                f"validation ຮອບທຳອິດ: DDoS Recall ເພີ່ມຈາກ {_pct(initial_ddos['recall'])} "
                f"ເປັນ {_pct(ddos['recall'])} ໃນ held-out scenario ທີ່ກຳນົດ, ແຕ່ FPR "
                f"{_pct(ddos['false_positive_rate'])}, Web/Bot/CSIC gaps, same-capture "
                "bias ແລະ BETH label semantics ຍັງຈຳກັດ external validity. ຜົນຈຶ່ງສະໜັບສະໜູນ "
                "domain-specific scoring ແລະ safe staging, ບໍ່ໄດ້ຢືນຢັນ general zero-day "
                "detection ຫຼື production autonomous response.",
            )
        ],
    )

    biography_matches = [
        paragraph
        for paragraph in root.iter(W + "p")
        if _text(paragraph).startswith("\\ປະຫວັດຫຍໍ້ຜູ້ຂຽນ")
    ]
    if len(biography_matches) != 1:
        raise RuntimeError(
            f"Expected one primary author biography heading, found {len(biography_matches)}"
        )
    biography = biography_matches[0]
    reference_template = biography.getprevious()
    if reference_template is None:
        reference_template = body_template
    beth_reference = (
        "Highnam, K., Arulkumaran, K., Hanif, Z., & Jennings, N. R. (2021). BETH Dataset: "
        "Real Cybersecurity Data for Anomaly Detection Research. arXiv:2103.05015. "
        "https://doi.org/10.48550/arXiv.2103.05015"
    )
    if beth_reference.split(". arXiv")[0] not in " ".join(root.itertext()):
        _insert_before(biography, [_clone(reference_template, beth_reference)])

    revised_document = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    revised_settings = _mark_fields_for_update(settings_xml)
    temporary = OUTPUT.with_suffix(".tmp.docx")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output_archive:
        for info, data in members:
            if info.filename == "word/document.xml":
                data = revised_document
            elif info.filename == "word/settings.xml":
                data = revised_settings
            output_archive.writestr(info, data)
    temporary.replace(OUTPUT)
    _refresh_document_indexes(OUTPUT)
    _append_static_table_index_entries(OUTPUT)

    with ZipFile(OUTPUT) as check:
        corrupt = check.testzip()
        if corrupt:
            raise RuntimeError(f"Corrupt DOCX member: {corrupt}")
        etree.fromstring(check.read("word/document.xml"))
        etree.fromstring(check.read("word/settings.xml"))

    print(f"Source SHA-256: {_sha256(SOURCE)}")
    print(f"Output SHA-256: {_sha256(OUTPUT)}")
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    revise()
