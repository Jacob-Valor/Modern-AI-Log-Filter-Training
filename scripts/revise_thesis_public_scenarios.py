#!/usr/bin/env python3
"""Create a revised thesis DOCX containing the public-scenario evaluation.

The source document is never overwritten. The script performs surgical OOXML edits
and copies all other package members unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

etree: Any = importlib.import_module("lxml.etree")

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "final_Thesis_updated.docx"
OUTPUT = ROOT / "final_Thesis_revised_public_scenarios.docx"
SUMMARY_PATH = ROOT / "report" / "public_scenario_evaluation" / "summary.json"

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W = f"{{{NS['w']}}}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

ENGLISH_ABSTRACT = (
    "This project presents the design, implementation, and evaluation of an AI-powered "
    "cybersecurity log filtering and threat-prioritization prototype operating alongside a "
    "Security Information and Event Management (SIEM) platform. The pipeline combines Sigma "
    "rules, XGBoost classifiers, SecureBERT 2.0 Bi-Encoder retrieval, Named Entity Recognition, "
    "Cross-Encoder re-ranking, and LEEF 2.0 routing to IBM QRadar. Internal evaluation on 500 "
    "synthetic labelled syslog records produced 96.0% recall, 74.4% precision, 83.8% F1-score, "
    "and an 8.25% false-positive rate; these figures describe only that synthetic evaluation. "
    "To assess external validity, a label-blind test was additionally performed on public CSIC "
    "2010 HTTP requests and CIC-IDS2017 flow records. After deterministic sampling and exact-input "
    "deduplication, 3,881 records were scored at the frozen 0.20 threshold. The system detected "
    "15 of 271 CSIC anomalous requests (5.5% recall; 88.2% precision), one of 500 CIC DDoS "
    "flows, and none of 1,000 CIC Web Attack or 493 CIC Bot-labelled flow summaries. These "
    "results show "
    "that the software pipeline functions end to end, but its current text-template features do "
    "not generalize to numeric network-flow telemetry. CIC Bot results do not prove malware "
    "execution, and held-out/cross-dataset testing is an open-set proxy rather than genuine "
    "zero-day evidence. In this study, response is limited to enrichment, prioritization, and "
    "routing; automated containment remains a dry-run prototype."
)

LAO_ABSTRACT = (
    "ໂຄງການນີ້ນຳສະເໜີການອອກແບບ, ພັດທະນາ ແລະ ປະເມີນລະບົບຕົ້ນແບບສຳລັບການຄັດກອງ "
    "ແລະ ຈັດລຳດັບຄວາມສຳຄັນຂອງ Log ດ້ານຄວາມປອດໄພດ້ວຍ AI ເພື່ອເຮັດວຽກຮ່ວມກັບ "
    "SIEM. Pipeline ປະກອບດ້ວຍ Sigma Rules, XGBoost, SecureBERT 2.0 Bi-Encoder, NER, "
    "Cross-Encoder ແລະ ການສົ່ງຜົນແບບ LEEF 2.0 ໄປ IBM QRadar. ການປະເມີນພາຍໃນດ້ວຍ "
    "synthetic syslog 500 ລາຍການ ໄດ້ Recall 96.0%, Precision 74.4%, F1-score 83.8% ແລະ "
    "FPR 8.25%; ຄ່າເຫຼົ່ານີ້ອະທິບາຍໄດ້ສະເພາະຊຸດທົດສອບ synthetic ດັ່ງກ່າວ. ເພື່ອກວດ "
    "external validity, ໄດ້ເພີ່ມການທົດສອບແບບບໍ່ເປີດເຜີຍ label ໃຫ້ model ໂດຍໃຊ້ CSIC "
    "2010 HTTP ແລະ CIC-IDS2017. ຫຼັງຈາກສຸ່ມແບບກຳນົດ seed ແລະ ຕັດ input ທີ່ຊ້ຳກັນ, "
    "ລະບົບປະມວນຜົນ 3,881 ລາຍການດ້ວຍ threshold 0.20. ລະບົບກວດພົບ CSIC anomalous "
    "request 15/271 ລາຍການ (Recall 5.5%, Precision 88.2%), CIC DDoS 1/500 ແລະ "
    "ບໍ່ກວດພົບ CIC Web Attack 1,000 ຫຼື CIC Bot 493 flow summaries. ຜົນນີ້ພິສູດວ່າ "
    "software pipeline "
    "ເຮັດວຽກ end-to-end ໄດ້, ແຕ່ feature ແບບ text/template ປັດຈຸບັນຍັງບໍ່ generalize ໄປຫາ "
    "numeric network-flow telemetry. CIC Bot ສະແດງພຽງ botnet-associated activity, ບໍ່ໄດ້ຢືນຢັນ "
    "malware execution; ສ່ວນ cross-dataset/held-out-family ແມ່ນ open-set proxy ບໍ່ແມ່ນຫຼັກຖານ "
    "zero-day ທີ່ແທ້ຈິງ. Response ໃນການສຶກສານີ້ຈຳກັດຢູ່ການ enrich, ຈັດລຳດັບ ແລະ "
    "route event; automated containment ຍັງເປັນ DRY_RUN prototype."
)

METHODOLOGY_PARAGRAPHS = [
    (
        "ເພື່ອຕອບ comment ກ່ຽວກັບ threat scenario, ການສຶກສາໄດ້ເພີ່ມ external validation "
        "ດ້ວຍ public datasets 2 ຊຸດ: HTTP DATASET CSIC 2010 ສຳລັບ request-level web anomaly "
        "ແລະ CIC-IDS2017 ສຳລັບ DDoS-labelled, Web Attack-labelled ແລະ Bot-labelled network "
        "flow summaries. ການປະເມີນນີ້ແຍກອອກຈາກ HDFS/synthetic evaluation ເພາະ dataset "
        "ແຕ່ລະຊຸດມີ domain ແລະ schema ຕ່າງກັນ. ແຜນ 72-hour Wazuh-vs-QRadar replay "
        "ທີ່ອະທິບາຍກ່ອນໜ້າບໍ່ໄດ້ຖືກດຳເນີນການຄົບ; ຜົນຕົວຈິງໃນບົດ 4 ແມ່ນ "
        "offline label-blind evaluation ແລະ Sigma-only proxy."
    ),
    (
        "Ground truth ຖືກເກັບໄວ້ໃນ sidecar ເທົ່ານັ້ນ. Model input ບໍ່ມີ label, attack "
        "category, MITRE ID ຫຼື severity ທີ່ສ້າງຈາກ label. CICFlowMeter values ຖືກແປເປັນ "
        "neutral CEF flow observation ໂດຍໃຊ້ພຽງຄ່າທີ່ສັງເກດໄດ້ ເຊັ່ນ destination port, flow "
        "duration, packet counts, byte/packet rates, flag counts, active mean ແລະ idle mean. CSIC "
        "request-line ຖືກແປເປັນ Apache combined access-log ດ້ວຍ constant source IP/status/size "
        "ທີ່ນຳໃຊ້ຄືກັນທັງ benign ແລະ anomalous; POST body ບໍ່ຖືກນຳໃສ່ access-log input."
    ),
    (
        "ການສຸ່ມໃຊ້ seed 20260729, ຈຳກັດສູງສຸດ 500 ລາຍການຕໍ່ exact class, ຕັດ exact "
        "duplicate model inputs, ແລະໃຊ້ deployed LOW threshold = 0.20 ໂດຍບໍ່ tune ກັບ public "
        "test labels. Full pipeline ປະມວນຜົນແບບ batch size 128 ເທິງ NVIDIA RTX 4050. Metrics "
        "ປະກອບມີ TP, FP, TN, FN, Recall, Precision, F1, FPR, Balanced Accuracy, AUROC, AUPRC "
        "ແລະ Wilson 95% CI. Prediction ທຸກລາຍການຖືກບັນທຶກໃນ "
        "report/public_scenario_evaluation/predictions.jsonl ເພື່ອໃຫ້ກວດສອບຊ້ຳໄດ້."
    ),
    (
        "Threat claims ຖືກຈຳກັດຕາມຫຼັກຖານ: CIC DDoS ແມ່ນການຈຳແນກ flow summary "
        "ບໍ່ແມ່ນ live mitigation; CIC Bot ແມ່ນ botnet-associated network activity ບໍ່ແມ່ນ "
        "ການຢືນຢັນ malware execution; ແລະ family ທີ່ບໍ່ໄດ້ນຳມາ train/tune ຖືກເອີ້ນວ່າ "
        "unseen-family/open-set proxy ບໍ່ແມ່ນ genuine zero-day."
    ),
]

RESULT_PARAGRAPHS = [
    (
        "ການທົດສອບ external validation ແລ່ນສຳເລັດ 3,881/3,881 ລາຍການ ໂດຍບໍ່ມີ "
        "scoring failure. Dataset checksums, adapter policy, audit counts ແລະ per-event predictions "
        "ຖືກເກັບໃນ report/public_scenario_evaluation/. ການກວດ label leakage ຜ່ານທັງໝົດ."
    ),
    (
        "CSIC HTTP: ຫຼັງຕັດ duplicate ແລະ model inputs ທີ່ມີ ground truth ຂັດແຍ້ງກັນ "
        "274 ລາຍການ, ເຫຼືອ anomalous 271 ແລະ benign 118 requests. ລະບົບໄດ້ TP=15, "
        "FP=2, TN=116, FN=256; Recall=5.5% (Wilson 95% CI 3.4–8.9%), Precision=88.2%, "
        "F1=10.4%, FPR=1.7%, AUROC=0.517. ຜົນນີ້ສະແດງວ່າ "
        "parser ຮັບ Apache-style access log ໄດ້ ແຕ່ detection coverage ຕໍ່ unseen HTTP payloads "
        "ຍັງຕ່ຳ, ໂດຍສະເພາະ POST payload ທີ່ access log ບໍ່ບັນທຶກ body."
    ),
    (
        "CIC-IDS2017: DDoS test ມີ 500 attack/500 benign ໄດ້ TP=1, FP=0, Recall=0.2%, "
        "FPR=0%, AUROC=0.503. Web Attack test ມີ 1,000 attack/500 benign ໄດ້ TP=0, FP=0 "
        "ແລະ AUROC=0.500. Bot test ມີ 493 attack/499 benign ໄດ້ TP=0, FP=0 ແລະ "
        "AUROC=0.506. ສາເຫດທາງສະຖາປັດຕະຍະກຳຄື model-visible text ຂອງ neutral CEF "
        "flows ຄ້າຍຄືກັນຫຼາຍ; numeric extension fields ບໍ່ໄດ້ຖືກປ່ຽນເປັນ feature ຂອງ "
        "XGBoost/SecureBERT, ແລະ deduplication ຈຶ່ງຫຼຸດຄະແນນຂອງເກືອບທຸກ flow."
    ),
    (
        "ຜົນ external validation ບໍ່ສະໜັບສະໜູນການ claim ວ່າລະບົບປັດຈຸບັນກວດ DDoS, "
        "malware ຫຼື zero-day ໄດ້ທົ່ວໄປ. ຜົນທີ່ຖືກຕ້ອງຄື pipeline ສາມາດ ingest, "
        "normalize, score ແລະ route public-data-derived events ໄດ້, ແຕ່ model ຕ້ອງເພີ່ມ "
        "flow-aware numeric features, time-window aggregation, request-body/WAF telemetry ແລະ "
        "cross-dataset retraining ກ່ອນນຳໄປ claim threat-family detection."
    ),
]

RESULT_TABLE = [
    ["Scenario", "Attack/Benign", "TP/FP/FN", "Recall", "Precision/F1", "FPR/AUROC"],
    ["CSIC HTTP", "271/118", "15/2/256", "5.5%", "88.2%/10.4%", "1.7%/0.517"],
    ["CIC DDoS", "500/500", "1/0/499", "0.2%", "100%/0.4%", "0.0%/0.503"],
    ["CIC Web", "1,000/500", "0/0/1,000", "0.0%", "0.0%/0.0%", "0.0%/0.500"],
    ["CIC Bot", "493/499", "0/0/493", "0.0%", "0.0%/0.0%", "0.0%/0.506"],
]

CONCLUSION_EXTERNAL = (
    "ການເພີ່ມ external validation ດ້ວຍ public attack datasets ສະແດງຜົນທີ່ຕ່າງຈາກ "
    "synthetic evaluation ຢ່າງຊັດເຈນ. ລະບົບກວດ CSIC anomalous HTTP requests ໄດ້ "
    "15/271 (Recall 5.5%), CIC DDoS 1/500 ແລະບໍ່ກວດພົບ Web Attack ຫຼື Bot flows ໃນ sample "
    "ທີ່ປະເມີນ. ດັ່ງນັ້ນ ຜົນ 96.0% ຕ້ອງຈຳກັດໃຫ້ອ້າງອີງສະເພາະ synthetic syslog "
    "500 ລາຍການ; ບໍ່ສາມາດ generalize ເປັນ DDoS, malware ຫຼື zero-day detection. "
    "ຜົນລົບນີ້ເປັນຫຼັກຖານສຳຄັນວ່າ production threat detection ຕ້ອງມີ domain-specific "
    "feature adapter, cross-dataset validation ແລະ response-safety testing ເພີ່ມເຕີມ."
)

REFERENCES = [
    (
        "Torrano-Gimenez, C., Perez-Villegas, A., & Alvarez, G. (2010). An anomaly-based approach "
        "for intrusion detection in web traffic. In Computational Intelligence in Security for "
        "Information Systems 2010, 89–96. https://doi.org/10.1007/978-3-642-15869-4_36"
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paragraph_text(paragraph: Any) -> str:
    return "".join(node.text or "" for node in paragraph.iter(W + "t")).strip()


def style_name(paragraph: Any) -> str:
    style = paragraph.find("./w:pPr/w:pStyle", NS)
    return style.get(W + "val", "") if style is not None else ""


def find_paragraph(
    root: Any,
    text_start: str,
    *,
    style: str | None = None,
) -> Any:
    matches = [
        paragraph
        for paragraph in root.iter(W + "p")
        if paragraph_text(paragraph).startswith(text_start)
        and (style is None or style_name(paragraph) == style)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one paragraph {text_start!r} style={style!r}, found {len(matches)}"
        )
    return matches[0]


def set_paragraph_text(paragraph: Any, text: str, *, bold: bool = False) -> None:
    first_run = paragraph.find(".//w:r", NS)
    run_properties = first_run.find("./w:rPr", NS) if first_run is not None else None
    paragraph_properties = paragraph.find("./w:pPr", NS)
    for child in list(paragraph):
        if child is not paragraph_properties:
            paragraph.remove(child)
    run = etree.SubElement(paragraph, W + "r")
    if run_properties is not None:
        run.append(copy.deepcopy(run_properties))
    elif bold:
        properties = etree.SubElement(run, W + "rPr")
        etree.SubElement(properties, W + "b")
        etree.SubElement(properties, W + "bCs")
    if bold:
        properties = run.find("./w:rPr", NS)
        if properties is None:
            properties = etree.SubElement(run, W + "rPr")
        if properties.find("./w:b", NS) is None:
            etree.SubElement(properties, W + "b")
        if properties.find("./w:bCs", NS) is None:
            etree.SubElement(properties, W + "bCs")
    text_node = etree.SubElement(run, W + "t")
    text_node.set(XML_SPACE, "preserve")
    text_node.text = text


def cloned_paragraph(template: Any, text: str, *, bold: bool = False) -> Any:
    paragraph = copy.deepcopy(template)
    set_paragraph_text(paragraph, text, bold=bold)
    return paragraph


def insert_before(anchor: Any, elements: list[Any]) -> None:
    parent = anchor.getparent()
    if parent is None:
        raise RuntimeError("Anchor paragraph has no parent")
    index = parent.index(anchor)
    for element in elements:
        parent.insert(index, element)
        index += 1


def cell_paragraph(text: str, *, header: bool = False) -> Any:
    paragraph = etree.Element(W + "p")
    paragraph_properties = etree.SubElement(paragraph, W + "pPr")
    justification = etree.SubElement(paragraph_properties, W + "jc")
    justification.set(W + "val", "center")
    run = etree.SubElement(paragraph, W + "r")
    run_properties = etree.SubElement(run, W + "rPr")
    fonts = etree.SubElement(run_properties, W + "rFonts")
    for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(W + attribute, "Phetsarath OT")
    size = etree.SubElement(run_properties, W + "sz")
    size.set(W + "val", "18")
    complex_size = etree.SubElement(run_properties, W + "szCs")
    complex_size.set(W + "val", "18")
    language = etree.SubElement(run_properties, W + "lang")
    language.set(W + "val", "lo-LA")
    if header:
        etree.SubElement(run_properties, W + "b")
        etree.SubElement(run_properties, W + "bCs")
    text_node = etree.SubElement(run, W + "t")
    text_node.text = text
    return paragraph


def make_table(rows: list[list[str]]) -> Any:
    table = etree.Element(W + "tbl")
    properties = etree.SubElement(table, W + "tblPr")
    width = etree.SubElement(properties, W + "tblW")
    width.set(W + "w", "0")
    width.set(W + "type", "auto")
    borders = etree.SubElement(properties, W + "tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = etree.SubElement(borders, W + side)
        border.set(W + "val", "single")
        border.set(W + "sz", "4")
        border.set(W + "space", "0")
        border.set(W + "color", "808080")
    for row_index, values in enumerate(rows):
        row = etree.SubElement(table, W + "tr")
        row_properties = etree.SubElement(row, W + "trPr")
        etree.SubElement(row_properties, W + "cantSplit")
        for value in values:
            cell = etree.SubElement(row, W + "tc")
            cell_properties = etree.SubElement(cell, W + "tcPr")
            cell_width = etree.SubElement(cell_properties, W + "tcW")
            cell_width.set(W + "w", "1500")
            cell_width.set(W + "type", "dxa")
            cell.append(cell_paragraph(value, header=row_index == 0))
    return table


def set_table_cell_text(cell: Any, text: str) -> None:
    paragraphs = cell.findall("./w:p", NS)
    if not paragraphs:
        paragraph = etree.SubElement(cell, W + "p")
    else:
        paragraph = paragraphs[0]
    set_paragraph_text(paragraph, text)
    for extra in paragraphs[1:]:
        cell.remove(extra)


def correct_internal_results_table(root: Any) -> None:
    tables = list(root.iter(W + "tbl"))
    target = None
    for table in tables:
        visible = " | ".join(paragraph_text(cell) for cell in table.iter(W + "tc"))
        if "[94.8%, 97.2%]" in visible and "74.4%" in visible:
            target = table
            break
    if target is None:
        raise RuntimeError("Could not find internal model-performance table")
    rows = target.findall("./w:tr", NS)
    corrections = {
        1: ("[90.2%, 98.4%]", "Exceeded (synthetic set)"),
        2: ("[5.9%, 11.4%]", "Met (point estimate)"),
        3: ("[66.3%, 81.2%]", "Below Target"),
        4: ("[78.3%, 88.7%]*", "Below Target"),
        5: ("Not calculated", "Point estimate only"),
    }
    for row_index, (interval, status) in corrections.items():
        cells = rows[row_index].findall("./w:tc", NS)
        set_table_cell_text(cells[3], interval)
        set_table_cell_text(cells[4], status)

    caption_matches = [
        paragraph
        for paragraph in root.iter(W + "p")
        if style_name(paragraph) == "Caption" and "n=226,000" in paragraph_text(paragraph)
    ]
    if len(caption_matches) != 1:
        raise RuntimeError(
            f"Expected one n=226,000 caption replacement, found {len(caption_matches)}"
        )
    caption = caption_matches[0]
    replacements = 0
    for node in caption.iter(W + "t"):
        if node.text and "n=226,000" in node.text:
            node.text = node.text.replace("n=226,000", "n=500 synthetic syslog samples")
            replacements += 1
    if replacements != 1:
        raise RuntimeError(f"Expected one caption text-node replacement, found {replacements}")

    cached_matches = [
        paragraph
        for paragraph in root.iter(W + "p")
        if style_name(paragraph) != "Caption" and "n=226,000" in paragraph_text(paragraph)
    ]
    if len(cached_matches) != 1:
        raise RuntimeError(f"Expected one cached list-of-tables entry, found {len(cached_matches)}")
    for node in cached_matches[0].iter(W + "t"):
        if node.text and "n=226,000" in node.text:
            node.text = node.text.replace("n=226,000", "n=500 synthetic syslog samples")


def mark_fields_for_update(settings_xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(settings_xml, parser)
    update = root.find("./w:updateFields", NS)
    if update is None:
        update = etree.SubElement(root, W + "updateFields")
    update.set(W + "val", "true")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def copy_zip_info(info: ZipInfo) -> ZipInfo:
    clone = ZipInfo(info.filename, date_time=info.date_time)
    clone.compress_type = info.compress_type
    clone.comment = info.comment
    clone.extra = info.extra
    clone.create_system = info.create_system
    clone.create_version = info.create_version
    clone.extract_version = info.extract_version
    clone.flag_bits = info.flag_bits
    clone.volume = info.volume
    clone.internal_attr = info.internal_attr
    clone.external_attr = info.external_attr
    return clone


def revise() -> None:
    if not SOURCE.exists() or not SUMMARY_PATH.exists():
        raise SystemExit("Source thesis or public-scenario summary is missing")
    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read public-scenario summary: {exc}") from exc
    if summary["run"]["id"] != "fc88947605086f72bae5a988":
        raise SystemExit(
            "Unexpected evaluation run; review the new results before revising the thesis"
        )

    with ZipFile(SOURCE) as archive:
        document_xml = archive.read("word/document.xml")
        settings_xml = archive.read("word/settings.xml")
        members = [
            (copy_zip_info(info), archive.read(info.filename)) for info in archive.infolist()
        ]

    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(document_xml, parser)

    set_paragraph_text(find_paragraph(root, "This Project presents"), ENGLISH_ABSTRACT)
    set_paragraph_text(find_paragraph(root, "ໂຄງການນີ້ນຳສະເໜີ"), LAO_ABSTRACT)

    set_paragraph_text(
        find_paragraph(root, "ຂໍ້ມູນ Logs ຫຼັກໃຊ້ HDFS"),
        (
            "ຂໍ້ມູນຫຼັກໃຊ້ HDFS TraceBench ສຳລັບ training/internal validation ແລະ "
            "synthetic syslog 500 ລາຍການສຳລັບ full-pipeline evaluation. ການທົດສອບ external "
            "validation ໃຊ້ public HTTP DATASET CSIC 2010 ແລະ CIC-IDS2017 ສຳລັບ DDoS, "
            "Web Attack ແລະ Bot-labelled flow summaries. Public datasets ບໍ່ໄດ້ໃຊ້ retrain "
            "model ຫຼື tune threshold ໃນຜົນທີ່ລາຍງານ."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "ລວບລວມ Logs ຈາກ Firewall"),
        (
            "ການພັດທະນາ model ຕົວຈິງໃຊ້ HDFS TraceBench ແລະ synthetic syslog "
            "ສຳລັບ training/internal validation. CICIDS2017 ແລະ CSIC 2010 ຖືກນຳໃຊ້ "
            "ພາຍຫຼັງເປັນ external test ເທົ່ານັ້ນ; NSL-KDD ແລະ partner-organization logs "
            "ບໍ່ໄດ້ຖືກນຳໃຊ້ໃນຜົນທີ່ລາຍງານ."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "ເພີ່ມປະສິດທິພາບການກວດຈັບໄພຄຸກຄາມ"),
        (
            "ປະເມີນຄວາມສາມາດຂອງ AI/ML ໃນການຈັດລຳດັບ threat-related logs, "
            "ລວມທັງການທົດສອບ unseen-family/open-set proxy; ບໍ່ຖືວ່າເປັນຫຼັກຖານ "
            "genuine zero-day ຫຼື APT detection ຈົນກວ່າຈະມີ temporal novelty ແລະ "
            "multi-source incident ground truth."
        ),
    )

    set_paragraph_text(
        find_paragraph(root, "ບໍ່ສາມາດກວດຈັບໄພຄຸກຄາມໃໝ່"),
        (
            "Static signature systems ອາດພາດໄພຄຸກຄາມທີ່ pattern ບໍ່ຢູ່ໃນຖານ signature; "
            "anomaly/open-set methods ອາດຊ່ວຍຈັດລຳດັບ event ແຕ່ບໍ່ຮັບປະກັນ "
            "zero-day ຫຼື APT detection."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "Zero-day Attack:"),
        (
            "Zero-day Attack ແມ່ນການໃຊ້ vulnerability ທີ່ຍັງບໍ່ຮູ້ຈັກ ຫຼື ຍັງບໍ່ມີ patch. "
            "Anomaly detection ອາດຊ່ວຍຊີ້ພຶດຕິກຳຜິດປົກກະຕິ, ແຕ່ການຢືນຢັນ zero-day "
            "ຕ້ອງມີ temporal novelty ແລະ exploit/incident ground truth."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "Signature-based Detection:"),
        (
            "Signature-based Detection: ປຽບທຽບ events ກັບ known patterns; "
            "ອາດພາດ unseen behavior ແຕ່ບໍ່ຄວນສະຫຼຸບວ່າກວດ zero-day ບໍ່ໄດ້ທຸກກໍລະນີ."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "Anomaly-based Detection:"),
        (
            "Anomaly-based Detection: ຮຽນຮູ້ baseline ແລະ alert ເມື່ອພົບ deviation; "
            "ໃຊ້ເປັນ open-set signal ໄດ້ ແຕ່ບໍ່ເທົ່າກັບການຢືນຢັນ zero-day."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "ຂ Unsupervised Learning ກວດຈັບ Zero-day"),
        "ຂ Unsupervised Learning ສຳລັບ Novelty/Open-set Scoring",
    )
    set_paragraph_text(
        find_paragraph(root, "ລະບົບ Rule-Based ສາມາດກວດຈັບໄດ້ແຕ່"),
        (
            "Rule-based systems ມີຈຸດແຂງຕໍ່ known patterns ແຕ່ອາດມີ coverage ຈຳກັດຕໍ່ "
            "unseen behavior; AI anomaly score ເປັນພຽງ open-set signal ບໍ່ແມ່ນການຢືນຢັນ "
            "zero-day ຫຼື APT."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "Zero-day/rare event detection"),
        "Novelty/open-set scoring (not proof of genuine zero-day detection)",
    )
    set_paragraph_text(
        find_paragraph(root, "ຈຸດປະສົງ: ກວດຈັບ Rare/Unusual Events"),
        (
            "ຈຸດປະສົງ: ຈັດລຳດັບ Rare/Unusual Events ເປັນ open-set candidates; "
            "ຜົນ anomaly ຢ່າງດຽວບໍ່ຢືນຢັນ genuine zero-day."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "Zero-day detection"),
        "Unseen-family/open-set proxy only",
    )

    set_paragraph_text(
        find_paragraph(root, "ຈຸດປະສົງຫຼັກຂອງການຄົ້ນຄວ້ານີ້"),
        "ຈຸດປະສົງຫຼັກຂອງການຄົ້ນຄວ້ານີ້ ມີ 4 ຂໍ້ ດັ່ງຕໍ່ໄປນີ້:",
    )

    objective_template = find_paragraph(root, "ເພື່ອປຽບທຽບ ແລະ ປະເມີນ", style="orderlist1")
    scope_heading = find_paragraph(root, "ຂອບເຂດຂອງການສຶກສາ", style="Heading2")
    insert_before(
        scope_heading,
        [
            cloned_paragraph(
                objective_template,
                (
                    "ເພື່ອປະເມີນ external validity ດ້ວຍ public labelled data ໂດຍແຍກ "
                    "ground truth ອອກຈາກ model input ແລະລາຍງານ DDoS flow, web request, "
                    "botnet-associated ແລະ unseen-family results ຕາມຂອບເຂດຫຼັກຖານ."
                ),
            )
        ],
    )

    set_paragraph_text(
        find_paragraph(root, "ປະເມີນ AI Model:"),
        (
            "ປະເມີນ AI Model: ໃຊ້ Confusion Matrix ຄຳນວນ Recall, FPR, Precision "
            "ແລະ F1-score ຈາກ labelled ground truth ໂດຍແຍກ ground truth ອອກຈາກ "
            "model-visible input."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "ວິເຄາະ ແລະ ສະຫຼຸບ: ນຳ Metrics"),
        (
            "ວິເຄາະ ແລະ ສະຫຼຸບ: ບັນທຶກ confusion counts, point estimates, confidence "
            "intervals, per-family results, runtime ແລະ limitations. ແຜນ Mann-Whitney U test "
            "ບໍ່ໄດ້ດຳເນີນການ ເນື່ອງຈາກ full 72-hour paired deployment ບໍ່ໄດ້ເກີດຂຶ້ນ."
        ),
    )
    body_template = find_paragraph(root, "ລະບົບຖືກຖືວ່າ ພິສູດສຳເລັດ")
    set_paragraph_text(
        body_template,
        (
            "ການພິສູດຜົນໃຊ້ການລາຍງານຕາມຫຼັກຖານ ໂດຍບໍ່ຖືວ່າ target crossing "
            "ເປັນຫຼັກຖານພຽງຢ່າງດຽວ. Metric ທີ່ບໍ່ບັນລຸ target ຍັງເປັນ valid result "
            "ແລະຕ້ອງຖືກລາຍງານພ້ອມສາເຫດ, uncertainty ແລະ limitation."
        ),
    )
    heading3_template = find_paragraph(root, "ວິທິການພິສູດຜົນ", style="Heading3")
    chapter3_anchor = find_paragraph(root, "ຂັ້ນຕອນການພັດທະນາລະບົບ", style="Heading2")
    chapter3_elements = [cloned_paragraph(heading3_template, "ການປະເມີນກັບ Public Threat Scenarios")]
    chapter3_elements.extend(
        cloned_paragraph(body_template, text) for text in METHODOLOGY_PARAGRAPHS
    )
    insert_before(chapter3_anchor, chapter3_elements)

    correct_internal_results_table(root)
    set_paragraph_text(
        find_paragraph(root, "ໝາຍເຫດ: 95% CI ຂ້າງເທີງ"),
        (
            "ໝາຍເຫດ: Recall, FPR ແລະ Precision 95% CI ຖືກຄຳນວນຈາກ confusion counts "
            "n=500 ດ້ວຍ Wilson interval. F1 interval ເປັນ bootstrap approximation; AUC 0.94 "
            "ເປັນ point estimate ແລະບໍ່ໄດ້ລາຍງານ CI."
        ),
    )
    precision_note = find_paragraph(root, "ໝາຍເຫດ: Precision 74.4%")
    set_paragraph_text(
        precision_note,
        (
            "ໝາຍເຫດ: Precision 74.4% ແລະ F1-score 83.8% ຕ່ຳກວ່າເກນ 85% ແລະ 88%. "
            "ສາເຫດທີ່ມີຫຼັກຖານແມ່ນ model/dataset domain mismatch, synthetic template bias "
            "ແລະ threshold 0.20. CPU-only inference ມີຜົນຫຼັກຕໍ່ latency/throughput ບໍ່ແມ່ນ "
            "ສາເຫດໂດຍກົງຂອງ precision ຕ່ຳ ນອກຈາກຈະມີ timeout ຫຼື degraded fallback. "
            "ຕາຕະລາງ 95% CI ໃຊ້ Wilson intervals ສຳລັບ Recall/FPR/Precision; F1 interval "
            "ເປັນ bootstrap approximation ແລະ AUC interval ບໍ່ໄດ້ຄຳນວນ."
        ),
    )

    chapter4_anchor = find_paragraph(root, "ຂໍ້ຈຳກັດ ແລະ ຄວາມເປັນໄປໄດ້ໃນອະນາຄົດ", style="Heading2")
    chapter4_heading_template = find_paragraph(
        root, "ການປຽບທຽບກັບ Rule-based Baseline", style="Heading3"
    )
    caption_template = find_paragraph(root, "ຕາຕະລາງ 4.12", style="Caption")
    chapter4_elements = [
        cloned_paragraph(
            chapter4_heading_template,
            "ການທົດສອບ Public Threat Scenarios (External Validation)",
        ),
        cloned_paragraph(body_template, RESULT_PARAGRAPHS[0]),
        cloned_paragraph(
            caption_template,
            "ຕາຕະລາງ 4.13 ຜົນ Label-blind Public Threat Scenario Evaluation",
        ),
        make_table(RESULT_TABLE),
    ]
    chapter4_elements.extend(
        cloned_paragraph(body_template, text) for text in RESULT_PARAGRAPHS[1:]
    )
    insert_before(chapter4_anchor, chapter4_elements)

    set_paragraph_text(
        find_paragraph(root, "ການສຶກສາໃນຄັ້ງນີ້ໄດ້ພັດທະນາ"),
        (
            "ການສຶກສານີ້ໄດ້ພັດທະນາ AI-powered log filtering and threat-prioritization "
            "prototype ທີ່ເຮັດວຽກຄຽງຄູ່ SIEM. ລະບົບເນັ້ນການ ingest, normalize, score, "
            "enrich ແລະ route security events. ຄຳວ່າ response ໃນຜົນທີ່ປະເມີນໝາຍເຖິງ "
            "prioritization/routing ໄປ SIEM; automated block/quarantine ຍັງບໍ່ໄດ້ປະເມີນ."
        ),
    )
    set_paragraph_text(
        find_paragraph(root, "ຜົນໄດ້ຮັບດັ່ງກ່າວຢືນຢັນວ່າ"),
        (
            "ໃນ synthetic syslog 500 ລາຍການ, Full Pipeline ມີ Recall ແລະ F1 ສູງກວ່າ "
            "Sigma-only proxy (96.0% ທຽບ 59.0% Recall; 83.8% ທຽບ 71.1% F1), ແຕ່ມີ "
            "Precision ຕ່ຳກວ່າ ແລະ FPR ສູງກວ່າ. ຜົນນີ້ຈຳກັດຢູ່ dataset ດຽວ ແລະ "
            "baseline ເປັນ proxy, ຈຶ່ງບໍ່ຄວນຕີຄວາມເປັນການຢືນຢັນວ່າດີກວ່າ deployed "
            "traditional SIEM ໃນ production."
        ),
    )
    strengths_anchor = find_paragraph(root, "ປະເມີນຈຸດດີ", style="Heading2")
    insert_before(strengths_anchor, [cloned_paragraph(body_template, CONCLUSION_EXTERNAL)])

    set_paragraph_text(
        find_paragraph(root, "ລະບົບໄດ້ສະແດງ Detection Rate"),
        (
            "ລະບົບໄດ້ Recall 96.0% ໃນ synthetic labelled syslog 500 ລາຍການ, ສະແດງວ່າ "
            "pipeline ສາມາດຈຳແນກ template ທີ່ຢູ່ໃນ domain ຂອງຊຸດທົດສອບໄດ້. ແຕ່ "
            "external public-dataset results ຕ່ຳກວ່າຫຼາຍ, ດັ່ງນັ້ນ 96.0% ບໍ່ແມ່ນຄ່າ "
            "general threat-detection performance."
        ),
    )

    set_paragraph_text(
        find_paragraph(root, "Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2023)"),
        (
            "Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a "
            "new intrusion detection dataset and intrusion traffic characterization. Proceedings "
            "of the 4th International Conference on Information Systems Security and Privacy, "
            "108–116. https://doi.org/10.5220/0006639801080116"
        ),
    )

    biography_matches = [
        paragraph
        for paragraph in root.iter(W + "p")
        if paragraph_text(paragraph).startswith("\\ປະຫວັດຫຍໍ້ຜູ້ຂຽນ")
    ]
    if len(biography_matches) != 2:
        raise RuntimeError(f"Expected two author biographies, found {len(biography_matches)}")
    biography = biography_matches[0]
    reference_template = biography.getprevious()
    if reference_template is None or reference_template.tag != W + "p":
        reference_template = body_template
    insert_before(
        biography,
        [cloned_paragraph(reference_template, reference) for reference in REFERENCES],
    )

    revised_document = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    revised_settings = mark_fields_for_update(settings_xml)

    temporary = OUTPUT.with_suffix(".tmp.docx")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output_archive:
        for info, data in members:
            if info.filename == "word/document.xml":
                data = revised_document
            elif info.filename == "word/settings.xml":
                data = revised_settings
            output_archive.writestr(info, data)
    temporary.replace(OUTPUT)

    with ZipFile(OUTPUT) as check:
        corrupt = check.testzip()
        if corrupt:
            raise RuntimeError(f"Corrupt DOCX member: {corrupt}")
        etree.fromstring(check.read("word/document.xml"))
        etree.fromstring(check.read("word/settings.xml"))

    print(f"Source SHA-256: {sha256(SOURCE)}")
    print(f"Output SHA-256: {sha256(OUTPUT)}")
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    revise()
