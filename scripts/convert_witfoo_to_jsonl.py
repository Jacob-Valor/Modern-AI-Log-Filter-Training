#!/usr/bin/env python3
"""
Convert WitFoo Precinct6 signals.parquet to pipeline-ready JSONL.

Outputs JSONL files organized by source type, suitable for:
  - POST /score/batch API endpoint
  - Direct pipeline scoring via scorer.score_batch()

Usage:
  python scripts/convert_witfoo_to_jsonl.py \\
      --input demo_data/witfoo/signals/signals.parquet \\
      --output-dir demo_data/witfoo/jsonl \\
      --max-events 10000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# ── Source-type mapping based on message_type patterns ─────────────────────

# Windows Security Events (by Event ID number)
WINEVENT_IDS = {
    "4624", "4625", "4634", "4647", "4648", "4656", "4658", "4662", "4663",
    "4672", "4673", "4674", "4688", "4690", "4697", "4698", "4700", "4702",
    "4703", "4719", "4720", "4722", "4723", "4724", "4725", "4726", "4727",
    "4728", "4729", "4730", "4731", "4732", "4733", "4734", "4735", "4737",
    "4738", "4739", "4740", "4741", "4742", "4743", "4754", "4755", "4756",
    "4767", "4768", "4769", "4770", "4771", "4776", "4778", "4779", "4780",
    "4781", "4782", "4783", "4784", "4785", "4786", "4787", "4793", "4794",
    "4797", "4798", "4799", "4800", "4801", "4802", "4803", "4816", "4817",
    "4818", "4819", "4820", "4821", "4822", "4823", "4824", "4825", "4826",
    "4830", "4864", "4865", "4866", "4867", "4868", "4870", "4872", "4873",
    "4874", "4875", "4876", "4880", "4881", "4882", "4883", "4890", "4891",
    "4892", "4893", "4894", "4895", "4896", "4897", "4898", "4900", "4902",
    "4904", "4905", "4906", "4907", "4908", "4909", "4910", "4911", "4912",
    "4928", "4929", "4930", "4931", "4932", "4933", "4944", "4945", "4946",
    "4947", "4948", "4949", "4950", "4951", "4952", "4953", "4954", "4955",
    "4956", "4957", "4958", "4960", "4961", "4962", "4963", "4964", "4965",
    "4976", "4977", "4978", "4979", "4980", "4981", "4982", "4983", "4984",
    "4985", "5024", "5025", "5027", "5028", "5029", "5030", "5031", "5032",
    "5033", "5034", "5035", "5037", "5038", "5039", "5040", "5041", "5042",
    "5043", "5044", "5045", "5046", "5047", "5048", "5049", "5050", "5051",
    "5052", "5053", "5054", "5055", "5056", "5057", "5058", "5059", "5060",
    "5061", "5062", "5063", "5064", "5065", "5066", "5067", "5068", "5069",
    "5070", "5071", "5072", "5073", "5074", "5075", "5076", "5077", "5078",
    "5079", "5080", "5081", "5082", "5083", "5084", "5085", "5086", "5087",
    "5088", "5089", "5090", "5091", "5092", "5093", "5094", "5095", "5096",
    "5097", "5098", "5099", "5100", "5101", "5102", "5120", "5121", "5122",
    "5123", "5124", "5125", "5126", "5127", "5136", "5137", "5138", "5139",
    "5140", "5141", "5142", "5143", "5144", "5145", "5146", "5147", "5148",
    "5149", "5150", "5151", "5152", "5153", "5154", "5155", "5156", "5157",
    "5158", "5159", "5168", "5169", "5170", "5171", "5172", "5173", "5174",
    "5175", "5176", "5177", "5178", "5179", "5180", "5181", "5182", "5183",
    "5184", "5185", "5186", "5187", "5188", "5189", "5190", "5191", "5192",
    "5193", "5194", "5195", "5196", "5197", "5198", "5199", "5200", "5376",
    "5377", "5378", "5379", "5380", "5381", "5382", "5440", "5441", "5442",
    "5443", "5444", "5446", "5447", "5448", "5449", "5450", "5451", "5452",
    "5453", "5456", "5457", "5458", "5459", "5460", "5461", "5462", "5463",
    "5464", "5465", "5466", "5467", "5468", "5469", "5470", "5471", "5472",
    "5473", "5474", "5475", "5477", "5478", "5479", "5480", "5481", "5482",
    "5483", "5484", "5485", "5632", "5633", "5634", "5635", "5636", "5637",
    "6144", "6145", "6272", "6273", "6274", "6275", "6276", "6277", "6278",
    "6279", "6280", "6281", "6401", "6402", "6403", "6404", "6405", "6406",
    "6407", "6408", "6409", "6410", "6416", "6417", "6418", "6419", "6420",
    "6421", "6422", "6423", "6424", "6425",
}

# Well-known Windows message types
WINEVENT_SPECIAL = {
    "security_audit_event", "account_logon", "account_logoff",
    "account_management", "detailed_tracking", "logon_logoff",
    "object_access", "policy_change", "privilege_use",
    "process_creation", "process_access", "registry_event",
    "winevent", "security", "system_event",
}

# Firewall / network security product names
FIREWALL_PRODUCTS = {
    "Cisco", "Palo Alto", "Fortinet", "Check Point", "pfSense",
    "Suricata", "Snort", "iptables", "nftables", "firewalld",
    "Juniper", "SonicWall", "Zscaler", "Barracuda",
}

WEB_MESSAGE_TYPES = {
    "access_log", "web_access", "http_log", "apache_access",
    "nginx_access", "iis_log", "web_proxy", "squid_access",
}

CLOUDTRAIL_PIPELINES = {"aws_cloudtrail"}

NETWORK_FLOW_PIPELINES = {"network_flows"}

# ── Prefix patterns to strip ──────────────────────────────────────────────

_WINLOG_PREFIX = re.compile(r"^\S+\s+:::\s*", re.DOTALL)
_FIREWALL_PREFIX = re.compile(r"^\S+-Artifact\s+:::\s*", re.DOTALL)

# ── Mapping logic ─────────────────────────────────────────────────────────


def infer_source_type(row: dict) -> str | None:
    """Return a pipeline source_type hint, or None for auto-detect."""
    msg_type = str(row.get("message_type", ""))
    pipeline = str(row.get("pipeline", ""))
    product = str(row.get("product_name", ""))
    vendor = str(row.get("vendor_name", ""))

    # CloudTrail
    if pipeline in CLOUDTRAIL_PIPELINES:
        return "cloudtrail"

    # Network flows
    if pipeline in NETWORK_FLOW_PIPELINES:
        return None  # generic

    # Windows Event Logs
    if msg_type in WINEVENT_IDS or msg_type in WINEVENT_SPECIAL:
        return "winevent"

    # Firewall events
    if msg_type == "firewall_action":
        return "firewall"
    if any(p in product for p in FIREWALL_PRODUCTS):
        return "firewall"
    if any(p in vendor for p in FIREWALL_PRODUCTS):
        return "firewall"

    # Web / access logs
    if msg_type in WEB_MESSAGE_TYPES:
        return "web"

    # Syslog-default
    return "syslog"


def clean_message(raw: str, source_type: str | None) -> str:
    """Strip WitFoo-specific prefixes to expose the raw log underneath."""
    if source_type == "winevent":
        # Strip "HOST-XXXX-WinLogBeat ::: { ... }" prefix to get raw JSON
        cleaned = _WINLOG_PREFIX.sub("", raw).strip()
        if cleaned.startswith("{"):
            return cleaned
        # Fall through to look for embedded JSON
        brace_idx = raw.find("{")
        if brace_idx >= 0:
            return raw[brace_idx:]
        return raw

    if source_type == "firewall":
        # Strip "HOST-XXXX-Artifact ::: key=value ::: message={...}" prefix
        cleaned = _FIREWALL_PREFIX.sub("", raw).strip()
        # If there's a nested "message={...}" inside, extract it
        msg_match = re.search(r'message=\{(.+)\}', cleaned, re.DOTALL)
        if msg_match:
            return msg_match.group(0)
        return cleaned

    # For syslog/web, the message is typically already in raw syslog format
    if source_type in ("syslog", "web"):
        return raw

    return raw


def convert(
    input_path: str,
    output_dir: str,
    max_events: int = 0,
    sample_per_type: int = 500,
) -> None:
    """Convert WitFoo parquet to per-source-type JSONL files."""
    df = pd.read_parquet(input_path)

    # Only use events with an actual message
    df = df[df["message_sanitized"].notna() & (df["message_sanitized"] != "")]
    total = len(df)
    print(f"Loaded {total} events with messages")

    if max_events > 0 and total > max_events:
        df = df.head(max_events)
        print(f"Limited to {max_events} events")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}
    writers: dict[str, object] = {}

    for _, row in df.iterrows():
        rec = row.to_dict()
        source_type = infer_source_type(rec)
        raw_message = rec.get("message_sanitized", "")

        # Clean the message for better parsing
        cleaned = clean_message(raw_message, source_type)

        # Build the ScoreRequest-like record
        entry = {"raw": cleaned}
        if source_type:
            entry["source_type"] = source_type

        # Plus metadata for verification
        entry["_message_type"] = rec.get("message_type", "")
        entry["_pipeline"] = rec.get("pipeline", "")
        entry["_label"] = rec.get("label_binary", "")
        entry["_suspicion_score"] = float(rec.get("suspicion_score", 0))
        entry["_attack_techniques"] = rec.get("attack_techniques", "")
        entry["_attack_tactics"] = rec.get("attack_tactics", "")
        entry["_product"] = rec.get("product_name", "")

        if source_type not in writers:
            fpath = out_dir / f"{source_type}.jsonl"
            writers[source_type] = open(fpath, "w")
            counters[source_type] = 0

        writers[source_type].write(json.dumps(entry, default=str) + "\n")
        counters[source_type] = counters.get(source_type, 0) + 1

    for src_type, fh in writers.items():
        fh.close()
        print(f"  {src_type}: {counters[src_type]} events → {out_dir / src_type}.jsonl")

    print(f"\nDone. {sum(counters.values())} total events written.")
    print("\nSource type distribution:")
    for src_type, count in sorted(counters.items(), key=lambda x: -x[1]):
        print(f"  {src_type:15s} {count:6d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert WitFoo parquet to JSONL")
    parser.add_argument("--input", default="demo_data/witfoo/signals/signals.parquet")
    parser.add_argument("--output-dir", default="demo_data/witfoo/jsonl")
    parser.add_argument("--max-events", type=int, default=50000,
                        help="Max events to process (0 = all)")
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_dir=args.output_dir,
        max_events=args.max_events,
    )


if __name__ == "__main__":
    main()
