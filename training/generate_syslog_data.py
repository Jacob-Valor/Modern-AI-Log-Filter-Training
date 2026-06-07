"""
Generate synthetic syslog training data for Tier-1 classifier retraining.

Produces bag-of-events count vectors covering:
- SSH brute force attacks
- Web attacks (path traversal, SQLi, XSS)
- Firewall/IDS alerts
- Authentication failures
- Windows event anomalies
- Benign traffic (normal SSH, HTTP, system logs)

Output: CSV files compatible with training/train.py format.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ── Event vocabulary ───────────────────────────────────────────────────────────
# Each event template: (category, event_name, severity_weight)
# severity_weight: higher = more suspicious

EVENT_TEMPLATES: list[tuple[str, str, float]] = [
    # ── SSH events ─────────────────────────────────────────────────────────────
    # Feature names use tokens that appear in real sshd log lines so the
    # runtime bag-of-events scorer can match them via token overlap.
    ("ssh", "sshd+failed password", 0.9),
    ("ssh", "sshd+failed password invalid user", 0.95),
    ("ssh", "sshd+connection closed authenticating", 0.7),
    ("ssh", "sshd+disconnecting authenticating too many", 0.85),
    ("ssh", "sshd+accepted publickey", 0.1),
    ("ssh", "sshd+accepted password", 0.15),
    ("ssh", "sshd+pam_unix session opened", 0.1),
    ("ssh", "sshd+pam_unix session closed", 0.1),
    ("ssh", "sshd+illegal user attempt", 0.8),
    ("ssh", "sshd+reverse mapping getaddrinfo", 0.3),
    ("ssh", "sshd+did not receive identification", 0.5),
    ("ssh", "sshd+connection reset peer", 0.4),
    ("ssh", "sshd+timeout waiting authentication", 0.3),
    ("ssh", "sshd+maximum authentication attempts exceeded", 0.7),

    # ── Web server events ──────────────────────────────────────────────────────
    # Use tokens from real access-log and error-log lines.
    ("web", "get+request+returned", 0.1),
    ("web", "post+request+returned", 0.15),
    ("web", "429+rate limit exceeded", 0.4),
    ("web", "directory traversal+passwd", 0.95),
    ("web", "sql injection+select union", 0.95),
    ("web", "xss+script+query string", 0.9),
    ("web", "command injection+exec", 0.95),
    ("web", "file inclusion+etc passwd", 0.9),
    ("web", "sqlmap+scanner", 0.85),
    ("web", "wordpress+wp-login denied", 0.6),
    ("web", "login failed+401 unauthorized", 0.5),
    ("web", "session hijack+cookie steal", 0.9),
    ("web", "path traversal+dotdot", 0.85),
    ("web", "413+request body too large", 0.3),
    ("web", "504+upstream timeout", 0.2),
    ("web", "ssl handshake failure", 0.4),
    ("web", "403+forbidden request", 0.3),
    ("web", "500+internal server error", 0.2),
    ("web", "503+service unavailable", 0.2),

    # ── Firewall / IDS events ─────────────────────────────────────────────────
    # Use tokens from real iptables/snort/CEF log lines.
    ("firewall", "iptables+DROP+INPUT", 0.6),
    ("firewall", "iptables+DROP+FORWARD", 0.6),
    ("firewall", "iptables+ACCEPT+ESTABLISHED", 0.1),
    ("firewall", "ufw+BLOCK+IN", 0.7),
    ("firewall", "ufw+ALLOW+OUT", 0.1),
    ("firewall", "snort+eternalblue+MS17-010", 0.99),
    ("snort", "snort+sql+injection", 0.95),
    ("snort", "snort+xss+attack", 0.9),
    ("snort", "snort+port scan", 0.7),
    ("snort", "snort+brute force", 0.85),
    ("snort", "snort+malware download", 0.95),
    ("snort", "snort+c2+communication", 0.99),
    ("snort", "snort+dns tunnel", 0.9),
    ("firewall", "paloalto+threat+ssh brute", 0.85),
    ("firewall", "paloalto+threat+port scan", 0.7),
    ("firewall", "paloalto+traffic+allowed", 0.05),
    ("firewall", "paloalto+traffic+denied", 0.5),
    ("firewall", "fortigate+intrusion detected", 0.8),
    ("firewall", "fortigate+malware blocked", 0.9),

    # ── Authentication events ──────────────────────────────────────────────────
    # Use tokens from real auth.log / secure lines.
    ("auth", "sudo+COMMAND", 0.15),
    ("auth", "sudo+authentication failure", 0.6),
    ("auth", "failed login+from host", 0.7),
    ("auth", "accepted password+from host", 0.1),
    ("auth", "pam_unix+authentication failure", 0.75),
    ("auth", "pam_unix+authentication success", 0.1),
    ("auth", "pam+account locked", 0.8),
    ("auth", "pam+password expired", 0.3),
    ("auth", "sssd+authentication failed", 0.7),
    ("sssd", "sssd+user account created", 0.6),
    ("auth", "krb5+principal unknown", 0.5),
    ("auth", "ldap+bind authentication failed", 0.7),
    ("auth", "radius+access reject", 0.6),
    ("auth", "oauth+token validation failed", 0.7),
    ("auth", "oauth+token refresh successful", 0.1),

    # ── Windows events ─────────────────────────────────────────────────────────
    # Use tokens from real Windows Event XML / text exports.
    ("windows", "eventid+4625+failed logon", 0.7),
    ("windows", "eventid+4624+successful logon", 0.1),
    ("windows", "eventid+4720+user account created", 0.6),
    ("windows", "eventid+4726+user account deleted", 0.5),
    ("windows", "eventid+4723+password changed", 0.3),
    ("windows", "eventid+4672+special privileges", 0.95),
    ("windows", "eventid+4688+process creation", 0.2),
    ("windows", "eventid+7045+service installed", 0.6),
    ("windows", "eventid+4657+firewall rule", 0.4),
    ("windows", "eventid+1102+audit log cleared", 0.9),
    ("windows", "eventid+4740+account lockout", 0.7),
    ("windows", "eventid+4625+interactive logon", 0.7),
    ("windows", "eventid+4672+privileges assigned", 0.8),
    ("windows", "eventid+4719+security policy changed", 0.6),

    # ── System / infrastructure events ─────────────────────────────────────────
    # Use tokens from real syslog / journal lines.
    ("system", "systemd+Started", 0.05),
    ("system", "systemd+Stopped", 0.05),
    ("system", "kernel+segfault at", 0.5),
    ("system", "kernel+out of memory", 0.4),
    ("system", "cron+CROND+CMD", 0.05),
    ("system", "rsyslog+ forwarded", 0.05),
    ("system", "ntpd+time sync", 0.05),
    ("system", "dhclient+lease", 0.1),
    ("system", "named+query", 0.05),
    ("system", "sshd+connection established", 0.1),
    ("system", "sshd+connection closed", 0.1),

    # ── HDFS trace events (keep existing vocabulary overlap) ───────────────────
    ("hdfs", "getfileinfo+success", 0.1),
    ("hdfs", "getblocklocations+success", 0.1),
    ("hdfs", "bestnode+chosen", 0.1),
    ("hdfs", "create+nodereport", 0.1),
    ("hdfs", "delete+success", 0.1),
    ("hdfs", "rename+success", 0.1),
    ("hdfs", "append+success", 0.1),
    ("hdfs", "getlisting+success", 0.1),
]

# ── Scenario templates ─────────────────────────────────────────────────────────
# Each scenario defines which events to combine and their relative frequencies

SCENARIOS: list[tuple[str, list[tuple[str, int, int]], bool]] = [
    # (name, [(event_name, min_count, max_count), ...], is_anomaly)

    # ── Benign scenarios (label=0) ─────────────────────────────────────────────
    (
        "normal_ssh_session",
        [
            ("sshd+accepted publickey", 1, 2),
            ("sshd+pam_unix session opened", 1, 1),
            ("systemd+Started", 1, 2),
            ("sshd+pam_unix session closed", 1, 1),
            ("sshd+connection closed authenticating", 0, 1),
        ],
        False,
    ),
    (
        "normal_web_traffic",
        [
            ("get+request+returned", 10, 50),
            ("post+request+returned", 1, 5),
            ("503+service unavailable", 0, 1),
            ("500+internal server error", 0, 2),
        ],
        False,
    ),
    (
        "normal_system_operations",
        [
            ("systemd+Started", 5, 20),
            ("systemd+Stopped", 3, 15),
            ("cron+CROND+CMD", 2, 10),
            ("rsyslog+ forwarded", 10, 50),
            ("ntpd+time sync", 1, 3),
        ],
        False,
    ),
    (
        "normal_firewall_traffic",
        [
            ("iptables+ACCEPT+ESTABLISHED", 20, 100),
            ("paloalto+traffic+allowed", 10, 50),
            ("ufw+ALLOW+OUT", 5, 20),
        ],
        False,
    ),
    (
        "normal_dns_dhcp",
        [
            ("named+query", 20, 100),
            ("dhclient+lease", 1, 5),
        ],
        False,
    ),
    (
        "normal_windows_station",
        [
            ("eventid+4624+successful logon", 5, 20),
            ("eventid+4688+process creation", 50, 200),
            ("eventid+4723+password changed", 0, 1),
            ("eventid+4657+firewall rule", 0, 2),
        ],
        False,
    ),
    (
        "normal_hdfs_operations",
        [
            ("getfileinfo+success", 5, 20),
            ("getblocklocations+success", 3, 10),
            ("bestnode+chosen", 2, 5),
            ("create+nodereport", 1, 3),
        ],
        False,
    ),

    # ── Anomalous scenarios (label=1) ─────────────────────────────────────────
    (
        "ssh_brute_force",
        [
            ("sshd+failed password", 5, 50),
            ("sshd+failed password invalid user", 2, 20),
            ("sshd+connection closed authenticating", 3, 30),
            ("sshd+disconnecting authenticating too many", 1, 5),
            ("sshd+maximum authentication attempts exceeded", 1, 3),
            ("sshd+illegal user attempt", 1, 10),
            ("sudo+COMMAND", 0, 1),
        ],
        True,
    ),
    (
        "ssh_brute_force_with_privilege_escalation",
        [
            ("sshd+failed password", 10, 100),
            ("sshd+accepted publickey", 0, 1),
            ("sudo+COMMAND", 0, 1),
            ("sudo+authentication failure", 0, 2),
            ("pam_unix+authentication failure", 2, 10),
            ("pam+account locked", 0, 1),
        ],
        True,
    ),
    (
        "web_attack_path_traversal",
        [
            ("directory traversal+passwd", 3, 20),
            ("path traversal+dotdot", 2, 15),
            ("file inclusion+etc passwd", 1, 10),
            ("get+request+returned", 5, 20),
        ],
        True,
    ),
    (
        "web_attack_sqli",
        [
            ("sql injection+select union", 2, 15),
            ("get+request+returned", 5, 20),
            ("post+request+returned", 2, 10),
        ],
        True,
    ),
    (
        "web_attack_xss",
        [
            ("xss+script+query string", 2, 10),
            ("xss+script+query string", 1, 8),
            ("get+request+returned", 5, 20),
        ],
        True,
    ),
    (
        "web_attack_scanning",
        [
            ("sqlmap+scanner", 3, 20),
            ("snort+port scan", 1, 5),
            ("get+request+returned", 10, 50),
        ],
        True,
    ),
    (
        "firewall_exploit",
        [
            ("snort+eternalblue+MS17-010", 1, 3),
            ("iptables+DROP+INPUT", 5, 20),
            ("snort+malware download", 0, 2),
        ],
        True,
    ),
    (
        "firewall_c2_communication",
        [
            ("snort+c2+communication", 1, 3),
            ("snort+dns tunnel", 0, 2),
            ("iptables+DROP+FORWARD", 3, 10),
        ],
        True,
    ),
    (
        "auth_failure_cascade",
        [
            ("pam_unix+authentication failure", 5, 30),
            ("sssd+authentication failed", 2, 10),
            ("failed login+from host", 3, 15),
            ("pam+account locked", 0, 3),
            ("ldap+bind authentication failed", 0, 5),
        ],
        True,
    ),
    (
        "windows_privilege_escalation",
        [
            ("eventid+4672+special privileges", 1, 5),
            ("eventid+4672+privileges assigned", 1, 3),
            ("eventid+1102+audit log cleared", 0, 2),
            ("eventid+4719+security policy changed", 0, 2),
            ("eventid+4625+failed logon", 2, 10),
        ],
        True,
    ),
    (
        "windows_account_manipulation",
        [
            ("eventid+4720+user account created", 2, 10),
            ("eventid+7045+service installed", 1, 5),
            ("eventid+4740+account lockout", 1, 3),
            ("eventid+4625+failed logon", 3, 15),
        ],
        True,
    ),
    (
        "windows_log_tampering",
        [
            ("eventid+1102+audit log cleared", 1, 3),
            ("eventid+4719+security policy changed", 1, 3),
            ("eventid+4657+firewall rule", 1, 5),
            ("eventid+4688+process creation", 20, 100),
        ],
        True,
    ),
    (
        "multi_vector_attack",
        [
            ("sshd+failed password", 5, 30),
            ("directory traversal+passwd", 2, 10),
            ("snort+port scan", 1, 5),
            ("pam_unix+authentication failure", 3, 15),
            ("iptables+DROP+INPUT", 5, 20),
            ("sql injection+select union", 1, 5),
        ],
        True,
    ),
]


def build_event_index() -> dict[str, int]:
    """Map event template names to column indices."""
    return {name: idx for idx, (_, name, _) in enumerate(EVENT_TEMPLATES)}


def generate_sample(
    scenario: tuple[str, list[tuple[str, int, int]], bool],
    event_index: dict[str, int],
    n_events: int,
) -> np.ndarray:
    """Generate a single count vector for a scenario."""
    vec = np.zeros(n_events, dtype=np.float32)
    _, event_counts, _ = scenario
    for event_name, lo, hi in event_counts:
        if event_name in event_index:
            vec[event_index[event_name]] = random.randint(lo, hi)
    return vec


def generate_dataset(
    n_normal: int = 10000,
    n_anomaly: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a full training dataset."""
    random.seed(seed)
    np.random.seed(seed)

    event_index = build_event_index()
    n_events = len(EVENT_TEMPLATES)

    normal_scenarios = [s for s in SCENARIOS if not s[2]]
    anomaly_scenarios = [s for s in SCENARIOS if s[2]]

    rows: list[np.ndarray] = []
    labels: list[int] = []

    # Generate normal samples
    for _ in range(n_normal):
        scenario = random.choice(normal_scenarios)
        rows.append(generate_sample(scenario, event_index, n_events))
        labels.append(0)

    # Generate anomaly samples
    for _ in range(n_anomaly):
        scenario = random.choice(anomaly_scenarios)
        rows.append(generate_sample(scenario, event_index, n_events))
        labels.append(1)

    columns = [name for _, name, _ in EVENT_TEMPLATES]
    df = pd.DataFrame(np.array(rows), columns=columns)
    y = pd.Series(labels, name="label")

    return df, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic syslog training data"
    )
    parser.add_argument(
        "--n-normal", type=int, default=10000,
        help="Number of normal (benign) samples (default: 10000)"
    )
    parser.add_argument(
        "--n-anomaly", type=int, default=2000,
        help="Number of anomaly (attack) samples (default: 2000)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="HDFS_v3_TraceBench/preprocessed",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_normal} normal + {args.n_anomaly} anomaly samples...")
    df, y = generate_dataset(args.n_normal, args.n_anomaly, args.seed)

    # Add TaskID column (required by training pipeline)
    df.insert(0, "TaskID", range(len(df)))

    # Split into normal and failure CSVs
    normal_mask = y == 0
    failure_mask = y == 1

    normal_df = df[normal_mask].copy()
    failure_df = df[failure_mask].copy()

    normal_path = output_dir / "real_syslog_normal.csv"
    failure_path = output_dir / "real_syslog_failure.csv"

    normal_df.to_csv(normal_path, index=False)
    failure_df.to_csv(failure_path, index=False)

    print(f"Normal samples: {len(normal_df)} → {normal_path}")
    print(f"Failure samples: {len(failure_df)} → {failure_path}")
    print(f"Event vocabulary size: {len(EVENT_TEMPLATES)}")
    print(f"Scenario types: {len(SCENARIOS)}")
    print(f"  Normal: {len([s for s in SCENARIOS if not s[2]])}")
    print(f"  Anomaly: {len([s for s in SCENARIOS if s[2]])}")


if __name__ == "__main__":
    main()
