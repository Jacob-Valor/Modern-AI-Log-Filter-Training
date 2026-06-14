#!/usr/bin/env python3
"""
Demo script for AI-Powered Threat Detection System
Shows real-time scoring of syslog messages
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.config import load_config
from logfilter.pipeline.scorer import LogScorer
from logfilter.pipeline.normalizer import LogNormalizer

# ANSI colors for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header():
    print(f"\n{'='*70}")
    print(f"{BOLD}  AI-Powered Cyber Threat Detection System - Live Demo{RESET}")
    print(f"{'='*70}\n")

def print_event(log_line, result, latency_ms):
    # Color based on priority
    if result.ai_priority == "HIGH":
        color = RED
    elif result.ai_priority == "MEDIUM":
        color = YELLOW
    elif result.ai_priority == "LOW":
        color = BLUE
    else:
        color = GREEN
    
    print(f"{BOLD}Input:{RESET} {log_line[:80]}...")
    print(f"{color}{BOLD}Output:{RESET} Priority={color}{result.ai_priority}{RESET} | "
          f"Score={result.ai_threat_score:.4f} | "
          f"Latency={latency_ms:.1f}ms")
    if result.ai_mitre_technique:
        print(f"  {BOLD}MITRE:{RESET} {result.ai_mitre_technique}")
    print()

def main():
    print_header()
    
    # Load config and models
    print(f"{BOLD}[1/3] Loading AI models...{RESET}")
    config = load_config()
    scorer = LogScorer(config)
    normalizer = LogNormalizer()
    print(f"{GREEN}✓ All models loaded successfully{RESET}\n")
    
    # Demo syslog messages (realistic attack patterns)
    demo_logs = [
        # Normal activity
        "Jun  9 12:00:01 webserver sshd[12345]: Accepted password for admin from 10.0.0.50 port 22 ssh2",
        
        # Brute force attack
        "Jun  9 12:00:02 webserver sshd[12346]: Failed password for root from 192.168.1.100 port 22 ssh2",
        "Jun  9 12:00:03 webserver sshd[12347]: Failed password for root from 192.168.1.100 port 22 ssh2",
        "Jun  9 12:00:04 webserver sshd[12348]: Failed password for root from 192.168.1.100 port 22 ssh2",
        
        # Firewall block
        "Jun  9 12:00:05 webserver kernel: [UFW BLOCK] IN=eth0 OUT= SRC=192.168.1.200 DST=10.0.0.1 PROTO=TCP DPT=443",
        
        # Privilege escalation
        "Jun  9 12:00:06 webserver sudo: admin : TTY=pts/0 ; PWD=/home/admin ; USER=root ; COMMAND=/bin/bash",
        
        # SQL injection attempt
        "Jun  9 12:00:07 webserver apache2[192.168.1.1]: 192.168.1.1 - - \"GET /admin?id=1' OR '1'='1 HTTP/1.1\" 403 4523",
        
        # Malware download attempt
        "Jun  9 12:00:08 webserver snort[2345]: [1:1000001:0] MALWARE-C2 Beacon detected from 10.0.0.100 to external 45.33.32.156",
    ]
    
    print(f"{BOLD}[2/3] Processing {len(demo_logs)} syslog messages...{RESET}\n")
    print(f"{'-'*70}")
    
    import time
    total_latency = 0
    
    for i, log in enumerate(demo_logs, 1):
        event = normalizer.normalize(log)
        t0 = time.time()
        result = scorer.score(event)
        latency = (time.time() - t0) * 1000
        total_latency += latency
        
        print_event(log, result, latency)
    
    print(f"{'-'*70}")
    
    # Summary
    avg_latency = total_latency / len(demo_logs)
    print(f"{BOLD}[3/3] Demo Summary{RESET}\n")
    print(f"  Messages Processed: {len(demo_logs)}")
    print(f"  Average Latency:   {avg_latency:.1f}ms")
    print(f"  Throughput:        {1000/avg_latency:.1f} logs/sec")
    print(f"\n{GREEN}✓ Demo completed successfully!{RESET}\n")

if __name__ == "__main__":
    main()
