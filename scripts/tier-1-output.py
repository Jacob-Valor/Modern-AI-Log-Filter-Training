import json
import sys

sys.path.insert(0, "src")
from logfilter.config import load_config
from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer

config = load_config()
scorer = LogScorer(config)
normalizer = LogNormalizer()

event = normalizer.normalize(
    "Jan 15 11:07:53 prod-srv01 sshd[22345]: Accepted publickey for admin from 192.168.1.100 port 22 ssh2"
)
result = scorer.score(event)
print("classifier_score (Tier-1):", result.classifier_score)
print("tier2_used:", result.tier2_used)
print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))
