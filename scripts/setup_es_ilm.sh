#!/usr/bin/env bash
# Elasticsearch ILM setup for raw-logs indices.
# Run once after ES is healthy: bash scripts/setup_es_ilm.sh

set -euo pipefail

ES_HOST="${ES_HOST:-http://localhost:9200}"
ES_USER="${ES_USER:-elastic}"
ES_PASS="${ES_PASSWORD:?Set ES_PASSWORD first}"
INDEX_PREFIX="${INDEX_PREFIX:-raw-logs}"

curl -sf -u "${ES_USER}:${ES_PASS}" -X PUT "${ES_HOST}/_ilm/policy/${INDEX_PREFIX}-policy" \
  -H 'Content-Type: application/json' \
  -d '{
  "policy": {
    "phases": {
      "hot": {
        "min_age": "0ms",
        "actions": {
          "rollover": {
            "max_primary_shard_size": "30gb",
            "max_age": "7d"
          },
          "set_priority": { "priority": 100 }
        }
      },
      "warm": {
        "min_age": "30d",
        "actions": {
          "shrink": { "number_of_shards": 1 },
          "forcemerge": { "max_num_segments": 1 },
          "set_priority": { "priority": 50 }
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}'

echo ""

curl -sf -u "${ES_USER}:${ES_PASS}" -X PUT "${ES_HOST}/_index_template/${INDEX_PREFIX}-template" \
  -H 'Content-Type: application/json' \
  -d "{
  \"index_patterns\": [\"${INDEX_PREFIX}-*\"],
  \"template\": {
    \"settings\": {
      \"number_of_shards\": 1,
      \"number_of_replicas\": 0,
      \"index.lifecycle.name\": \"${INDEX_PREFIX}-policy\",
      \"index.lifecycle.rollover_alias\": \"${INDEX_PREFIX}\"
    }
  },
  \"priority\": 200
}"

echo ""
echo "ILM policy '${INDEX_PREFIX}-policy' created: hot(7d/30gb) → warm(30d) → delete(90d)"
