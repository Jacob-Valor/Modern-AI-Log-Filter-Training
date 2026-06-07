#!/usr/bin/env bash
# Elasticsearch snapshot backup for raw-logs indices.
# Usage: bash scripts/es_backup.sh [snapshot-name]
# Default snapshot name: backup-YYYYMMDD-HHMMSS

set -euo pipefail

ES_HOST="${ES_HOST:-http://localhost:9200}"
ES_USER="${ES_USER:-elastic}"
ES_PASS="${ES_PASSWORD:?Set ES_PASSWORD first}"
SNAPSHOT_REPO="${ES_SNAPSHOT_REPO:-logfilter-backups}"
SNAPSHOT_NAME="${1:-backup-$(date +%Y%m%d-%H%M%S)}"
INDEX_PREFIX="${INDEX_PREFIX:-raw-logs}"

echo "==> Creating snapshot repository if missing..."
curl -sf -u "${ES_USER}:${ES_PASS}" -X PUT "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}" \
  -H 'Content-Type: application/json' \
  -d '{
  "type": "fs",
  "settings": {
    "location": "/usr/share/elasticsearch/data/backup",
    "compress": true
  }
}' 2>/dev/null || echo "    (repository already exists or FS path not configured — continuing)"

echo "==> Creating snapshot: ${SNAPSHOT_NAME}"
RESP=$(curl -sf -u "${ES_USER}:${ES_PASS}" -X PUT \
  "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}/${SNAPSHOT_NAME}?wait_for_completion=true" \
  -H 'Content-Type: application/json' \
  -d "{
  \"indices\": \"${INDEX_PREFIX}-*\",
  \"ignore_unavailable\": true,
  \"include_global_state\": false
}" 2>&1)

echo "${RESP}" | python3 -m json.tool 2>/dev/null || echo "${RESP}"
echo ""
echo "==> Snapshot '${SNAPSHOT_NAME}' complete."
