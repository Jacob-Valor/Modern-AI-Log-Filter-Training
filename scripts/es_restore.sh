#!/usr/bin/env bash
# Elasticsearch snapshot restore for raw-logs indices.
# Usage: bash scripts/es_restore.sh <snapshot-name>
# List available snapshots: curl -u "${ES_USER}:${ES_PASS}" "${ES_HOST}/_snapshot/logfilter-backups/_all"

set -euo pipefail

ES_HOST="${ES_HOST:-http://localhost:9200}"
ES_USER="${ES_USER:-elastic}"
ES_PASS="${ES_PASSWORD:?Set ES_PASSWORD first}"
SNAPSHOT_REPO="${ES_SNAPSHOT_REPO:-logfilter-backups}"
INDEX_PREFIX="${INDEX_PREFIX:-raw-logs}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <snapshot-name>"
  echo ""
  echo "Available snapshots:"
  curl -sf -u "${ES_USER}:${ES_PASS}" "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}/_all" | \
    python3 -c "import sys,json; [print(f\"  {s['snapshot']}\") for s in json.load(sys.stdin).get('snapshots',[])]" 2>/dev/null || echo "  (none or repository not configured)"
  exit 1
fi

SNAPSHOT_NAME="$1"

echo "==> Restoring snapshot: ${SNAPSHOT_NAME}"
echo "    WARNING: This will overwrite existing ${INDEX_PREFIX}-* indices."

read -rp "Continue? [y/N] " confirm
if [[ ! "$confirm" =~ ^[yY]$ ]]; then
  echo "Aborted."
  exit 0
fi

# Close indices to allow restore
echo "==> Closing indices matching ${INDEX_PREFIX}-*..."
curl -sf -u "${ES_USER}:${ES_PASS}" -X POST "${ES_HOST}/${INDEX_PREFIX}-*/_close" \
  -H 'Content-Type: application/json' 2>/dev/null || true

RESP=$(curl -sf -u "${ES_USER}:${ES_PASS}" -X POST \
  "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}/${SNAPSHOT_NAME}/_restore?wait_for_completion=true" \
  -H 'Content-Type: application/json' \
  -d "{
  \"indices\": \"${INDEX_PREFIX}-*\",
  \"ignore_unavailable\": true,
  \"include_global_state\": false
}" 2>&1)

echo "${RESP}" | python3 -m json.tool 2>/dev/null || echo "${RESP}"
echo ""
echo "==> Restore of '${SNAPSHOT_NAME}' complete."
