#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PASS="logfilter-dev"
DAYS=365
SUBJ_CA="/CN=LogFilter-Dev-CA/O=LogFilter/L=Local"
SUBJ_SERVER="/CN=logfilter-api/OU=Dev/O=LogFilter/L=Local"

echo "==> Generating CA key + cert..."
openssl genrsa -out services-ca.key 2048 2>/dev/null
openssl req -new -x509 -key services-ca.key -out services-ca.crt -days "$DAYS" \
  -subj "$SUBJ_CA" -passout pass:"$PASS"

echo "==> Generating server key + CSR..."
openssl genrsa -out services-server.key 2048 2>/dev/null
openssl req -new -key services-server.key -out services-server.csr \
  -subj "$SUBJ_SERVER"

echo "==> Signing server cert with CA..."
cat > services-san.ext <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
DNS.2 = logfilter-api
DNS.3 = logfilter-collector
DNS.4 = logfilter-nginx
IP.1 = 127.0.0.1
EOF

openssl x509 -req -in services-server.csr -CA services-ca.crt -CAkey services-ca.key \
  -CAcreateserial -out services-server.crt -days "$DAYS" \
  -extfile services-san.ext 2>/dev/null

rm -f services-server.csr services-san.ext services-ca.srl

echo "==> Done. Generated:"
ls -la services-*.key services-*.crt 2>/dev/null
echo ""
echo "Use in docker-compose via volume mounts:"
echo "  ./scripts/certs/services-server.crt:/etc/tls/server.crt:ro"
echo "  ./scripts/certs/services-server.key:/etc/tls/server.key:ro"
