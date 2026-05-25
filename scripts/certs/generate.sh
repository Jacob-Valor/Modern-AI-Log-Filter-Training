#!/usr/bin/env bash
# Generate self-signed Kafka TLS certs and SASL credentials for local development.
# In production, use a proper PKI (e.g. cert-manager in Kubernetes).

set -euo pipefail

cd "$(dirname "$0")"

PASS="logfilter-dev"
CN="kafka"

echo "Generating Kafka TLS certificates …"

# Clean up old artifacts
rm -f *.jks *.creds

# Create keystore
keytool -keystore kafka.server.keystore.jks -alias localhost -validity 365 -genkey -keyalg RSA \
  -storepass "$PASS" -keypass "$PASS" -dname "CN=$CN, OU=Dev, O=LogFilter, L=Local, ST=NA, C=XX" \
  -ext "SAN=dns:localhost,dns:kafka,ip:127.0.0.1"

# Create CA
openssl req -new -x509 -keyout ca-key -out ca-cert -days 365 -subj "/CN=LogFilter-Dev-CA" \
  -passout pass:"$PASS"

# Generate CSR
certreq="ca-cert-req"
keytool -keystore kafka.server.keystore.jks -alias localhost -certreq -file "$certreq" \
  -storepass "$PASS" -keypass "$PASS"

# Sign broker cert with CA
openssl x509 -req -CA ca-cert -CAkey ca-key -in "$certreq" -out ca-cert-signed -days 365 \
  -passin pass:"$PASS" -CAcreateserial

# Import CA and signed cert into keystore
keytool -keystore kafka.server.keystore.jks -alias CARoot -import -file ca-cert \
  -storepass "$PASS" -keypass "$PASS" -noprompt
keytool -keystore kafka.server.keystore.jks -alias localhost -import -file ca-cert-signed \
  -storepass "$PASS" -keypass "$PASS" -noprompt

# Create truststore
keytool -keystore kafka.server.truststore.jks -alias CARoot -import -file ca-cert \
  -storepass "$PASS" -keypass "$PASS" -noprompt

# Write credential files for Docker secrets
echo "$PASS" > keystore_creds
echo "$PASS" > sslkey_creds
echo "$PASS" > truststore_creds

# Generate SASL SCRAM credentials for logfilter user
echo "logfilter" > sasl_username
echo "$(openssl rand -base64 32)" > sasl_password

echo "Done. Files generated:"
ls -la *.jks *.creds sasl_* 2>/dev/null || true

echo ""
echo "To enable SASL_SSL in clients, set:"
echo "  KAFKA_SECURITY_PROTOCOL=SASL_SSL"
echo "  KAFKA_SASL_MECHANISM=SCRAM-SHA-512"
echo "  KAFKA_SASL_USERNAME=logfilter"
echo "  KAFKA_SASL_PASSWORD=<from sasl_password>"
