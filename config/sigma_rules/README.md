# Sigma rules

Place vetted Sigma rule files in this directory using the `*.yml` extension.
The runtime scorer safely parses the supported text-matching subset and no-ops
when the directory is empty. Complex conditions and structured fields are
intentionally left to a full Sigma backend.

Do not treat this directory as a dumping ground for unvalidated community rules.
Before production use, tune each rule against representative logs, document the
expected false-positive behavior, and review any environment-specific fields.
