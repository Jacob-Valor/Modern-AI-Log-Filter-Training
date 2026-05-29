# Sigma rules

Place vetted Sigma rule files in this directory using the `*.yml` extension.
The runtime scorer looks for `config/sigma_rules/*.yml` and safely no-ops when
the directory is empty or pySigma is unavailable.

Do not treat this directory as a dumping ground for unvalidated community rules.
Before production use, tune each rule against representative logs, document the
expected false-positive behavior, and review any environment-specific fields.
