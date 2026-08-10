# Data notice

The MIT License in this repository applies to ApexOracle Core source code. It
does not relicense third-party datasets, database records, model weights, or
other externally sourced assets.

## DBAASP-derived validation table

`experiments/peplink_validation/peplink_0.1.2/roundtrip_records.csv` contains
16,896 row-level validation records derived from peptide records identified in
the Database of Antimicrobial Activity and Structure of Peptides (DBAASP). It
is retained to make the PepLink 0.1.2 round-trip analysis supplied during peer
review independently inspectable. Users should cite DBAASP and comply with the
database's current terms when reusing these records.

- Database: <https://dbaasp.org/>
- Terms and Conditions: <https://dbaasp.org/docs/DBAASP_Terms_And_Conditions.pdf>
- API policy: <https://dbaasp.org/api?page=rest>

The terms reviewed on 2026-08-10 contained both public-domain/free-
distribution language and a visitor non-distribution clause. The project
authors therefore do not represent this table as MIT-licensed or grant rights
in the underlying DBAASP records. The authors elected to publish the existing
derived audit table for scientific-review reproducibility despite that textual
ambiguity. This is a provenance and scope notice, not legal advice or a claim
of permission from DBAASP.

Questions or requests concerning this derived table may be raised through the
repository issue tracker. If the database owner requests a change, the project
will review the affected release asset while preserving aggregate validation
results and reproducible code where possible.

## Other assets

Model checkpoints and example embeddings are distributed separately with
their own model cards and provenance manifests. Source datasets, pretrained
models, and optional dependencies retain the terms stated by their respective
providers.
