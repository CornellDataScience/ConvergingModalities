"""Validate resumable cache identity across the source-only folder migration."""


def resume_identity(cached, current, legacy_hash):
    """Accept the known old source hash only when all extraction settings match.

    Return the cached identity so resumed chunks keep one consistent provenance.
    Arbitrary source hashes and changed model/device/batch settings still fail.
    """
    previous = dict(current, extractor_hash=legacy_hash)
    if cached != current and (legacy_hash is None or cached != previous):
        raise ValueError("Feature cache identity mismatch")
    return cached


def migrated_legacy_hash(source, legacy_hash):
    """Allow migration only for the recorded, unmodified reorganized extractor."""
    import hashlib
    import json
    from pathlib import Path

    migrations = json.loads(
        Path(__file__).with_name("extractor_migrations.json").read_text()
    )
    current_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return legacy_hash if migrations.get(Path(source).name) == current_hash else None
