from __future__ import annotations

import numpy as np

from apexoracle.data.peptide_classifier import (
    TEST,
    TRAIN,
    VALIDATION,
    source_from_mol_id,
    split_codes_for_digests,
    stable_digest,
)
from apexoracle.training.peptide_classifier import (
    deterministic_mask,
    move_chance_at_t,
)


def test_source_prefixes_cover_v1_data():
    assert source_from_mol_id("SmProt2_1") == "smprot2"
    assert source_from_mol_id("uni_2") == "uniprot_uniref"
    assert source_from_mol_id("Generated_pep_CLM_3") == "generated_peptideclm"
    assert source_from_mol_id("pubchem_4") == "pubchem"


def test_sequence_component_overrides_molecule_split():
    first = stable_digest("first")
    second = stable_digest("second")
    ordinary = stable_digest("ordinary")
    real = np.sort(np.asarray([first, second], dtype="V16"))
    roots = np.full(2, 77, dtype=np.uint64)
    codes, molecule_ids = split_codes_for_digests(
        np.asarray([first, second, ordinary], dtype="V16"),
        seed=20260726,
        real_canonicals=real,
        real_sequence_roots=roots,
    )
    assert codes[0] == codes[1]
    assert set(codes.tolist()) <= {int(TRAIN), int(VALIDATION), int(TEST)}
    assert molecule_ids[0] != molecule_ids[1]


def test_stateless_noise_is_reproducible_and_molecule_specific():
    molecules = np.asarray([11, 12], dtype=np.uint64)
    chance = move_chance_at_t(0.5)
    first = deterministic_mask(
        molecules, 1024, replicate=0, move_chance=chance
    )
    repeated = deterministic_mask(
        molecules, 1024, replicate=0, move_chance=chance
    )
    next_replicate = deterministic_mask(
        molecules, 1024, replicate=1, move_chance=chance
    )
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first[0], first[1])
    assert not np.array_equal(first, next_replicate)
    assert abs(first.mean() - 0.4995) < 0.05
