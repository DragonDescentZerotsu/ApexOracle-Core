from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from apexoracle.evaluation.genome_condition_reviewer import (
    NON_BACTERIAL_SPECIES,
    build_saved_tensor_windows,
    build_fragment_windows,
    classify_annotation,
    overlapping_fragment_indices,
    parse_skani_sparse,
    select_nearest_same_species_neighbors,
)
from apexoracle.evaluation.genome_fragment_validation import (
    ProbeConfig,
    deterministic_probe_cohort,
    parse_boolean_series,
)
from scripts.audit.analyze_genome_fragment_variation import (
    mutual_best_alignments,
    normalize_pair_manifest,
    read_best_alignments,
)
from scripts.audit import analyze_genome_fragment_variation as fragment_analysis
from scripts.audit.plot_genome_representation_validation import (
    build_caption,
    normalize_svg_whitespace,
    plot_fragment_panel,
    prepare_fragment_plot_data,
)


def test_fragment_windows_include_terminal_partial_fragment_and_overlap_features():
    windows = build_fragment_windows([21_500, 500])
    assert windows == [
        {"fragment_index": 0, "contig_index": 0, "start": 0, "end": 11_000},
        {"fragment_index": 1, "contig_index": 0, "start": 10_000, "end": 21_000},
        {"fragment_index": 2, "contig_index": 0, "start": 20_000, "end": 21_500},
        {"fragment_index": 3, "contig_index": 1, "start": 0, "end": 500},
    ]
    assert overlapping_fragment_indices(
        windows, contig_index=0, start=10_500, end=10_800
    ) == [0, 1]
    assert overlapping_fragment_indices(
        windows, contig_index=1, start=100, end=200
    ) == [3]


def test_saved_tensor_window_reconstruction_matches_frozen_indexing():
    windows = build_saved_tensor_windows([21_500, 10_000, 35_000])
    assert windows == [
        {"fragment_index": 0, "contig_index": 0, "start": 0, "end": 11_000},
        {
            "fragment_index": 1,
            "contig_index": 0,
            "start": 10_000,
            "end": 21_000,
        },
        {
            "fragment_index": 2,
            "contig_index": 0,
            "start": 20_000,
            "end": 21_500,
        },
        {
            "fragment_index": 3,
            "contig_index": 2,
            "start": 30_000,
            "end": 35_000,
        },
    ]


def test_fragment_alignment_filter_and_mutual_best_contract(tmp_path):
    forward_path = tmp_path / "forward.paf"
    reverse_path = tmp_path / "reverse.paf"
    forward_path.write_text(
        "f0\t11000\t0\t10000\t+\tf2\t11000\t0\t10000\t9950\t10000\t60\n"
        "f1\t11000\t0\t10000\t-\tf3\t11000\t0\t10000\t9990\t10000\t60\n",
        encoding="utf-8",
    )
    reverse_path.write_text(
        "f2\t11000\t0\t10000\t+\tf0\t11000\t0\t10000\t9950\t10000\t60\n",
        encoding="utf-8",
    )
    forward = read_best_alignments(forward_path, minimum_coverage=0.8, minimum_mapq=20)
    reverse = read_best_alignments(reverse_path, minimum_coverage=0.8, minimum_mapq=20)
    mutual = mutual_best_alignments(forward, reverse)

    assert forward[["query_index", "target_index"]].values.tolist() == [[0, 2]]
    assert mutual[["query_index", "target_index"]].values.tolist() == [[0, 2]]


def test_fragment_helpers_do_not_require_edlib_until_distance_runtime(
    monkeypatch,
):
    def missing_dependency(name):
        assert name == "edlib"
        raise ImportError(name)

    monkeypatch.setattr(fragment_analysis.importlib, "import_module", missing_dependency)
    with pytest.raises(RuntimeError, match="requires edlib>=1.3.9"):
        fragment_analysis.load_edlib()


def test_pair_manifest_normalizes_canonical_and_legacy_inputs():
    canonical = pd.DataFrame(
        [{"genome_a": "1", "genome_b": "2", "species": "same", "ani": 99.5}]
    )
    legacy = pd.DataFrame(
        [
            {
                "target_id": "2",
                "donor_id": "1",
                "species": "same",
                "ani": 99.5,
                "fold": 0,
            },
            {
                "target_id": "1",
                "donor_id": "2",
                "species": "same",
                "ani": 99.5,
                "fold": 1,
            },
        ]
    )

    assert normalize_pair_manifest(canonical)[
        ["genome_a", "genome_b"]
    ].values.tolist() == [["1", "2"]]
    assert normalize_pair_manifest(legacy)[
        ["genome_a", "genome_b"]
    ].values.tolist() == [["1", "2"]]


def test_fragment_figure_data_excludes_identical_and_high_divergence_rows():
    frame = pd.DataFrame(
        [
            {
                "pair_id": "a|b",
                "species": "same",
                "whole_genome_ani": 99.9,
                "fragment_a_index": 0,
                "fragment_b_index": 1,
                "global_sequence_divergence": 0.0,
                "cosine_distance": -1e-15,
            },
            {
                "pair_id": "a|b",
                "species": "same",
                "whole_genome_ani": 99.9,
                "fragment_a_index": 1,
                "fragment_b_index": 2,
                "global_sequence_divergence": 0.01,
                "cosine_distance": 2e-6,
            },
            {
                "pair_id": "a|b",
                "species": "same",
                "whole_genome_ani": 99.9,
                "fragment_a_index": 2,
                "fragment_b_index": 3,
                "global_sequence_divergence": 0.06,
                "cosine_distance": 4e-6,
            },
        ]
    )

    plotted = prepare_fragment_plot_data(frame, maximum_divergence=0.05)

    assert len(plotted) == 1
    assert plotted["sequence_divergence_percent"].tolist() == [1.0]
    assert plotted["cosine_distance"].tolist() == [2e-6]
    assert "divergence_bin" not in plotted.columns


def test_fragment_figure_log_scale_starts_below_data_and_distinguishes_ani_subset():
    fragments = pd.DataFrame(
        {
            "sequence_divergence_percent": [0.1, 1.0, 4.0],
            "cosine_distance": [6e-8, 2e-6, 3e-3],
            "whole_genome_ani": [98.5, 99.1, 99.9],
        }
    )
    summary = {
        "results": {
            "all_pairs": {
                "pooled_spearman_sequence_divergence_vs_cosine_distance": 0.695
            },
            "whole_genome_ani_ge_99": {
                "pooled_spearman_sequence_divergence_vs_cosine_distance": 0.714
            },
        }
    }
    figure, axis = plt.subplots()
    plot_fragment_panel(axis, fragments, summary)

    assert axis.get_yscale() == "log"
    assert axis.get_ylim()[0] == 1e-8
    assert axis.get_ylim()[1] > fragments["cosine_distance"].max()
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "95% ≤ ANI <99% ($n$=1)",
        "ANI ≥99% ($n$=2)",
    ]
    assert "logarithmic" in build_caption()
    plt.close(figure)


def test_svg_normalization_removes_only_line_end_whitespace(tmp_path):
    path = tmp_path / "figure.svg"
    path.write_text("<svg>  \n  <path d='M 0 0 ' />\t\n</svg>\n", encoding="utf-8")

    normalize_svg_whitespace(path)

    assert path.read_text(encoding="utf-8") == "<svg>\n  <path d='M 0 0 ' />\n</svg>\n"


def test_conservative_annotation_dictionary_separates_amr_and_mobile_elements():
    amr, mge, matches = classify_annotation(
        "CDS", {"gene": ["mecA"], "product": ["penicillin-binding protein 2a"]}
    )
    assert amr is True
    assert mge is False
    assert "amr_gene:mecA" in matches

    amr, mge, matches = classify_annotation(
        "mobile_element", {"product": ["IS3 family transposase"]}
    )
    assert amr is False
    assert mge is True
    assert "mge_feature:mobile_element" in matches

    amr, mge, matches = classify_annotation(
        "CDS", {"product": ["DNA mismatch repair protein"]}
    )
    assert (amr, mge, matches) == (False, False, [])


def test_nearest_donor_is_same_species_heldout_and_deterministic():
    raw = pd.DataFrame(
        [
            [
                "/x/Escherichia_coli_ATCC_1.fasta",
                "/x/Escherichia_coli_ATCC_2.fasta",
                99.1,
                90,
                91,
            ],
            [
                "/x/Escherichia_coli_ATCC_1.fasta",
                "/x/Escherichia_coli_ATCC_3.fasta",
                99.1,
                95,
                94,
            ],
            [
                "/x/Escherichia_coli_ATCC_1.fasta",
                "/x/Staphylococcus_aureus_ATCC_4.fasta",
                99.9,
                99,
                99,
            ],
        ],
        columns=[
            "Ref_file",
            "Query_file",
            "ANI",
            "Align_fraction_ref",
            "Align_fraction_query",
        ],
    )
    pairs = parse_skani_sparse(raw)
    donors = select_nearest_same_species_neighbors(
        pairs,
        eligible_ids={"1", "2", "3", "4"},
        species_by_id={
            "1": "Escherichia coli",
            "2": "Escherichia coli",
            "3": "Escherichia coli",
            "4": "Staphylococcus aureus",
        },
    )
    selected = donors.set_index("target_id")["donor_id"].to_dict()
    assert selected["1"] == "3"
    assert selected["2"] == "1"
    assert selected["3"] == "1"
    assert "4" not in selected


def test_donor_threshold_excludes_low_ani_or_low_bidirectional_coverage():
    pairs = pd.DataFrame(
        [
            {
                "ref_id": "a",
                "query_id": "b",
                "ANI": 94.9,
                "Align_fraction_ref": 90,
                "Align_fraction_query": 90,
            },
            {
                "ref_id": "a",
                "query_id": "c",
                "ANI": 99.0,
                "Align_fraction_ref": 90,
                "Align_fraction_query": 49.9,
            },
        ]
    )
    donors = select_nearest_same_species_neighbors(
        pairs,
        eligible_ids={"a", "b", "c"},
        species_by_id={"a": "same", "b": "same", "c": "same"},
    )
    assert donors.empty


def test_frozen_non_bacterial_scope_contains_only_the_known_fungal_species():
    assert NON_BACTERIAL_SPECIES == {
        "Aspergillus fumigatus",
        "Candida albicans",
        "Candida krusei",
        "Candida tropicalis",
        "Cryptococcus neoformans",
        "Saccharomyces cerevisiae",
    }


def test_manifest_boolean_parser_rejects_ambiguous_values():
    parsed = parse_boolean_series(pd.Series(["True", "false", True]))
    assert parsed.tolist() == [True, False, True]

    try:
        parse_boolean_series(pd.Series(["yes"]))
    except ValueError as error:
        assert "Unexpected boolean values" in str(error)
    else:
        raise AssertionError("Ambiguous manifest boolean should be rejected")


def test_probe_cohort_is_deterministic_and_excludes_negative_only_genomes():
    labels = pd.DataFrame(
        {
            "genome_id": ["a"] * 8 + ["b"] * 4,
            "fragment_index": list(range(8)) + list(range(4)),
            "amr_associated": [True, False, False, False, False, False, False, False]
            + [False] * 4,
        }
    )
    config = ProbeConfig(
        label_column="amr_associated",
        display_name="AMR",
        negative_ratio=3,
        seed=7,
    )

    first = deterministic_probe_cohort(labels, config=config)
    second = deterministic_probe_cohort(labels, config=config)

    pd.testing.assert_frame_equal(first, second)
    assert first["genome_id"].unique().tolist() == ["a"]
    assert len(first) == 4
    assert int(first["amr_associated"].sum()) == 1
