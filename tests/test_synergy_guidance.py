from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from apexoracle.data.synergy import (
    PreparedSynergyData,
    build_legacy_synergy_all_data_routes,
)
from apexoracle.data.synergy_dataset import (
    TokenizedSynergyPairDataset,
    collate_tokenized_synergy_genome_text,
    collate_tokenized_synergy_text_only,
)
from apexoracle.models.synergy_checkpoint import (
    inspect_synergy_guidance_checkpoint,
)
from apexoracle.training.synergy import (
    legacy_synergy_guidance_checkpoint_payload,
    synergy_guidance_pair_forward,
    synergy_guidance_pair_step,
)
from apexoracle.training.synergy_guidance_runner import GuidanceConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/synergy/legacy_guidance.yaml"


class TinyTokenizer:
    pad_token_id = 3
    mask_token_id = 4

    def __call__(self, text, **_kwargs):
        values = torch.tensor([[1] + [ord(char) % 5 + 5 for char in text] + [2]])
        return {"input_ids": values, "attention_mask": torch.ones_like(values)}


class TinyPairEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 3)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def encode_pairs(self, input_ids):
        first = self.embedding(input_ids[::2, 0])
        second = self.embedding(input_ids[1::2, 0])
        output = torch.empty(input_ids.shape[0], 3, device=input_ids.device)
        output[::2] = first
        output[1::2] = second
        return output


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False)

    def forward(self, mol_cls_emb, genome_embs, key_padding_mask):
        del key_padding_mask
        return self.projection(mol_cls_emb) + genome_embs[:, 0, :]


class RecordingAttention(TinyAttention):
    def __init__(self, name, calls):
        super().__init__()
        self.name = name
        self.calls = calls

    def forward(self, mol_cls_emb, genome_embs, key_padding_mask):
        self.calls.append(self.name)
        return super().forward(mol_cls_emb, genome_embs, key_padding_mask)


def _table():
    return pd.DataFrame(
        [
            [1, "drug-a", "strain", "CC", "CO", 0.25],
            [2, "drug-b", "strain", "CCC", "CN", 0.75],
        ],
        columns=(
            "DBAASP_id",
            "antibio_id_or_name",
            "strain_name",
            "AMP_smiles",
            "antibiotic_smiles",
            "FICI",
        ),
    )


def _batch(*, has_genome=True):
    batch = {
        "input_ids": torch.tensor(
            [[1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        "label": torch.tensor([1.0, 0.0]),
        "padded_text_embeddings": torch.randn(4, 1, 3),
        "text_attn_masks": torch.ones(4, 1, dtype=torch.uint8),
        "strain_names": ["a", "b"],
        "pair_keys": [(1, 2, "a"), (3, 4, "b")],
    }
    if has_genome:
        batch.update(
            {
                "padded_genome_embeddings": torch.randn(4, 1, 3),
                "genome_attn_masks": torch.ones(4, 1, dtype=torch.uint8),
            }
        )
    return batch


def test_guidance_profiles_keep_observed_run_boundaries():
    short = GuidanceConfig.load(
        CONFIG, REPO_ROOT, profile="short_judger"
    )
    long = GuidanceConfig.load(
        CONFIG, REPO_ROOT, profile="guidance_40epoch"
    )
    assert short.epochs == 2
    assert long.epochs == 40
    assert short.lora_rank == long.lora_rank == 64
    assert short.paths.output_dir != short.paths.observed_output
    assert long.paths.output_dir != long.paths.observed_output
    assert short.model_fixed_length == 1024


def test_tokenized_guidance_dataset_and_collates_preserve_pair_order_and_padding():
    tokenizer = TinyTokenizer()
    genome = {"strain": torch.ones(2, 3, dtype=torch.bfloat16)}
    text = {"strain": torch.ones(3, 3, dtype=torch.bfloat16)}
    dataset = TokenizedSynergyPairDataset(
        _table(),
        tokenizer=tokenizer,
        selfies_encoder=lambda value: value,
        genome_embeddings=genome,
        text_embeddings=text,
    )
    batch = collate_tokenized_synergy_genome_text(
        [dataset[0], dataset[1]],
        pad_token_id=tokenizer.pad_token_id,
        fixed_length=8,
    )
    assert batch["input_ids"].shape == (4, 8)
    assert batch["input_ids"][:, 0].tolist() == [1, 1, 1, 1]
    assert batch["label"].tolist() == [1.0, 0.0]
    assert batch["pair_keys"] == [
        (1, "drug-a", "strain"),
        (2, "drug-b", "strain"),
    ]
    assert batch["padded_genome_embeddings"].shape[0] == 4
    text_dataset = TokenizedSynergyPairDataset(
        _table(),
        tokenizer=tokenizer,
        selfies_encoder=lambda value: value,
        text_embeddings=text,
    )
    text_batch = collate_tokenized_synergy_text_only(
        [text_dataset[0]],
        pad_token_id=tokenizer.pad_token_id,
        fixed_length=8,
    )
    assert text_batch["input_ids"].shape == (2, 8)
    assert "padded_genome_embeddings" not in text_batch


def test_guidance_forward_matches_separate_first_second_legacy_formula():
    torch.manual_seed(7)
    encoder = TinyPairEncoder()
    genome_attention = TinyAttention()
    text_attention = TinyAttention()
    head = nn.Linear(12, 1)
    batch = _batch()
    result = synergy_guidance_pair_forward(
        batch,
        device=torch.device("cpu"),
        molecule_encoder=encoder,
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=head,
        criterion=nn.BCEWithLogitsLoss(),
        missing_genome_embedding=nn.Parameter(torch.zeros(1, 3)),
        has_genome=True,
        autocast_enabled=False,
    )
    molecules = encoder.encode_pairs(batch["input_ids"])
    fused = []
    for offset in (0, 1):
        fused.append(
            torch.cat(
                (
                    genome_attention(
                        molecules[offset::2],
                        batch["padded_genome_embeddings"][offset::2],
                        1 - batch["genome_attn_masks"][offset::2],
                    ),
                    text_attention(
                        molecules[offset::2],
                        batch["padded_text_embeddings"][offset::2],
                        1 - batch["text_attn_masks"][offset::2],
                    ),
                ),
                dim=1,
            )
        )
    expected = (head(torch.cat(fused, dim=1)) + head(torch.cat(fused[::-1], dim=1))) / 2
    torch.testing.assert_close(result.logits, expected)
    torch.testing.assert_close(
        result.loss,
        nn.BCEWithLogitsLoss()(expected.squeeze(), batch["label"]),
    )


@pytest.mark.parametrize(
    ("has_genome", "expected_calls"),
    [
        (True, ["genome", "text", "genome", "text"]),
        (False, ["genome", "genome", "text", "text"]),
    ],
)
def test_guidance_forward_preserves_route_specific_attention_order(
    has_genome, expected_calls
):
    calls = []
    synergy_guidance_pair_forward(
        _batch(has_genome=has_genome),
        device=torch.device("cpu"),
        molecule_encoder=TinyPairEncoder(),
        genome_attention=RecordingAttention("genome", calls),
        text_attention=RecordingAttention("text", calls),
        prediction_head=nn.Linear(12, 1),
        criterion=nn.BCEWithLogitsLoss(),
        missing_genome_embedding=nn.Parameter(torch.zeros(1, 3)),
        has_genome=has_genome,
        autocast_enabled=False,
    )
    assert calls == expected_calls


def test_guidance_step_preserves_dead_legacy_cpu_rng_draw():
    encoder = TinyPairEncoder()
    genome_attention = TinyAttention()
    text_attention = TinyAttention()
    head = nn.Linear(12, 1)
    optimizer = torch.optim.Adam(
        list(genome_attention.parameters())
        + list(text_attention.parameters())
        + list(head.parameters()),
        lr=1e-3,
    )
    batch = _batch()
    seed = 613
    torch.manual_seed(seed)
    torch.randn(1)
    expected_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    synergy_guidance_pair_step(
        batch,
        device=torch.device("cpu"),
        molecule_encoder=encoder,
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=head,
        criterion=nn.BCEWithLogitsLoss(),
        missing_genome_embedding=nn.Parameter(torch.zeros(1, 3)),
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        has_genome=True,
        autocast_enabled=False,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), expected_state)


def test_all_data_routes_preserve_alias_merge_and_complete_row_sets():
    genome = _table().copy()
    genome["strain_name"] = ["g1", "g2"]
    text = _table().iloc[[0]].copy()
    text["strain_name"] = "t1"
    combined = pd.concat((genome, text), ignore_index=True)
    prepared = PreparedSynergyData(
        genome_text=genome,
        text_only=text,
        combined=combined,
        standard_strain_groups={
            name: genome.loc[genome["strain_name"] == name].copy()
            for name in ("g1", "g2")
        },
        all_strain_groups={
            name: combined.loc[combined["strain_name"] == name].copy()
            for name in ("g1", "g2", "t1")
        },
        species_to_strains={"species-old": ["g1"], "species-new": ["g2", "t1"]},
        taxonomy_aliases={"species-old": "species-new", "species-new": "species-old"},
    )
    routes = build_legacy_synergy_all_data_routes(prepared)
    assert set(routes.genome_text["strain_name"]) == {"g1", "g2"}
    assert set(routes.combined_text["strain_name"]) == {"g1", "g2", "t1"}
    assert set(routes.strain_order) == {"g1", "g2", "t1"}


def test_guidance_checkpoint_payload_and_contract_keep_full_states():
    encoder = TinyPairEncoder()
    genome_attention = TinyAttention()
    text_attention = TinyAttention()
    head = nn.Sequential()
    head.dense_1 = nn.Linear(12, 3)
    head.dense_2 = nn.Linear(3, 2)
    head.out_proj = nn.Linear(2, 1)
    genome_state = genome_attention.state_dict()
    genome_state["projection.lora_A.default.weight"] = torch.zeros(64, 3)
    text_state = text_attention.state_dict()
    text_state["projection.lora_A.default.weight"] = torch.zeros(64, 3)
    genome_attention.state_dict = lambda: genome_state
    text_attention.state_dict = lambda: text_state
    optimizer = torch.optim.Adam(head.parameters())
    payload = legacy_synergy_guidance_checkpoint_payload(
        auroc=0.8,
        optimizer=optimizer,
        molecule_encoder=encoder,
        prediction_head=head,
        genome_attention=genome_attention,
        text_attention=text_attention,
        missing_genome_embedding=nn.Parameter(torch.zeros(1, 3)),
    )
    contract = inspect_synergy_guidance_checkpoint(payload)
    assert contract["auroc"] == 0.8
    assert contract["fusion_lora_rank"] == 64
    assert contract["head_dimensions"] == [12, 3, 2, 1]
    assert set(payload) == {
        "AUROC",
        "optimizer_state_dict",
        "mdlm_model_state_dict",
        "re_head_state_dict",
        "co_cross_attn_genome",
        "co_cross_attn_text",
        "learnable_embedding_weight",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_guidance_optimizer_step_cuda_smoke():
    device = torch.device("cuda:0")
    encoder = TinyPairEncoder().to(device).eval()
    genome_attention = TinyAttention().to(device)
    text_attention = TinyAttention().to(device)
    head = nn.Linear(12, 1).to(device)
    missing = nn.Parameter(torch.zeros(1, 3, device=device), requires_grad=False)
    optimizer = torch.optim.Adam(
        list(genome_attention.parameters())
        + list(text_attention.parameters())
        + list(head.parameters()),
        lr=1e-3,
    )
    scaler = torch.amp.GradScaler("cuda")
    batch = {
        "input_ids": torch.tensor(
            [[1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        "label": torch.tensor([1.0, 0.0]),
        "padded_genome_embeddings": torch.randn(
            4, 1, 3, dtype=torch.bfloat16, device=device
        ),
        "genome_attn_masks": torch.ones(4, 1, dtype=torch.uint8, device=device),
        "padded_text_embeddings": torch.randn(
            4, 1, 3, dtype=torch.bfloat16, device=device
        ),
        "text_attn_masks": torch.ones(4, 1, dtype=torch.uint8, device=device),
        "strain_names": ["a", "b"],
        "pair_keys": [(1, 2, "a"), (3, 4, "b")],
    }
    before = [parameter.detach().clone() for parameter in head.parameters()]
    result = synergy_guidance_pair_step(
        batch,
        device=device,
        molecule_encoder=encoder,
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=head,
        criterion=nn.BCEWithLogitsLoss(),
        missing_genome_embedding=missing,
        optimizer=optimizer,
        scaler=scaler,
        has_genome=True,
        autocast_enabled=True,
    )
    assert torch.isfinite(result.loss)
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, head.parameters())
    )
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
