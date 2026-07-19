from dataclasses import replace
import gc
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

from apexoracle.benchmarks.molecule_encoders.encoders import HF_ENCODERS
from apexoracle.data.hierarchical_mic import (
    OnlineMoleculeStrainDataset,
    collate_online_genome_text_regression,
)
from apexoracle.models.hf_molecule_encoder import (
    load_legacy_molecule_encoder,
    load_legacy_tokenizer,
    unfreeze_legacy_molecule_encoder,
)
from apexoracle.models.hierarchical_mic_checkpoint import inspect_checkpoint_contract
from apexoracle.models.strain_fusion import FirstTokenAttentionGenome, RegressionHead
from apexoracle.training.hierarchical_mic import (
    hierarchical_mic_batch_forward,
    legacy_hierarchical_checkpoint_payload,
)
from apexoracle.training import hierarchical_mic_runner
from apexoracle.training.hierarchical_mic_runner import (
    HierarchicalMicConfig,
    build_model_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/hierarchical_mic/legacy_fig2c_comparators.yaml"


class TinyTokenizer:
    pad_token_id = 9

    def __call__(self, text, **_kwargs):
        ids = torch.tensor([[1] + [ord(char) % 7 + 2 for char in text] + [2]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 3)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyAttention(nn.Module):
    def forward(self, mol_cls_emb, genome_embs, key_padding_mask):
        del key_padding_mask
        return mol_cls_emb + genome_embs[:, 0, :]


def test_comparator_profiles_freeze_exact_legacy_behavior():
    expected = {
        "chemberta_mtr": (384, "eval", 3000, 1),
        "chemberta_mlm": (384, "train", 3, 7),
        "molformer": (768, "train", 3, 7),
        "peptideclm": (768, "eval", 5000, 1),
    }
    for name, values in expected.items():
        config = HierarchicalMicConfig.load(
            CONFIG,
            REPO_ROOT,
            holdout_protocol="strain",
            molecule_encoder_name=name,
        )
        assert config.molecule_encoder is not None
        assert (
            config.molecule_embedding_dim,
            config.molecule_encoder.initial_mode,
            config.freeze_epochs,
            config.ensembles_per_group,
        ) == values
        assert config.batch_size == 70
        assert config.molecule_encoder.pooling == "first_token"
        assert config.molecule_encoder.optimizer_learning_rate == 3e-6
        assert config.molecule_encoder.checkpoint_state_key == "ChemBERTa_state_dict"


def test_comparator_revisions_match_canonical_weight_manifest():
    with (REPO_ROOT / "configs/model_weights.yaml").open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)["weights"]
    for name in ("chemberta_mtr", "chemberta_mlm", "molformer", "peptideclm"):
        config = HierarchicalMicConfig.load(
            CONFIG,
            REPO_ROOT,
            holdout_protocol="strain",
            molecule_encoder_name=name,
        )
        revision = manifest[f"fig2b_{name}"]["source"]["revision"]
        assert config.molecule_encoder.revision == revision
        assert HF_ENCODERS[name].revision == revision


def test_online_dataset_and_collate_preserve_raw_smiles_tokenization():
    frame = pd.DataFrame(
        [
            [1, "strain", "CC", 10.0],
            [2, "strain", "CCC", 1.0],
        ],
        columns=["DBAASP_id", "strain_name", "SMILES", "MIC"],
    )
    genome = {"strain": torch.ones(2, 3, dtype=torch.bfloat16)}
    text = {"strain": torch.ones(3, 3, dtype=torch.bfloat16)}
    dataset = OnlineMoleculeStrainDataset(
        frame,
        TinyTokenizer(),
        genome,
        text,
        "fixture",
    )
    batch = collate_online_genome_text_regression(
        [dataset[0], dataset[1]],
        pad_token_id=TinyTokenizer.pad_token_id,
    )
    assert batch["input_ids"].tolist() == [[1, 6, 6, 2, 9], [1, 6, 6, 6, 2]]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ]
    torch.testing.assert_close(batch["label"], torch.tensor([-0.0, 1.0]))
    assert batch["padded_genome_embeddings"].dtype == torch.bfloat16
    assert batch["genome_attn_masks"].dtype == torch.uint8


def test_online_first_token_forward_matches_inline_legacy_formula():
    torch.manual_seed(12)
    encoder = TinyEncoder()
    genome_attention = TinyAttention()
    text_attention = TinyAttention()
    head = nn.Linear(6, 1)
    batch = {
        "input_ids": torch.tensor([[1, 3], [1, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "label": torch.tensor([0.25, -0.5]),
        "padded_genome_embeddings": torch.randn(2, 1, 3),
        "genome_attn_masks": torch.ones(2, 1, dtype=torch.uint8),
        "padded_text_embeddings": torch.randn(2, 1, 3),
        "text_attn_masks": torch.ones(2, 1, dtype=torch.uint8),
        "strain_names": ["a", "b"],
    }
    result = hierarchical_mic_batch_forward(
        batch,
        device=torch.device("cpu"),
        molecule_encoder=encoder,
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=head,
        criterion=nn.MSELoss(),
        missing_genome_embedding=nn.Parameter(torch.zeros(1, 3)),
        has_genome=True,
        reshape_outputs=True,
        autocast_enabled=False,
    )
    molecule = encoder(
        batch["input_ids"], batch["attention_mask"]
    ).last_hidden_state[:, 0, :]
    fused = torch.cat(
        (
            molecule + batch["padded_genome_embeddings"][:, 0, :],
            molecule + batch["padded_text_embeddings"][:, 0, :],
        ),
        dim=1,
    )
    expected_logits = head(fused).squeeze()
    torch.testing.assert_close(result.logits, expected_logits)
    torch.testing.assert_close(
        result.loss, nn.MSELoss()(expected_logits, batch["label"])
    )


def test_online_checkpoint_adds_legacy_misnamed_backbone_key():
    encoder = TinyEncoder()
    head = nn.Linear(2, 1)
    attention = nn.Linear(2, 2)
    missing = nn.Parameter(torch.zeros(1, 2))
    optimizer = torch.optim.Adam(
        list(head.parameters()) + list(attention.parameters()) + [missing]
    )
    payload = legacy_hierarchical_checkpoint_payload(
        r2=0.5,
        optimizer=optimizer,
        regression_head=head,
        classification_head=head,
        genome_attention=attention,
        text_attention=attention,
        missing_genome_embedding=missing,
        molecule_encoder=encoder,
        molecule_encoder_state_key="ChemBERTa_state_dict",
    )
    assert set(payload) == {
        "R2",
        "optimizer_state_dict",
        "ChemBERTa_state_dict",
        "re_head_state_dict",
        "cls_head_state_dict",
        "co_cross_attn_genome",
        "co_cross_attn_text",
        "learnable_embedding_weight",
    }
    assert payload["ChemBERTa_state_dict"].keys() == encoder.state_dict().keys()


def test_unfreeze_changes_grad_flags_without_changing_module_mode():
    encoder = TinyEncoder().eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    unfreeze_legacy_molecule_encoder(encoder)
    assert not encoder.training
    assert all(parameter.requires_grad for parameter in encoder.parameters())


def test_online_optimizer_groups_preserve_legacy_order(monkeypatch):
    config = HierarchicalMicConfig.load(
        CONFIG,
        REPO_ROOT,
        holdout_protocol="strain",
        molecule_encoder_name="chemberta_mlm",
        epochs=2,
    )
    config = replace(
        config,
        molecule_embedding_dim=3,
        genome_embedding_dim=8,
        text_embedding_dim=4,
        head_hidden_dims=(3, 128),
        molecule_encoder=replace(config.molecule_encoder, hidden_size=3),
    )
    encoder = TinyEncoder()
    monkeypatch.setattr(
        hierarchical_mic_runner,
        "load_legacy_molecule_encoder",
        lambda *_args, **_kwargs: encoder,
    )
    bundle = build_model_bundle(
        config,
        genome_dim=8,
        text_dim=4,
        device=torch.device("cpu"),
    )
    expected_modules = [
        bundle.genome_attention,
        bundle.text_attention,
        bundle.regression_head,
        bundle.classification_head,
    ]
    for group, module in zip(bundle.optimizer.param_groups[:4], expected_modules):
        assert {id(parameter) for parameter in group["params"]} == {
            id(parameter) for parameter in module.parameters()
        }
    assert len(bundle.optimizer.param_groups[4]["params"]) == 1
    assert (
        bundle.optimizer.param_groups[4]["params"][0]
        is bundle.missing_genome_embedding
    )
    assert {id(parameter) for parameter in bundle.optimizer.param_groups[5]["params"]} == {
        id(parameter) for parameter in encoder.parameters()
    }
    assert [group["lr"] for group in bundle.optimizer.param_groups] == [
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        3e-6,
    ]


def test_checkpoint_inspection_recognizes_online_huggingface_payload():
    genome_attention = FirstTokenAttentionGenome(6, 8, 4, 0.1)
    text_attention = FirstTokenAttentionGenome(6, 4, 4, 0.1)
    head = RegressionHead(12, 3, 2, 1, 0.2)
    checkpoint = {
        "R2": 0.25,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "re_head_state_dict": head.state_dict(),
        "cls_head_state_dict": head.state_dict(),
        "co_cross_attn_genome": genome_attention.state_dict(),
        "co_cross_attn_text": text_attention.state_dict(),
        "learnable_embedding_weight": torch.randn(1, 8),
        "ChemBERTa_state_dict": {
            "embeddings.word_embeddings.weight": torch.randn(600, 6)
        },
    }
    payload = inspect_checkpoint_contract(checkpoint)["optional_payloads"][
        "ChemBERTa_state_dict"
    ]
    assert payload == {
        "interpretation": "online_huggingface_molecule_encoder_state_dict",
        "key_count": 1,
        "vocab_embedding_shape": [600, 6],
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pinned_comparator_backbones_cuda_smoke():
    device = torch.device("cuda:0")
    for name in ("chemberta_mtr", "chemberta_mlm", "molformer", "peptideclm"):
        config = HierarchicalMicConfig.load(
            CONFIG,
            REPO_ROOT,
            holdout_protocol="strain",
            molecule_encoder_name=name,
        )
        profile = config.molecule_encoder
        tokenizer = load_legacy_tokenizer(profile, REPO_ROOT)
        encoder = load_legacy_molecule_encoder(profile, device=device)
        encoded = tokenizer(
            ["CCO", "NCC(=O)O"],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state[:, 0, :]
        assert output.shape == (2, profile.hidden_size)
        assert torch.isfinite(output).all()
        assert encoder.training == (profile.initial_mode == "train")
        assert not any(parameter.requires_grad for parameter in encoder.parameters())
        if config.freeze_epochs == 3:
            unfreeze_legacy_molecule_encoder(encoder)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state[:, 0, :].float().square().mean()
            loss.backward()
            assert any(parameter.grad is not None for parameter in encoder.parameters())
        del encoder, tokenizer, encoded, input_ids, attention_mask, output
        gc.collect()
        torch.cuda.empty_cache()
