from __future__ import annotations

import torch
import torch.nn as nn

from apexoracle.data.synergy_dataset import collate_synergy_genome_text
from apexoracle.models.strain_fusion import RegressionHead
from apexoracle.models.strain_fusion import FirstTokenAttentionGenome
from apexoracle.models.synergy_checkpoint import (
    build_legacy_synergy_components,
    load_legacy_synergy_member,
)
from apexoracle.training.synergy import (
    legacy_synergy_checkpoint_payload,
    symmetric_pair_logits,
    synergy_pair_forward,
)


def _batch():
    items = []
    for offset, label in ((0, 1.0), (1, 0.0)):
        items.append(
            {
                "label": torch.tensor(label),
                "genome_embedding": torch.arange(
                    16, dtype=torch.bfloat16
                ).reshape(2, 8),
                "text_embedding": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
                "strain_name": f"s{offset}",
                "pair_key": (f"a{offset}", f"b{offset}", f"s{offset}"),
                "mol_emb_1": torch.arange(6, dtype=torch.float32) + offset,
                "mol_emb_2": torch.arange(6, dtype=torch.float32) + 10 + offset,
            }
        )
    return collate_synergy_genome_text(items)


def test_symmetric_pair_logits_are_invariant_to_pair_order() -> None:
    torch.manual_seed(3)
    head = RegressionHead(24, 8, 4, 1, 0.0).eval()
    fused = torch.randn(4, 12)
    forward = symmetric_pair_logits(fused, head)
    reversed_pairs = fused.reshape(2, 2, 12).flip(1).reshape(4, 12)
    reverse = symmetric_pair_logits(reversed_pairs, head)
    torch.testing.assert_close(forward, reverse, rtol=0, atol=0)


def test_synergy_forward_preserves_pair_shapes_and_loss() -> None:
    class _Attention(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.projection = nn.Linear(6, output_dim)

        def forward(self, mol_cls_emb, genome_embs, key_padding_mask):
            del genome_embs, key_padding_mask
            return self.projection(mol_cls_emb)

    torch.manual_seed(5)
    batch = _batch()
    genome = _Attention(8).eval()
    text = _Attention(4).eval()
    head = RegressionHead(24, 8, 4, 1, 0.0).eval()
    result = synergy_pair_forward(
        batch,
        device=torch.device("cpu"),
        genome_attention=genome,
        text_attention=text,
        prediction_head=head,
        criterion=nn.BCEWithLogitsLoss(),
        missing_genome_embedding=nn.Parameter(torch.zeros(1, 8), requires_grad=False),
        has_genome=True,
        autocast_enabled=False,
    )
    assert result.logits.shape == (2, 1)
    assert result.labels.tolist() == [1.0, 0.0]
    expected = nn.BCEWithLogitsLoss()(result.logits.squeeze(), result.labels)
    torch.testing.assert_close(result.loss, expected, rtol=0, atol=0)


def test_synergy_checkpoint_keeps_only_lora_attention_weights() -> None:
    class _Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(2, 2)
            self.lora_A = nn.Linear(2, 1, bias=False)

    genome = _Adapter()
    text = _Adapter()
    head = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(
        list(genome.parameters()) + list(text.parameters()) + list(head.parameters())
    )
    payload = legacy_synergy_checkpoint_payload(
        auroc=0.7,
        optimizer=optimizer,
        prediction_head=head,
        genome_attention=genome,
        text_attention=text,
        missing_genome_embedding=nn.Parameter(torch.zeros(1, 2), requires_grad=False),
    )
    assert set(payload) == {
        "AUROC",
        "optimizer_state_dict",
        "re_head_state_dict",
        "co_cross_attn_genome",
        "co_cross_attn_text",
        "learnable_embedding_weight",
    }
    assert set(payload["co_cross_attn_genome"]) == {"lora_A.weight"}


def test_base_and_member_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(17)
    genome = FirstTokenAttentionGenome(6, 8, 2, 0.0)
    text = FirstTokenAttentionGenome(6, 4, 2, 0.0)
    base_path = tmp_path / "base.pth"
    torch.save(
        {
            "co_cross_attn_genome": genome.state_dict(),
            "co_cross_attn_text": text.state_dict(),
            "learnable_embedding_weight": torch.randn(1, 8),
        },
        base_path,
    )
    components = build_legacy_synergy_components(
        base_path,
        device=torch.device("cpu"),
        molecule_dim=6,
        genome_dim=8,
        text_dim=4,
        attention_heads=2,
        lora_rank=2,
    )
    optimizer = torch.optim.Adam(
        [
            parameter
            for module in (
                components.genome_attention,
                components.text_attention,
                components.prediction_head,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
    )
    member_path = tmp_path / "member.ckpt"
    torch.save(
        legacy_synergy_checkpoint_payload(
            auroc=0.75,
            optimizer=optimizer,
            prediction_head=components.prediction_head,
            genome_attention=components.genome_attention,
            text_attention=components.text_attention,
            missing_genome_embedding=components.missing_genome_embedding,
        ),
        member_path,
    )
    expected = {
        key: value.clone() for key, value in components.prediction_head.state_dict().items()
    }
    with torch.no_grad():
        for parameter in components.prediction_head.parameters():
            parameter.zero_()
    assert load_legacy_synergy_member(member_path, components=components) == 0.75
    for key, value in components.prediction_head.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
