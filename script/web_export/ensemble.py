"""The model that ships to the browser.

The paper's deployed artefact is a single model that has never seen the query's
disease. No such checkpoint exists here: what exists is 13 leave-one-disease-out
folds, each trained on 12 of the 13 diseases. Averaging their logits is the
closest available stand-in, and it is what the site runs. The honest
generalisation number stays the per-fold LODO AUC, which the export measures
separately.

Two things make bundling 13 folds practical.

The embedding table is **identical in all 416 checkpoints** -- it is loaded from
a pretrained file and frozen (`otu_attention` sets ``requires_grad = False``),
so training never touched it. One copy therefore serves every fold, and it is
shipped as a separate float16 blob that the browser gathers from directly. Only
the 13 encoders, about 100 KB each, go into the ONNX graph.

`OtuAttentionEncoder.forward` is reused unchanged. Its ``embedding`` attribute
is swapped for :class:`torch.nn.Identity` so the graph takes already-gathered
vectors instead of token indices; every other line of the forward -- the
attention mask, the abundance scaling, the pooling, the head -- is the original
code path. :func:`check_equivalence` pins that down against the untouched
module.
"""

import torch
from torch import nn


class FoldEnsemble(nn.Module):
    """Average the logits of several `OtuAttentionEncoder` folds.

    Parameters
    ----------
    nets : sequence of OtuAttentionEncoder
        One encoder per fold, each with its ``embedding`` replaced by
        :class:`torch.nn.Identity`. The caller owns that substitution because
        it is what lets the folds share one embedding table.

    Attributes
    ----------
    nets : torch.nn.ModuleList
        The wrapped encoders.

    Notes
    -----
    The attention output is reduced the same way for every fold and then
    averaged: over heads, then over query positions, leaving one weight per
    *key*. A sequence position's importance is how much attention the sample as
    a whole paid it, not how much one arbitrary query did.
    """

    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, inputs, weight, mask):
        """Score a batch through every fold and average.

        Parameters
        ----------
        inputs : torch.Tensor
            Gathered OTU embeddings of shape ``(batch, seq_len, d_model)``.
        weight : torch.Tensor
            Rank-normalized abundances of shape ``(batch, seq_len)``.
        mask : torch.Tensor
            Attention mask of shape ``(batch, seq_len)``, 0 at padding and at
            positions outside the vocabulary.

        Returns
        -------
        logits : torch.Tensor
            Fold-averaged logits of shape ``(batch, 1)``.
        attention : torch.Tensor
            Fold-, head- and query-averaged attention over the keys, shape
            ``(batch, seq_len)``.
        """
        logits = []
        attentions = []
        for net in self.nets:
            out, attn = net(inputs, weight, mask)
            logits.append(out)
            attentions.append(attn.mean(dim=1).mean(dim=1))
        return torch.stack(logits).mean(dim=0), torch.stack(attentions).mean(dim=0)


def strip_embedding(net):
    """Replace an encoder's embedding table with a pass-through.

    The encoder then expects ``(batch, seq_len, d_model)`` float vectors in
    place of ``(batch, seq_len)`` token indices, and everything downstream is
    untouched. Call this only after :func:`check_equivalence` has run, since it
    discards the table that check compares against.
    """
    net.embedding = nn.Identity()
    return net


def check_equivalence(net, indices, weight, mask, atol=1e-5):
    """Assert a stripped encoder agrees with the original module.

    The intact encoder is fed token indices and uses its own embedding table;
    the stripped one is fed the rows that table would have returned. If the two
    disagree, the substitution is not faithful and the exported graph is
    wrong.

    Parameters
    ----------
    net : OtuAttentionEncoder
        Encoder with its embedding intact.
    indices : torch.Tensor
        Token indices of shape ``(batch, seq_len)``.
    weight, mask : torch.Tensor
        Abundance and mask inputs.
    atol : float, optional
        Absolute tolerance on the logits.

    Returns
    -------
    float
        The largest absolute logit difference seen.

    Raises
    ------
    AssertionError
        If the difference exceeds `atol`.
    """
    net.eval()
    with torch.no_grad():
        reference, _ = net(indices, weight, mask)
        table = net.embedding.weight.data
        gathered = table[indices]
        original = net.embedding
        net.embedding = nn.Identity()
        stripped, _ = net(gathered, weight, mask)
        net.embedding = original

    difference = float((reference - stripped).abs().max())
    assert difference <= atol, (
        f"stripped encoder deviates from the original by {difference:.3e}, "
        f"above the {atol:.1e} tolerance")
    return difference
