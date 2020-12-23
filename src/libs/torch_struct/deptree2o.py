from typing import Tuple

import torch
from fastNLP import seq_len_to_mask
from torch import Tensor

from .helpers import Chart, _Struct

# Constants
# A, B are right-facing and left-facing respectly. (see 6.b vectorized parsing)
# L, R, C, I, S are flags in eisner algorithm.
# U is for S only.
A, B = 0, 1
L, R, U = 0, 1, 0
I, C, S = 0, 1, 2


class DepTree2O(_Struct):
    """
    A projective dependency CRF with sibling.

    Parameters:
        scores: Tuple[arc_scores, sib_scores]
            arc_scores: Arc scores of shape (B, N + 1, N + 1) with root at index 0.
                parent to child, or arc_scores_in[0, i, j] is the score of arc i to j.
            sib_scores: Sibling scores of shape (B, N + 1, N + 1, N + 1) with root at index 0.
                [parent, child, sibling].

    Note: For single-root case, cache is forced to False

    """
    def _dp(self, scores, lengths=None, force_grad=False, cache=True):

        multiroot = getattr(self, "multiroot", False)
        cache = False if not multiroot else cache

        # 1. Check shape, length
        # 2. Call requires_grad_
        # 3. Mask out scores on invalid position
        scores, batch, N, lengths = self._check_potentials(scores, lengths)

        # Init, every chart has shape (semiring_size, batch, N[parent], N[span length])
        # Chart A,B have different direction, for example, for A, 0 is length=1, while for B, -1 is length=1.
        # This is because Length_A[i] + Legnth_B[i] = k (the k in the following for-loop)
        semiring = self.semiring
        alpha = [[[Chart((batch, N, N), scores, semiring, cache=cache) for _ in range(2 if _2 != S else 1)]
                  for _2 in range(3)] for _ in range(2)]
        semiring.one_(alpha[A][C][L].data[:, :, :, 0].data)
        semiring.one_(alpha[A][C][R].data[:, :, :, 0].data)
        semiring.one_(alpha[B][C][L].data[:, :, :, -1].data)
        semiring.one_(alpha[B][C][R].data[:, :, :, -1].data)

        start_idx = 0 if multiroot else 1
        # most comments in loops are from yzhangcs's supar/modules/treecrf.py:CRF2oDependency
        # https://github.com/yzhangcs/parser
        for k in range(1, N - start_idx):
            # bould of span, [i, j]
            f = torch.arange(start_idx, N - k), torch.arange(k + start_idx, N)

            # I(j->i) = logsum(exp(I(j->r) + S(j->r, i)) +, i < r < j
            #                  exp(C(j->j) + C(i->j-1)))
            #           + s(j->i)
            x = alpha[A][C][R][f[0], k - 1]
            if k > 1:
                ASU = alpha[A][S][U][start_idx:N - k, 1:k]
                BIL = alpha[B][I][L][k + start_idx:, N - k:-1]
                s = semiring.dot(
                    semiring.times(
                        ASU,
                        stripe(scores[..., f[1], f[0], :], N - k - start_idx, k - 1, (0, 1 + start_idx))), BIL)
                x = semiring.plus(x, s)
            x = semiring.times(x, scores[:, :, f[1], f[0], f[0]])
            alpha[A][I][L][start_idx:N - k, k] = x
            alpha[B][I][L][k + start_idx:N, N - k - 1] = x

            # I(i->j) = logsum(exp(I(i->r) + S(i->r, j)) +, i < r < j
            #                  exp(C(i->i) + C(j->i+1)))
            #           + s(i->j)
            x = alpha[B][C][L][f[1], N - k]
            if k > 1:
                AIR = alpha[A][I][R][start_idx:N - k, 1:k]
                BSU = alpha[B][S][U][k + start_idx:, N - k:-1]
                s = semiring.dot(
                    semiring.times(AIR, stripe(scores[..., f[0], f[1], :], N - k - start_idx, k - 1, (0, 1 + start_idx))),
                    BSU)
                x = semiring.plus(x, s)
            x = semiring.times(x, scores[:, :, f[0], f[1], f[1]])
            alpha[A][I][R][start_idx:N - k, k] = x
            alpha[B][I][R][k + start_idx:N, N - k - 1] = x

            # S(j, i) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            # S(i, j) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            ACR = alpha[A][C][R][start_idx:N - k, :k]
            BCL = alpha[B][C][L][k + start_idx:, N - k:]
            x = semiring.dot(ACR, BCL)
            alpha[A][S][U][start_idx:N - k, k] = x
            alpha[B][S][U][k + start_idx:, N - k - 1] = x

            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            ACL = alpha[A][C][L][start_idx:N - k, :k]
            BIL = alpha[B][I][L][k + start_idx:, N - k - 1:N - 1]
            new = semiring.dot(ACL, BIL)
            alpha[A][C][L][start_idx:N - k, k] = new
            alpha[B][C][L][k + start_idx:, N - k - 1] = new

            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            AIR = alpha[A][I][R][start_idx:N - k, 1:k + 1]
            BCR = alpha[B][C][R][k + start_idx:, N - k:]
            new = semiring.dot(AIR, BCR)
            alpha[A][C][R][start_idx:N - k, k] = new
            alpha[B][C][R][k + start_idx:, N - k - 1] = new

        if not multiroot:
            # if not multiroot, there are one extra arc from ROOT to a word.
            # root has not sibling score because there is only one arc from root.
            root_incomplete_span = semiring.times(alpha[A][C][L][1, :N - 1],
                                                  stripe(scores, N - 1, 1, (1, 1))[:, :, 0, :, 0])
            for k in range(1, N):
                AIR = root_incomplete_span[:, :, :k]
                BCR = alpha[B][C][R][k, N - k:]
                alpha[A][C][R][0, k] = semiring.dot(AIR, BCR)

        final = alpha[A][C][R][(0, )]
        v = torch.stack([final[:, i, l] for i, l in enumerate(lengths)], dim=1)
        return v, [scores], alpha

    def _check_potentials(self, scores, lengths=None):
        semiring = self.semiring
        batch, N, N2, N3 = self._get_dimension_and_requires_grad(scores)
        assert N == N2 == N3, "Non-square potentials"

        if lengths is None:
            lengths = torch.LongTensor([N - 1] * batch).to(scores.device)
        else:
            assert max(lengths) <= N, "Length longer than N"

        scores = semiring.convert(scores)
        scores = scores.clone()  # avoid leaf error when backward

        mask = seq_len_to_mask(lengths + 1, N)
        mask3d = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(-1) * mask.view(batch, 1, 1, N)
        semiring.zero_mask_(scores, ~mask3d)

        return scores, batch, N, lengths

    @staticmethod
    def convert(arc_scores: Tensor, sib_scores: Tensor) -> Tensor:
        """from (arc, sib) to one tensor.
        arc: batch x seq_len(parent) x seq_len(child)
        sib: batch x seq_len(parent) x seq_len(child) x seq_len(sibling), valid only when child!=sibling
        """
        new_scores = sib_scores.clone().contiguous()
        t = stripe(new_scores, arc_scores.shape[1], 1)
        t.copy_(arc_scores.unsqueeze(-1))
        return new_scores

    def _arrange_marginals(self, grads):
        return self.semiring.convert(self.semiring.unconvert(grads[0]))

    @staticmethod
    def to_parts(sequence: Tuple[Tensor, Tensor], extra=None, lengths=None) -> Tensor:
        """
        Convert a sequence representation to arcs

        Parameters:
            sequence : Tuple[arc, sib]
                both are b x (N+1) long tensor in [0, N] (indexing is +1), where 0 is root (and its value is ignored).
            lengths: seq_len without root
        Returns:
            arcs : b x (N+1) x (N+1) x (N+1) arc indicators, parent, child, sibling
        """
        arc, sib = sequence
        batch, N1 = arc.shape
        if lengths is None:
            lengths = torch.LongTensor([N1 - 1] * batch)
        labels = torch.zeros(batch, N1, N1, N1).long()
        batch_arange = torch.arange(batch)
        for n in range(1, N1):
            m = sib[:, n] != -1
            labels[batch_arange[m], arc[:, n][m], n, sib[:, n][m]] = 1
            labels[batch_arange, arc[:, n], n, n] = 1
        for b in range(batch):
            labels[b, lengths[b] + 1:] = 0
            labels[b, :, lengths[b] + 1:] = 0
            labels[b, :, :, lengths[b] + 1:] = 0
        return labels

    @staticmethod
    def from_parts(arcs):
        """
        Convert a arc representation to sequence

        Parameters:
            arcs : b x (N+1) x (N+1) arc indicators
        Returns:
            sequence : b x (N+1) long tensor in [0, N] (indexing is +1), where 0 is root (and its value is always 0).
        """
        raise NotImplementedError
        batch, N, _ = arcs.shape
        labels = torch.zeros(batch, N + 1).long()
        on = arcs.nonzero(as_tuple=False)
        for i in range(on.shape[0]):
            labels[on[i][0], on[i][2]] = on[i][1]
        labels[:, 0] = 0
        return labels, None


def stripe(x, n, w, offset=(0, 0), dim=1):
    # based on yzhangcs's supar/utils/fn.py:stripe
    # https://github.com/yzhangcs/parser
    # MODIFIED: on the last two dim
    # ORIG: on the first two dim
    r"""
    Returns a diagonal stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 1 if returns a horizontal stripe; 0 otherwise.

    Returns:
        a diagonal stripe of the tensor.

    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> stripe(x, 2, 3)
        tensor([[0, 1, 2],
                [6, 7, 8]])
        >>> stripe(x, 2, 3, (1, 1))
        tensor([[ 6,  7,  8],
                [12, 13, 14]])
        >>> stripe(x, 2, 3, (1, 1), 0)
        tensor([[ 6, 11, 16],
                [12, 17, 22]])
    """
    assert x.is_contiguous(), "x must be contiguous, or write on new view will lost."
    seq_len = x.size(-1)
    stride = list(x.stride())
    stride[-2] = seq_len + 1
    stride[-1] = 1 if dim == 1 else seq_len
    return x.as_strided(size=(
        *x.shape[:-2],
        n,
        w,
    ),
                        stride=stride,
                        storage_offset=x.storage_offset() + (offset[0] * seq_len + offset[1]))
