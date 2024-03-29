"""

A PyTorch implementation of CRF

An adaptation of `ADVANCED: MAKING DYNAMIC DECISIONS AND THE BI-LSTM CRF` at 
https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sadvanced-making-dynamic-decisions-and-the-bi-lstm-crf


"""


import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class CRF(nn.Module):
    def __init__(self, tag_to_ix):
        super(CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        device = self.transitions.device

        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.0, device=device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.0

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        device = self.transitions.device
        # Gives the score of a provided tag sequence
        score = torch.zeros(1, device=device)
        tags = torch.cat(
            [
                torch.tensor(
                    [self.tag_to_ix[START_TAG]], dtype=torch.long, device=device
                ),
                tags,
            ]
        )
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # feats: [seq_len, self.target_size]

        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full(
            (1, self.tagset_size), -10000.0, device=self.transitions.device
        )
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            # feat [self.target_size]

            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = (
                    forward_var + self.transitions[next_tag]
                )  # [1, self.target_size] 在本时刻，标签next_tag的前一标签的可能性分数
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, feats, tags):
        num_feats = feats.shape[0]
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return (forward_score - gold_score) / num_feats

    def inference(self, feats):  # dont confuse this with _forward_alg above.
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
