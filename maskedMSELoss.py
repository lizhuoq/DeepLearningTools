from package import *


class MaskedMSELoss(nn.MSELoss):
    def forward(self, pred, label, is_valid):
        """

        :param pred: shape: (batch_size, seq_len)
        :param label: shape: (batch_size, seq_len)
        :param is_valid: shape: (batch_size, seq_len)
        :return:
        """
        self.reduction = 'none'
        unweighted_loss = super(MaskedMSELoss, self).forward(pred, label)
        weighted_loss = unweighted_loss * is_valid
        return weighted_loss.sum() / is_valid.sum()
