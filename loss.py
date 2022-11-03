import torch.nn.functional as F
import torch.nn as nn

class KLLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.error_metric = nn.KLDivLoss(size_average=True, reduce=True, log_target=False)

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        # required while KLDivloss log_target==False
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss