import torch
import torch.nn as nn
import torch.nn.functional as F
import padertorch as pt 

device = torch.device("cuda")


class AngularPenaltySMLoss(pt.Module):
    def __init__(
            self,
            in_features,
            out_features,
            eps=1e-7,
            scale=30.,
            margin=0.2,
    ):
        """
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        """
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps
        self.out_features = out_features

    def forward(self, embeddings, labels):
        """
        input shape (N, in_features)
        """
        assert len(embeddings) == len(labels), (embeddings, labels)
        assert torch.min(labels) >= 0, labels
        assert torch.max(labels) < self.out_features, (
            labels, self.out_features
        )

        # Normalize weight matrix and input vector
        for _, module in self.fc.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(module.weight, p=2, dim=1)

        embeddings = F.normalize(embeddings, p=2, dim=1)  # (batch embedding)

        # Get final outputs for classification
        logits = self.fc(embeddings)     # (batch classes)

        numerator = self.scale * (
            torch.diagonal(logits.transpose(0, 1)[labels]) - self.margin
        )

        # Remove y-th element
        excl = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1:])).unsqueeze(0) for i, y in
             enumerate(labels)], dim=0
        )

        # Compute softmax + cross entropy
        denominator = torch.exp(numerator) + torch.sum(
            torch.exp(self.scale * excl), dim=1
        )
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

    def extra_repr(self) -> str:
        return (
                f'{super().extra_repr()} scale={self.scale}, margin={self.margin},'
        )

        
       
       