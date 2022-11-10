from torch import nn

class Aggregator(nn.Module):
    def __init__(self, embed_dim=256, aggregation='concat', normalize=False):
        super().__init__()
        self.aggregation = aggregation
        self.normalize = normalize

        if aggregation == 'concat':
            self.aggregator = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(True),
                nn.Linear(embed_dim, embed_dim)
            )
        elif aggregation == 'dot' or aggregation == 'sum':
            self.aggregator = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(True),
                nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, x1, x2):
        if self.normalize:
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
        if self.aggregation == 'concat':
            x = torch.cat((x1, x2), dim=-1)
        elif self.aggregation == 'dot':
            x = x1 * x2
        elif self.aggregation == 'sum':
            x = x1 + x2
        return self.aggregator(x)
