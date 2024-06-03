import torch
import torch.nn as nn


class VT_CNN2(torch.nn.Module):
    def __init__(
        self,
        n_classes: int = 10,
        dropout: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(VT_CNN2, self).__init__()

        self.device = device
        self.loss = nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            nn.ZeroPad2d(
                padding=(
                    2,
                    2,
                    0,
                    0,
                )
            ),  # zero pad front/back of each signal by 2
            nn.Conv2d(
                in_channels=1, out_channels=256, kernel_size=(1, 3), stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.ZeroPad2d(
                padding=(
                    2,
                    2,
                    0,
                    0,
                )
            ),  # zero pad front/back of each signal by 2
            nn.Conv2d(
                in_channels=256,
                out_channels=80,
                kernel_size=(2, 3),
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Flatten(),
            nn.Linear(in_features=10560, out_features=256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=n_classes, bias=True),
        )

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        x = x.to(self.device)
        y_pred = self.model(x)
        y_pred = y_pred.to("cpu")
        y_pred = torch.softmax(y_pred, dim=-1)
        values, indices = torch.max(y_pred, dim=-1)
        indices = indices.numpy()
        return indices
