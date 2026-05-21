"""
3개 모델 (TextCNN / BiLSTM / CNN-LSTM Hybrid) + 공통 4-head multi-task wrapper.
인코더만 다르고 임베딩·head 구조는 동일하게 두어 모델 비교가 공정해지도록 설계.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 공통 4-task head ----------
class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: dict[str, int], dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            name: nn.Linear(in_dim, k) for name, k in num_classes.items()
        })

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.dropout(h)
        return {name: head(h) for name, head in self.heads.items()}


# ---------- 1) TextCNN ----------
class TextCNN(nn.Module):
    """Yoon Kim (2014). multi-kernel CNN + max-over-time pooling."""
    def __init__(
        self,
        vocab_size: int,
        num_classes: dict[str, int],
        embed_dim: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        n_filters: int = 96,
        dropout: float = 0.4,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        out_dim = n_filters * len(kernel_sizes)
        self.head = MultiTaskHead(out_dim, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, L]
        e = self.embed(x).transpose(1, 2)  # [B, embed, L]
        feats = []
        for conv in self.convs:
            c = F.relu(conv(e))             # [B, F, L]
            c, _ = c.max(dim=2)             # max-over-time → [B, F]
            feats.append(c)
        h = torch.cat(feats, dim=1)         # [B, F*len(kernels)]
        return self.head(h)


# ---------- 2) BiLSTM ----------
class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: dict[str, int],
        embed_dim: int = 128,
        hidden: int = 128,
        n_layers: int = 1,
        dropout: float = 0.4,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=0.0,  # 1-layer라 LSTM 내부 dropout은 의미 없음
        )
        out_dim = hidden * 2
        self.head = MultiTaskHead(out_dim, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, L]
        mask = (x != self.pad_id).float().unsqueeze(-1)  # [B, L, 1]
        e = self.embed(x)                                 # [B, L, E]
        out, _ = self.lstm(e)                             # [B, L, 2H]
        # masked mean pooling (pad는 빼고 평균)
        out = out * mask
        h = out.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B, 2H]
        return self.head(h)


# ---------- 3) CNN-LSTM Hybrid ----------
class CNNLSTMHybrid(nn.Module):
    """
    CNN으로 local n-gram feature map을 뽑고, 길이 축으로 stride를 줘서 시퀀스 길이를 줄인 뒤
    그 시퀀스를 BiLSTM이 받아 전역 의존성을 인코딩.
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: dict[str, int],
        embed_dim: int = 128,
        n_filters: int = 96,
        kernel_size: int = 5,
        stride: int = 2,
        hidden: int = 128,
        dropout: float = 0.4,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.stride = stride
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.conv = nn.Conv1d(embed_dim, n_filters, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.lstm = nn.LSTM(
            n_filters, hidden, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        out_dim = hidden * 2
        self.head = MultiTaskHead(out_dim, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = (x != self.pad_id).float()                 # [B, L]
        e = self.embed(x).transpose(1, 2)                 # [B, E, L]
        c = F.relu(self.conv(e))                          # [B, F, L']
        c = c.transpose(1, 2)                             # [B, L', F]

        # CNN stride 적용된 mask (단순 다운샘플링)
        # padding으로 인해 길이가 정확히 ceil(L/stride)가 아닐 수 있어 실제 출력 길이에 맞춤
        L_out = c.size(1)
        # mask를 stride 단위로 풀링
        m = F.avg_pool1d(mask.unsqueeze(1), kernel_size=self.stride, stride=self.stride,
                         ceil_mode=False).squeeze(1)      # [B, L_out_approx]
        if m.size(1) < L_out:
            pad = torch.zeros(m.size(0), L_out - m.size(1), device=m.device)
            m = torch.cat([m, pad], dim=1)
        elif m.size(1) > L_out:
            m = m[:, :L_out]
        m = (m > 0).float().unsqueeze(-1)                 # [B, L_out, 1]

        out, _ = self.lstm(c)                             # [B, L_out, 2H]
        out = out * m
        h = out.sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # [B, 2H]
        return self.head(h)


# ---------- 모델 팩토리 ----------
def build_model(name: str, vocab_size: int, num_classes: dict[str, int]) -> nn.Module:
    name = name.lower()
    if name == "textcnn":
        return TextCNN(vocab_size, num_classes)
    if name == "bilstm":
        return BiLSTM(vocab_size, num_classes)
    if name == "hybrid":
        return CNNLSTMHybrid(vocab_size, num_classes)
    raise ValueError(f"unknown model: {name}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    from data import NUM_CLASSES
    vocab = 1600
    x = torch.randint(0, vocab, (4, 512))
    for name in ["textcnn", "bilstm", "hybrid"]:
        m = build_model(name, vocab, NUM_CLASSES)
        out = m(x)
        shapes = {k: tuple(v.shape) for k, v in out.items()}
        print(f"{name:8s} params={count_params(m):,}  out={shapes}")
