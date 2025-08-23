import os
import sys
import math
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurons.qlif import QLIF


class PoissonEncoder(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = int(T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        flat = x.view(B, -1).clamp(0.0, 1.0)  # (B, 784)
        spikes = torch.rand(self.T, B, flat.shape[1], device=x.device) < flat.unsqueeze(0)
        return spikes.float()


class SpikingMLP(nn.Module):
    def __init__(self, T=10, hidden=256, quantum_mode=True,
                 allow_dynamic_spike_probability=True, noise_std=0.3):
        super().__init__()
        self.T = int(T)
        self.encoder = PoissonEncoder(self.T)
        self.fc1 = nn.Linear(28*28, hidden, bias=True)
        self.neuron = QLIF(
            num_neurons=hidden,
            stochastic=True,
            noise_std=float(noise_std),
            quantum_mode=bool(quantum_mode),
            allow_dynamic_spike_probability=bool(allow_dynamic_spike_probability),
            learnable_threshold=True,
            learnable_qscale=True,
            learnable_qbias=True,
            use_ahp=True,
        )
        self.fc2 = nn.Linear(hidden, 10, bias=True)

        self.record_spikes: bool = False
        self.last_hidden_spike_hist: Optional[torch.Tensor] = None   # (T, hidden) for sample 0
        self.last_input_spike_counts: Optional[torch.Tensor] = None  # (28, 28) counts over T for sample 0

        for m in [self.fc1, self.fc2]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def reset_state(self):
        self.neuron.reset()

    def set_record(self, enabled: bool):
        self.record_spikes = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        B = x.shape[0]
        self.neuron.resize(B, device=dev)

        spikes_in = self.encoder(x)
        logits = torch.zeros(B, 10, device=dev)

        s_hist: List[torch.Tensor] = [] if self.record_spikes else None

        for t in range(self.T):
            I = self.fc1(spikes_in[t])      # (B, hidden)
            _ = self.neuron(I)              # updates internal state, sets .spike_values
            s = self.neuron.spike_values    # (B, hidden), float in [0,1] STE
            logits = logits + self.fc2(s)   # accumulate logits over time
            if self.record_spikes:
                s_hist.append(s.detach().clone())

        logits = logits / float(self.T)

        if self.record_spikes:
            if s_hist and x.shape[0] > 0:
                self.last_hidden_spike_hist = torch.stack(s_hist, dim=0)[:, 0, :].detach().cpu()
            else:
                self.last_hidden_spike_hist = None
            self.last_input_spike_counts = spikes_in[:, 0, :].sum(dim=0).view(28, 28).detach().cpu()

        return logits

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def seed_all(seed: int = 123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smoke_only(device="cpu"):
    seed_all(42)
    model = SpikingMLP(T=5, hidden=64, quantum_mode=True, allow_dynamic_spike_probability=True).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.rand(8, 1, 28, 28, device=device)  # fake images
    y = torch.randint(0, 10, (8,), device=device)

    model.reset_state()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"[SMOKE] forward/backward ok | loss={loss.item():.4f}")

def _ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def plot_spike_raster(hidden_spike_hist: torch.Tensor, max_neurons: int = 128,
                      s_threshold: float = 0.5, title: str = "Hidden spikes (raster)"):
    if hidden_spike_hist is None:
        plt.text(0.5, 0.5, "No spike history recorded", ha='center', va='center')
        plt.axis('off')
        return
    T, H = hidden_spike_hist.shape
    N = min(H, max_neurons)
    dat = hidden_spike_hist[:, :N].numpy()
    coords_t, coords_n = [], []
    for t in range(T):
        idx = (dat[t] > s_threshold).nonzero()[0]
        if idx.size > 0:
            coords_t.extend([t] * len(idx))
            coords_n.extend(idx.tolist())
    plt.scatter(coords_t, coords_n, s=4)
    plt.xlabel('t (timesteps)')
    plt.ylabel('neuron idx')
    plt.title(title)
    plt.tight_layout()


def plot_input_heatmap(input_counts: torch.Tensor, title: str = "Input spike counts (28x28)"):
    if input_counts is None:
        plt.text(0.5, 0.5, "No input counts recorded", ha='center', va='center')
        plt.axis('off')
        return
    plt.imshow(input_counts.numpy(), aspect='equal')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()


def plot_curves(train_losses: List[float], train_accs: List[float], test_accs: List[float],
                title_suffix: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses)
    axes[0].set_title(f"Loss{title_suffix}")
    axes[0].set_xlabel('step')
    axes[0].set_ylabel('loss')

    axes[1].plot([a * 100.0 for a in train_accs])
    axes[1].set_title(f"Train Acc%{title_suffix}")
    axes[1].set_xlabel('step')
    axes[1].set_ylabel('acc %')

    axes[2].plot([a * 100.0 for a in test_accs], marker='o')
    axes[2].set_title(f"Test Acc%{title_suffix}")
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('acc %')
    fig.tight_layout()
    return fig


def plot_side_by_side_comparison(A: Dict[str, Any], B: Dict[str, Any],
                                 labelA: str = "Run A (quantum+DSP)",
                                 labelB: str = "Run B (no-quantum+no-DSP)",
                                 outdir: Path = Path("figs"),
                                 raster_neurons: int = 128,
                                 s_threshold: float = 0.5) -> Tuple[Path, Path]:
    _ensure_outdir(outdir)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    plt.sca(axes[0, 0])
    plot_spike_raster(A.get('hidden_spike_hist'), raster_neurons, s_threshold, title=f"{labelA}: Hidden spikes")
    axes[0, 1].plot(A.get('train_losses', []))
    axes[0, 1].set_title(f"{labelA}: Loss")
    axes[0, 1].set_xlabel('step')
    axes[0, 1].set_ylabel('loss')
    axes[0, 2].plot([a * 100.0 for a in A.get('test_accs', [])], marker='o')
    axes[0, 2].set_title(f"{labelA}: Test Acc%")
    axes[0, 2].set_xlabel('epoch')
    axes[0, 2].set_ylabel('acc %')

    # B row
    plt.sca(axes[1, 0])
    plot_spike_raster(B.get('hidden_spike_hist'), raster_neurons, s_threshold, title=f"{labelB}: Hidden spikes")
    axes[1, 1].plot(B.get('train_losses', []))
    axes[1, 1].set_title(f"{labelB}: Loss")
    axes[1, 1].set_xlabel('step')
    axes[1, 1].set_ylabel('loss')
    axes[1, 2].plot([a * 100.0 for a in B.get('test_accs', [])], marker='o')
    axes[1, 2].set_title(f"{labelB}: Test Acc%")
    axes[1, 2].set_xlabel('epoch')
    axes[1, 2].set_ylabel('acc %')

    fig.tight_layout()
    p1 = outdir / "compare_runs__spikes_loss_acc.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    plt.sca(axes2[0])
    plot_input_heatmap(A.get('input_counts'), title=f"{labelA}: Input spike counts")
    plt.sca(axes2[1])
    plot_input_heatmap(B.get('input_counts'), title=f"{labelB}: Input spike counts")
    fig2.tight_layout()
    p2 = outdir / "compare_runs__input_heatmaps.png"
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)

    return p1, p2

def train_eval_one_run(args, run_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Prepare config
    cfg = vars(args).copy()
    if run_override:
        cfg.update(run_override)

    seed_all(cfg['seed'])
    device = torch.device(cfg['device'])

    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.MNIST(root=str(Path.home() / ".torch" / "datasets"),
                              train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=str(Path.home() / ".torch" / "datasets"),
                              train=False, download=True, transform=transform)

    if cfg['max_samples'] is not None:
        train_indices = list(range(min(cfg['max_samples'], len(train_ds))))
        train_ds = Subset(train_ds, train_indices)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=pin)

    model = SpikingMLP(T=cfg['T'], hidden=cfg['hidden'],
                       quantum_mode=not cfg['no_quantum'],
                       allow_dynamic_spike_probability=not cfg['no_dsp'],
                       noise_std=cfg['noise_std']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])

    print(model)
    print(f"Device: {device} | T={cfg['T']} | hidden={cfg['hidden']} | quantum_mode={not cfg['no_quantum']} | DSP={not cfg['no_dsp']}")

    train_losses: List[float] = []
    train_accs:   List[float] = []  # per-step (batch)
    test_accs:    List[float] = []  # per-epoch

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        batches = 0

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            if cfg['max_batches'] and batch_idx > cfg['max_batches']:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            model.reset_state()
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            if cfg['clip_grad'] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            opt.step()

            with torch.no_grad():
                acc = accuracy_from_logits(logits, y)

            train_losses.append(loss.item())
            train_accs.append(acc)
            batches += 1

            if batch_idx % 20 == 0 or batch_idx == 1:
                print(f"Epoch {epoch} | Step {batch_idx:04d} | loss {loss.item():.4f} | acc {acc*100:5.1f}%")

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                model.reset_state()
                logits = model(x)
                correct += (logits.argmax(1) == y).float().sum().item()
                total += y.numel()
        test_acc = correct / max(1, total)
        test_accs.append(test_acc)
        print(f"[TEST] epoch {epoch} | accuracy: {test_acc*100:.2f}%")

    hidden_hist: Optional[torch.Tensor] = None
    input_counts: Optional[torch.Tensor] = None
    try:
        model.eval()
        model.set_record(True)
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                model.reset_state()
                _ = model(x)
                hidden_hist = model.last_hidden_spike_hist
                input_counts = model.last_input_spike_counts
                break
    finally:
        model.set_record(False)

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'hidden_spike_hist': hidden_hist,
        'input_counts': input_counts,
        'cfg': cfg,
    }

def train_mnist(args):
    metrics = train_eval_one_run(args, run_override=None)

    if args.plot:
        outdir = _ensure_outdir(Path(args.outdir))

        fig = plot_curves(metrics['train_losses'], metrics['train_accs'], metrics['test_accs'])
        p_curves = outdir / "single_run__curves.png"
        fig.savefig(p_curves, dpi=140)
        plt.close(fig)

        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        plt.sca(axes[0])
        plot_spike_raster(metrics['hidden_spike_hist'], max_neurons=args.raster_neurons, s_threshold=args.spike_threshold,
                          title="Hidden spikes (raster)")
        plt.sca(axes[1])
        plot_input_heatmap(metrics['input_counts'], title="Input spike counts (28x28)")
        fig2.tight_layout()
        p_spikes = outdir / "single_run__spikes_and_input.png"
        fig2.savefig(p_spikes, dpi=140)
        plt.close(fig2)

        print(f"Saved figures to: {p_curves} and {p_spikes}")


def compare_two_runs(args):
    mode = args.compare_mode
    if mode not in {"both", "quantum", "dsp"}:
        raise ValueError("compare_mode must be one of: both | quantum | dsp")

    print("=== Training Run A (baseline) ===")
    A = train_eval_one_run(args, run_override=None)

    # Run B (modified)
    override: Dict[str, Any] = {}
    if mode in {"both", "quantum"}:
        override['no_quantum'] = True
    if mode in {"both", "dsp"}:
        override['no_dsp'] = True

    print("\n=== Training Run B (modified) ===")
    B = train_eval_one_run(args, run_override=override)

    outdir = _ensure_outdir(Path(args.outdir))
    labelA = f"Run A (quantum={not A['cfg']['no_quantum']}, DSP={not A['cfg']['no_dsp']})"
    labelB = f"Run B (quantum={not B['cfg']['no_quantum']}, DSP={not B['cfg']['no_dsp']})"

    p1, p2 = plot_side_by_side_comparison(A, B, labelA=labelA, labelB=labelB,
                                          outdir=outdir,
                                          raster_neurons=args.raster_neurons,
                                          s_threshold=args.spike_threshold)


    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(A['train_losses'], label=labelA)
    ax1.plot(B['train_losses'], label=labelB)
    ax1.set_title('Loss (overlay)')
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot([a * 100.0 for a in A['train_accs']], label=labelA)
    ax2.plot([a * 100.0 for a in B['train_accs']], label=labelB)
    ax2.set_title('Train Acc% (overlay)')
    ax2.set_xlabel('step')
    ax2.set_ylabel('acc %')
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot([a * 100.0 for a in A['test_accs']], marker='o', label=labelA)
    ax3.plot([a * 100.0 for a in B['test_accs']], marker='o', label=labelB)
    ax3.set_title('Test Acc% (overlay)')
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('acc %')
    ax3.legend()

    fig.tight_layout()
    p3 = Path(args.outdir) / "compare_runs__curves_overlay.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)

    print(f"Saved comparison figures to: {p1}, {p2}, {p3}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--T", type=int, default=10, help="timesteps for rate code")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--max-batches", type=int, default=100, help="limit batches per epoch for quick smoke run")
    p.add_argument("--max-samples", type=int, default=5000, help="subset train set for speed; set None for full")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--no-quantum", action="store_true", help="disable quantum_mode in QLIF for ablation")
    p.add_argument("--no-dsp", action="store_true", help="disable DynamicSpikeProbability ablation")

    p.add_argument("--plot", action="store_true", help="after single-run training, save plots (curves + spikes)")
    p.add_argument("--outdir", type=str, default="figs", help="directory to write figures")
    p.add_argument("--raster-neurons", type=int, default=128, help="max hidden neurons to show in raster")
    p.add_argument("--spike-threshold", type=float, default=0.5, help="threshold on s for marking a spike in raster")

    p.add_argument("--compare", action="store_true", help="train two runs and create side-by-side comparison figs")
    p.add_argument("--compare-mode", type=str, default="both", choices=["both", "quantum", "dsp"],
                   help="what to disable in Run B (relative to Run A)")

    args = p.parse_args()

    if args.smoke_only:
        smoke_only(args.device)
    else:
        if args.compare:
            compare_two_runs(args)
        else:
            train_mnist(args)
