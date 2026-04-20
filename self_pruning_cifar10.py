# CIFAR-10 Self-Pruning Neural Network
# I tried to implement learnable gate-based pruning from the paper we read in class.
# Took me a while to figure out the sparsity loss part - basically it's just pushing
# the gate sigmoins toward 0 so the weights die off naturally during training.
# Reference: "Learning both Weights and Connections for Efficient Neural Networks" (Han et al.)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# use GPU if available, else CPU (my laptop is slow without it lol)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


class PrunableLinear(nn.Module):
    # Custom linear layer where each weight has a learnable gate
    # gate goes through sigmoid so it's always between 0 and 1
    # when gate -> 0, weight is effectively dead = pruned
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

        # init gates small so sigmoid(gate) ~ 0.5 at start
        # tried zeros first but 0.5 everywhere felt more neutral
        self.gate_param = nn.Parameter(torch.zeros(out_features, in_features))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        g = torch.sigmoid(self.gate_param)
        w = self.weight * g
        return F.linear(x, w, self.bias)

    def gate_values(self):
        return torch.sigmoid(self.gate_param)


class MLP(nn.Module):
    # simple 5-layer MLP, fc1 is normal, rest are prunable
    # CIFAR images are 3x32x32 = 3072 input features
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3072, 512)           # first layer stays dense
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 64)
        self.fc5 = PrunableLinear(64, 10)          # 10 classes

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def all_gates(self):
        # collect all gate tensors into one flat vector for analysis
        g = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                g.append(m.gate_values().detach().view(-1))
        return torch.cat(g) if g else torch.tensor([])


def sparsity_percent(model):
    # sparsity = fraction of "dead" weights (gate close to 0)
    # using mean(gate) as density, so sparsity = 1 - density
    gates = model.all_gates()
    if len(gates) == 0:
        return 0.0
    return (1.0 - gates.mean().item()) * 100.0


def get_dataloaders(batch_size=128):
    # standard CIFAR-10 normalization
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_ds = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=tfm)
    test_ds  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=tfm)
    # num_workers=2 to avoid issues on some systems
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, test_dl


def evaluate(model, testloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc  = 100.0 * correct / total
    spar = sparsity_percent(model)
    return acc, spar


def sparsity_loss(model):
    # L1 on gates - sum of sigmoid values (all positive so it's basically L1)
    # optimizer will push these toward 0 to minimize total loss
    sp = torch.tensor(0.0, device=device)
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            sp = sp + m.gate_values().sum()
    return sp


def train(lambd=1e-3, epochs=15):
    print(f"\n=== lambda = {lambd:.0e} ===")
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # lr scheduler - helps squeeze out a bit more accuracy in later epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    train_dl, test_dl = get_dataloaders()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            out     = model(imgs)
            ce      = F.cross_entropy(out, labels)
            sp      = sparsity_loss(model)
            loss    = ce + lambd * sp

            loss.backward()
            # clip gradients - gates can get unstable without this
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        acc, spar = evaluate(model, test_dl)
        avg_loss = running_loss / len(train_dl)
        print(f"  epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.2f}%  sparsity={spar:.2f}%")

    final_acc, final_spar = evaluate(model, test_dl)
    print(f"  --> final  acc={final_acc:.2f}%  sparsity={final_spar:.2f}%")
    return final_acc, final_spar, model


# ---- run experiments ----
# trying 5 different lambdas including 0 as baseline (no pruning pressure)
lambdas = [0.0, 1e-4, 1e-3, 5e-3, 1e-2]
results = []
best_model = None

for lam in lambdas:
    acc, spar, mdl = train(lam, epochs=15)
    results.append({'lambda': lam, 'acc': acc, 'sparsity': spar})
    # saving the lambda=1e-3 model for gate histogram - seemed like best tradeoff
    if lam == 1e-3:
        best_model = mdl

# results table
print("\n\nResults:")
print(f"{'Lambda':<12} {'Accuracy (%)':>14} {'Sparsity (%)':>14}")
print("-" * 42)
for r in results:
    print(f"{r['lambda']:<12.0e} {r['acc']:>14.2f} {r['sparsity']:>14.2f}")


# ---- plots ----
lam_labels = [str(r['lambda']) for r in results]
accs  = [r['acc']      for r in results]
spars = [r['sparsity'] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Self-Pruning MLP on CIFAR-10", fontsize=13)

# accuracy bar chart
axes[0].bar(lam_labels, accs, color='steelblue', edgecolor='black', width=0.5)
axes[0].set_title("Test Accuracy")
axes[0].set_xlabel("Lambda")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 60)
for i, v in enumerate(accs):
    axes[0].text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=9)

# sparsity bar chart
axes[1].bar(lam_labels, spars, color='coral', edgecolor='black', width=0.5)
axes[1].set_title("Weight Sparsity")
axes[1].set_xlabel("Lambda")
axes[1].set_ylabel("Sparsity (%)")
axes[1].set_ylim(0, 100)
for i, v in enumerate(spars):
    axes[1].text(i, v + 1, f"{v:.1f}", ha='center', fontsize=9)

# gate distribution for best model
if best_model is not None:
    gates_np = best_model.all_gates().cpu().numpy()
    mean_g   = gates_np.mean()
    axes[2].hist(gates_np, bins=50, range=(0, 1), color='mediumseagreen', edgecolor='black', alpha=0.8)
    axes[2].axvline(mean_g, color='red', linestyle='--', linewidth=1.5, label=f"mean = {mean_g:.3f}")
    axes[2].set_title("Gate Distribution (λ=1e-3)")
    axes[2].set_xlabel("Gate Value (sigmoid)")
    axes[2].set_ylabel("Count")
    axes[2].legend()
    # bimodal peak near 0 means lots of pruned weights
    axes[2].annotate("many weights pruned\n(gate ≈ 0)", xy=(0.05, axes[2].get_ylim()[1]*0.7),
                     fontsize=8, color='grey')

plt.tight_layout()
plt.savefig("results_plot.png", dpi=150)
plt.show()
print("saved results_plot.png")
