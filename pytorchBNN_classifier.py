import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.draw import disk
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------- Binary Ops --------------------

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).float()

def binarize(x):
    return SignSTE.apply(x)

# -------------------- Binary Linear Layer --------------------

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        bin_w = binarize(self.weight)
        return F.linear(x, bin_w, self.bias)

# -------------------- XNOR Classifier --------------------

class XNORClassifier(nn.Module):
    def __init__(self, inference_mode=False):
        super().__init__()
        self.inference_mode = inference_mode
        self.fc1 = BinaryLinear(256, 128)
        self.fc2 = BinaryLinear(128, 64)
        self.fc3 = BinaryLinear(64, 1)

    def forward(self, x):
        x = binarize(x)
        x = binarize(self.fc1(x))
        x = binarize(self.fc2(x))
        x = self.fc3(x)
        if self.inference_mode:
            return x.sign()  # 1 or -1
        else:
            return torch.sigmoid(x)  # For BCE training

# -------------------- Dataset --------------------

class CanvasDataset(Dataset):
    def __init__(self, n=1000, size=16, fixed_shape=None):
        self.X = []
        self.y = []
        for _ in range(n):
            canvas = np.zeros((size, size), dtype=np.uint8)
            shape = fixed_shape or np.random.choice(["circle", "rect"])
            if shape == "circle":
                rr, cc = disk((size//2, size//2), size//3)
                canvas[rr, cc] = 1
                label = 1
            else:
                canvas[size//4:3*size//4, size//4:3*size//4] = 1
                label = 0
            self.X.append(canvas.flatten())
            self.y.append(label)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# -------------------- Training --------------------

def train(model, dataloader, epochs=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for xb, yb in dataloader:
            xb = xb.float()
            yb = yb.float().unsqueeze(1)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += ((preds > 0.5).float() == yb).sum().item()
        acc = correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}, Acc = {acc:.3f}")

# -------------------- Inference --------------------

def show_canvas(canvas, title=""):
    plt.imshow(canvas.reshape(16, 16), cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()

def evaluate(model, label_name, label_val, display=False):
    test_ds = CanvasDataset(n=100, fixed_shape=label_name)
    correct = 0
    model.inference_mode = True
    model.eval()
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y_true = test_ds[i]
            x = x.unsqueeze(0)
            pred = model(x.float()).item()
            pred_label = 1 if pred > 0 else 0
            if pred_label == y_true.item():
                correct += 1
            if display and i < 5:
                label_str = "circle" if pred_label == 1 else "rectangle"
                truth_str = "circle" if y_true.item() == 1 else "rectangle"
                show_canvas(x.squeeze().numpy(), title=f"Pred: {label_str} / True: {truth_str}")
    print(f"{label_name.capitalize()} Test Accuracy: {correct}/100 = {correct/100:.2f}")

# -------------------- Run Everything --------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    ds = CanvasDataset(n=500)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = XNORClassifier(inference_mode=False)
    train(model, dl, epochs=20)

    print("\n=== Inference with Hard Binary Output ===")
    #evaluate(model, "circle", 1, display=True)
    evaluate(model, "rect", 0, display=True)
