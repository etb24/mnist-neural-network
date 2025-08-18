import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# utils
def to_categorical(x, n_col=None):
    if n_col is None:
        n_col = int(np.amax(x)) + 1
    one_hot = np.zeros((x.shape[0], n_col), dtype=np.float32)
    one_hot[np.arange(x.shape[0]), x] = 1.0
    return one_hot

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred, axis=0) / len(y_true)

def batch_loader(X, y=None, batch_size=512, shuffle=True):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        sel = idx[i:i+batch_size]
        yield (X[sel], y[sel]) if y is not None else X[sel]

# activations and loss
class LeakyReLU:
    def __init__(self, alpha=0.2): self.alpha = alpha
    def __call__(self, x): return np.where(x >= 0, x, self.alpha * x)
    def gradient(self, x): return np.where(x >= 0, 1.0, self.alpha)

class Softmax:
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

class CrossEntropy:
    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1-1e-15)
        return -np.sum(y * np.log(p), axis=1)

class Activation:
    def __init__(self, activation, name="activation"):
        self.activation = activation
        self.input = None
        self.output = None
        self.name = name
    def forward(self, x):
        self.input = x
        self.output = self.activation(x)
        return self.output
    def backward(self, output_error, **kwargs):
        if isinstance(self.activation, Softmax):
            return output_error
        grad_fn = getattr(self.activation, "gradient", None)
        return output_error if grad_fn is None else grad_fn(self.input) * output_error
    def __call__(self, x): return self.forward(x)

# linear layer with Adam optimizer
class Linear:
    def __init__(self, n_in, n_out, name="linear"):
        # He-uniform with LeakyReLU gain
        alpha = 0.2
        gain = np.sqrt(2.0 / (1 + alpha**2))
        limit = gain * np.sqrt(6.0 / n_in)
        self.W = np.random.uniform(-limit, limit, (n_in, n_out)).astype(np.float32)
        self.b = np.zeros((1, n_out), dtype=np.float32)
        self.input = None
        self.output = None
        self.name = name
        # Adam state
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t = 0

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.W) + self.b
        return self.output

    def backward(self, output_error, lr=1e-3, weight_decay=1e-4,
                 beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        self.t += 1
        # grads + L2
        dW = np.dot(self.input.T, output_error) + weight_decay * self.W
        db = np.mean(output_error, axis=0, keepdims=True)

        # Adam for W
        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * (dW * dW)
        mWhat = self.mW / (1 - beta1 ** self.t)
        vWhat = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mWhat / (np.sqrt(vWhat) + eps)

        # Adam for b
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * (db * db)
        mbhat = self.mb / (1 - beta1 ** self.t)
        vbhat = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mbhat / (np.sqrt(vbhat) + eps)

        # backprop
        return np.dot(output_error, self.W.T)

    def __call__(self, x): return self.forward(x)

class Dropout:
    def __init__(self, keep_prob=0.9):
        self.keep_prob = keep_prob
        self.mask = None
        self.training = True
    def forward(self, x):
        if not self.training or self.keep_prob >= 1.0:
            return x
        self.mask = (np.random.rand(*x.shape) < self.keep_prob).astype(np.float32) / self.keep_prob
        return x * self.mask
    def backward(self, output_error, **kwargs):
        if not self.training or self.keep_prob >= 1.0:
            return output_error
        return output_error * self.mask
    def __call__(self, x): return self.forward(x)

class Network:
    def __init__(self, input_dim, output_dim, lr=1e-3, weight_decay=1e-4, keep_prob=0.9):
        self.layers = [
            Linear(input_dim, 1024, name="lin1"),
            Activation(LeakyReLU(), name="relu1"),
            Dropout(keep_prob),
            Linear(1024, 512, name="lin2"),
            Activation(LeakyReLU(), name="relu2"),
            Dropout(keep_prob),
            Linear(512, output_dim, name="out"),
            Activation(Softmax(), name="softmax"),
        ]
        self.lr = lr
        self.weight_decay = weight_decay
    def set_training(self, flag: bool):
        for l in self.layers:
            if isinstance(l, Dropout): l.training = flag
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, lr=self.lr, weight_decay=self.weight_decay)
    def __call__(self, x): return self.forward(x)

# train / eval
def main():
    np.random.seed(42)

    print("Fetching MNIST from OpenML ...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    # 10k test split
    X_train, X_test, y_train_idx, y_test_idx = train_test_split(
        X, y, test_size=10000, stratify=y, random_state=42
    )

    # standardize per-pixel (fit on train only)
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mu) / sigma
    X_test  = (X_test  - mu) / sigma

    y_train = to_categorical(y_train_idx)
    y_test  = to_categorical(y_test_idx)

    model = Network(28*28, 10, lr=1e-3, weight_decay=1e-4, keep_prob=0.9)
    criterion = CrossEntropy()

    EPOCHS, BATCH = 25, 512
    for epoch in range(EPOCHS):
        model.set_training(True)
        losses, accs = [], []
        for xb, yb in batch_loader(X_train, y_train, batch_size=BATCH, shuffle=True):
            out = model(xb)
            losses.append(np.mean(criterion.loss(yb, out)))
            accs.append(accuracy(np.argmax(yb, 1), np.argmax(out, 1)))
            error = (out - yb) / xb.shape[0]  # softmax+CE gradient wrt logits
            model.backward(error)

        # mild LR decay
        if epoch in {15, 20}:
            model.lr *= 0.5

        print(f"Epoch {epoch+1:2d}: Loss={np.mean(losses):.4f}  Acc={np.mean(accs):.4f}")

    # evaluate
    model.set_training(False)
    out_test = model(X_test)
    test_acc = accuracy(np.argmax(y_test, 1), np.argmax(out_test, 1))
    print(f"Test accuracy: {test_acc:.4f}")

    save_model(model, mu, sigma, "mnist_weights.npz")
    print("Saved weights (and mu/sigma) to mnist_weights.npz")

# save/load
def save_model(model, mu, sigma, path="mnist_weights.npz"):
    layers = [l for l in model.layers if isinstance(l, Linear)]
    np.savez(path,
             mu=mu.astype(np.float32),
             sigma=sigma.astype(np.float32),
             **{f"W{i}": L.W for i, L in enumerate(layers)},
             **{f"b{i}": L.b for i, L in enumerate(layers)})

def load_model_into(model, path="mnist_weights.npz"):
    data = np.load(path)
    layers = [l for l in model.layers if isinstance(l, Linear)]
    for i, L in enumerate(layers):
        L.W = data[f"W{i}"].astype(np.float32)
        L.b = data[f"b{i}"].astype(np.float32)

def load_model(input_dim=28*28, output_dim=10, path="mnist_weights.npz"):
    m = Network(input_dim, output_dim, lr=0.0, weight_decay=0.0, keep_prob=1.0)
    load_model_into(m, path)
    m.set_training(False)
    return m

if __name__ == "__main__":
    main()