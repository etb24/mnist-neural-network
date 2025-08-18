import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

from main import load_model

class MnistCanvasApp:
    def __init__(self, root, weights_path="mnist_weights.npz"):
        self.root = root
        self.root.title("CANVAS")

        # MNIST-like buffer (28×28, black bg, white strokes)
        self.H = self.W = 28
        self.scale = 10
        self.buf = np.zeros((self.H, self.W), dtype=np.float32)

        # MNIST-like brush (Gaussian stamp)
        self.kernel = self._gaussian_kernel(sigma=0.6, radius=3)  # ~7×7 soft stamp
        self.step_px = 0.35  # stamp every ~0.35 MNIST pixels along path

        # UI
        self.canvas = tk.Canvas(
            root,
            width=self.W*self.scale,
            height=self.H*self.scale,
            highlightthickness=0,
            bg="#000000"
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        tk.Button(root, text="Predict", command=self.predict, width=12)\
            .grid(row=1, column=0, padx=10, pady=6, sticky="ew")
        tk.Button(root, text="Clear",   command=self.clear,   width=12)\
            .grid(row=1, column=1, padx=10, pady=6, sticky="ew")
        self.lbl = tk.Label(root, text="Prediction: —", font=("Segoe UI", 14))
        self.lbl.grid(row=1, column=2, padx=10, pady=6, sticky="w")

        # mouse interaction
        self.last_u = self.last_v = None
        self.canvas.bind("<ButtonPress-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._up)

        if not os.path.exists(weights_path):
            messagebox.showinfo("Missing weights",
                                f"Could not find {weights_path}.\nTrain first to create it.")
        self.model = load_model(28*28, 10, weights_path)
        self.mu, self.sigma = None, None
        try:
            d = np.load(weights_path)
            if "mu" in d and "sigma" in d:
                self.mu = d["mu"].astype(np.float32)
                self.sigma = d["sigma"].astype(np.float32)
        except Exception:
            pass

        # initial render
        self._render()

    # MNIST-like brush helpers
    def _gaussian_kernel(self, sigma=0.9, radius=3):
        """Create a small normalized Gaussian stamp; peak=0.9 (stroke strength)."""
        ax = np.arange(-radius, radius+1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)).astype(np.float32)
        ker /= ker.max()  # peak = 1
        return ker * 0.9   # stroke intensity

    def _paint(self, u, v):
        """Add the Gaussian stamp centered at (u,v) in MNIST pixel coords (float)."""
        h, w = self.buf.shape
        kh, kw = self.kernel.shape
        x0 = int(round(u)) - kw//2
        y0 = int(round(v)) - kh//2

        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(w, x0+kw), min(h, y0+kh)
        if xs0 >= xs1 or ys0 >= ys1:
            return

        kx0, ky0 = xs0 - x0, ys0 - y0
        kx1, ky1 = kx0 + (xs1 - xs0), ky0 + (ys1 - ys0)

        self.buf[ys0:ys1, xs0:xs1] = np.clip(
            self.buf[ys0:ys1, xs0:xs1] + self.kernel[ky0:ky1, kx0:kx1],
            0.0, 1.0
        )

    def _paint_line(self, u0, v0, u1, v1):
        """Stamp along the mouse path so fast strokes don’t leave gaps."""
        du, dv = (u1 - u0), (v1 - v0)
        dist = max(abs(du), abs(dv))
        steps = max(1, int(dist / self.step_px))
        for t in np.linspace(0.0, 1.0, steps+1):
            self._paint(u0 + t*du, v0 + t*dv)

    # mouse handlers
    def _down(self, e):
        u, v = e.x / self.scale, e.y / self.scale
        self._paint(u, v)
        self.last_u, self.last_v = u, v
        self._render()

    def _move(self, e):
        if self.last_u is None:
            return
        u, v = e.x / self.scale, e.y / self.scale
        self._paint_line(self.last_u, self.last_v, u, v)
        self.last_u, self.last_v = u, v
        self._render()

    def _up(self, _):
        self.last_u = self.last_v = None

    # rendering
    def _render(self):
        img = Image.fromarray((self.buf * 255.0).astype(np.uint8))
        zoom = img.resize((self.W*self.scale, self.H*self.scale), Image.NEAREST)
        self.tkimg = ImageTk.PhotoImage(zoom)  # keep ref!
        self.canvas.create_image(0, 0, image=self.tkimg, anchor="nw")

    def clear(self):
        self.buf[...] = 0.0
        self.lbl.config(text="Prediction: —")
        self._render()

    def predict(self):
        x = self.buf.reshape(1, -1).astype(np.float32)  # 0..1 white-on-black
        if self.mu is not None and self.sigma is not None:
            x = (x - self.mu) / (self.sigma + 1e-8)
        probs = self.model(x)[0]
        self.lbl.config(text=f"Prediction: {int(np.argmax(probs))}")

def main():
    root = tk.Tk()
    app = MnistCanvasApp(root, weights_path="mnist_weights.npz")
    root.mainloop()

if __name__ == "__main__":
    main()