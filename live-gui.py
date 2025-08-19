#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk

from main import load_model

class MnistCanvasApp:
    def __init__(self, root, weights_path="mnist_weights.npz"):
        self.root = root
        self.root.title("MNIST – Live Prediction")

        # 28×28 MNIST buffer (black bg, white strokes)
        self.H = self.W = 28
        self.scale = 10
        self.buf = np.zeros((self.H, self.W), dtype=np.float32)

        # MNIST-like brush (Gaussian stamp)
        self.base_sigma = 0.6 # width
        self.kernel = self._gaussian_kernel(self.base_sigma)  # auto radius
        self.step_px = 0.2 

        # layout: canvas (left), sidebar (right)
        self.canvas = tk.Canvas(
            root, width=self.W*self.scale, height=self.H*self.scale,
            highlightthickness=0, bg="#000000"
        )
        self.canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="n")

        sidebar = tk.Frame(root)
        sidebar.grid(row=0, column=1, padx=10, pady=10, sticky="ns")

        # buttons under canvas
        btns = tk.Frame(root)
        btns.grid(row=1, column=0, padx=10, pady=(0,10), sticky="ew")
        btns.columnconfigure(0, weight=1)
        tk.Button(btns, text="Clear", command=self.clear)\
            .grid(row=0, column=0, padx=5, sticky="ew")

        # confidence bars (0–9)
        tk.Label(sidebar, text="Confidences", font=("Segoe UI", 12, "bold"))\
            .grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,5))

        self.digit_labels = []
        self.prog_vars = []
        self.perc_labels = []

        for d in range(10):
            dlbl = tk.Label(sidebar, text=str(d))
            dlbl.grid(row=1+d, column=0, sticky="e", padx=(0,6))
            var = tk.DoubleVar(value=0.0)
            pb = ttk.Progressbar(sidebar, variable=var, maximum=1.0, length=160)
            pb.grid(row=1+d, column=1, sticky="we", pady=1)
            pct = tk.Label(sidebar, text="0%")
            pct.grid(row=1+d, column=2, sticky="w", padx=(6,0))

            self.digit_labels.append(dlbl)
            self.prog_vars.append(var)
            self.perc_labels.append(pct)

        # final prediction label
        self.pred_lbl = tk.Label(sidebar, text="Prediction: —", font=("Segoe UI", 14))
        self.pred_lbl.grid(row=12, column=0, columnspan=3, sticky="w", pady=(10,0))

        # mouse events
        self.last_u = self.last_v = None
        self.canvas.bind("<ButtonPress-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._up)
        # quick key to clear
        self.root.bind("c", lambda _: self.clear())

        # load model + optional mu/sigma
        if not os.path.exists(weights_path):
            messagebox.showinfo("Missing weights", f"Could not find {weights_path}.\nTrain first to create it.")
        self.model = load_model(28*28, 10, weights_path)

        self.mu, self.sigma = None, None
        try:
            d = np.load(weights_path)
            if "mu" in d and "sigma" in d:
                self.mu = d["mu"].astype(np.float32)
                self.sigma = d["sigma"].astype(np.float32)
        except Exception:
            pass

        # initial render + prediction
        self._render()
        self._update_prediction()

    # brush and drawing
    def _gaussian_kernel(self, sigma=0.9):
        sigma = float(sigma)
        radius = max(1, int(np.ceil(3*sigma)))
        ax = np.arange(-radius, radius+1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)).astype(np.float32)
        ker /= max(ker.max(), 1e-8) # normalize peak=1
        return ker * 0.9 # stroke intensity

    def _paint(self, u, v):
        h, w = self.buf.shape
        kh, kw = self.kernel.shape
        x0 = int(round(u)) - kw//2
        y0 = int(round(v)) - kh//2
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(w, x0+kw), min(h, y0+kh)
        if xs0 >= xs1 or ys0 >= ys1: return
        kx0, ky0 = xs0 - x0, ys0 - y0
        kx1, ky1 = kx0 + (xs1 - xs0), ky0 + (ys1 - ys0)
        self.buf[ys0:ys1, xs0:xs1] = np.clip(
            self.buf[ys0:ys1, xs0:xs1] + self.kernel[ky0:ky1, kx0:kx1], 0.0, 1.0
        )

    def _paint_line(self, u0, v0, u1, v1):
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
        self._update_prediction()

    def _move(self, e):
        if self.last_u is None: return
        u, v = e.x / self.scale, e.y / self.scale
        self._paint_line(self.last_u, self.last_v, u, v)
        self.last_u, self.last_v = u, v
        self._render()
        self._update_prediction()

    def _up(self, _):
        self.last_u = self.last_v = None

    # rendering
    def _render(self):
        img = Image.fromarray((self.buf * 255.0).astype(np.uint8))
        zoom = img.resize((self.W*self.scale, self.H*self.scale), Image.NEAREST)
        self.tkimg = ImageTk.PhotoImage(zoom)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor="nw")

    # prediction pipeline
    def _prep_input(self):
        # copy buffer (0..1, white-on-black)
        arr = self.buf.copy()

        # center mass to the middle (MNIST-like centering)
        m = float(arr.sum())
        if m > 1e-6:
            ys, xs = np.indices(arr.shape, dtype=np.float32)
            cx = float((arr * xs).sum() / m)
            cy = float((arr * ys).sum() / m)
            tx, ty = (arr.shape[1]-1)/2.0, (arr.shape[0]-1)/2.0
            dx, dy = int(round(tx - cx)), int(round(ty - cy))
            shifted = np.zeros_like(arr)
            h, w = arr.shape
            sy0, sy1 = max(0, -dy), min(h, h - dy)
            ty0, ty1 = max(0,  dy), min(h, h + dy)
            sx0, sx1 = max(0, -dx), min(w, w - dx)
            tx0, tx1 = max(0,  dx), min(w, w + dx)
            shifted[ty0:ty1, tx0:tx1] = arr[sy0:sy1, sx0:sx1]
            arr = shifted

        x = arr.reshape(1, -1).astype(np.float32)

        # standardize exactly like training (if available)
        if self.mu is not None and self.sigma is not None:
            x = (x - self.mu) / (self.sigma + 1e-8)

        return x

    def _update_prediction(self):
        x = self._prep_input()
        probs = self.model(x)[0]
        self._update_bars(probs)
        pred = int(np.argmax(probs))
        self.pred_lbl.config(text=f"Prediction: {pred}")

    def _update_bars(self, probs):
        # normalize defensively
        s = float(np.sum(probs))
        p = probs if s == 0 else probs / s

        # find top class for highlighting
        top = int(np.argmax(p))
        for d in range(10):
            val = round(float(p[d]), 2)
            self.prog_vars[d].set(val)
            self.perc_labels[d].config(text=f"{val}")
            # bold the top class label
            self.digit_labels[d].config(font=("Segoe UI", 10, "bold" if d == top else "normal"))

    def clear(self):
        self.buf[...] = 0.0
        self.pred_lbl.config(text="Prediction: —")
        self._render()
        self._update_prediction()

def main():
    root = tk.Tk()
    app = MnistCanvasApp(root, weights_path="mnist_weights.npz")
    root.mainloop()

if __name__ == "__main__":
    main()
