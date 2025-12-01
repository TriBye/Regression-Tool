import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")


def _make_regression(kind: str, n_samples: int, n_features: int, noise: float, imbalance: float, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))

    if kind == "Logistic Regression":
        w = rng.normal(0, 1, size=(n_features,))
        logits = X @ w
        logits = logits + rng.normal(0, noise, size=logits.shape)
        threshold = np.quantile(logits, imbalance)
        y = (logits > threshold).astype(int)
        return X, y

    if kind == "Quadratic Regression":
        w = rng.normal(0, 1, size=(n_features,))
        quad = np.sum((X**2) * w, axis=1)
        y = quad + rng.normal(0, noise, size=n_samples)
        return X, y

    if kind == "Cubic Regression":
        w = rng.normal(0, 1, size=(n_features,))
        cubic = np.sum((X**3) * w, axis=1)
        y = cubic + rng.normal(0, noise, size=n_samples)
        return X, y

    if kind == "Logarithmic Regression":
        base = np.sum(np.sign(X) * np.log1p(np.abs(X)), axis=1)
        y = base + rng.normal(0, noise, size=n_samples)
        return X, y

    if kind == "Exponential Regression":
        lin = np.sum(X, axis=1)
        y = np.exp(np.clip(lin, -4, 4)) + rng.normal(0, noise, size=n_samples)
        return X, y

    if kind == "Trigonometric Regression":
        weights = rng.uniform(0.5, 1.5, size=n_features)
        trig = np.sum(np.sin(X * weights) + np.cos(X * weights * 0.5), axis=1)
        y = trig + rng.normal(0, noise, size=n_samples)
        return X, y

    # default linear / ridge / lasso
    w = rng.normal(0, 1, size=(n_features,))
    y = X @ w + rng.normal(0, noise, size=n_samples)
    return X, y


def launch_generator_window(parent=None, data_dir: str | None = None) -> None:
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Dataset Generator")
    win.geometry("900x650")

    params = {
        "model_type": tk.StringVar(value="Linear Regression"),
        "samples": tk.StringVar(value="300"),
        "features": tk.StringVar(value="2"),
        "noise": tk.StringVar(value="0.5"),
        "imbalance": tk.StringVar(value="0.5"),
        "seed": tk.StringVar(value="42"),
        "filename": tk.StringVar(value="generated.csv"),
    }

    top = tk.Frame(win, padx=10, pady=10)
    top.pack(side=tk.TOP, fill="x")
    tk.Label(top, text="Model type:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(
        top,
        textvariable=params["model_type"],
        state="readonly",
        values=[
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Random Forest Regression",
            "Quadratic Regression",
            "Cubic Regression",
            "Logarithmic Regression",
            "Exponential Regression",
            "Trigonometric Regression",
            "Logistic Regression",
        ],
        width=25,
    ).grid(row=0, column=1, sticky="w", padx=4)

    tk.Label(top, text="Samples:").grid(row=1, column=0, sticky="w")
    tk.Entry(top, textvariable=params["samples"], width=10).grid(row=1, column=1, sticky="w", padx=4)

    tk.Label(top, text="Features (dims):").grid(row=2, column=0, sticky="w")
    tk.Entry(top, textvariable=params["features"], width=10).grid(row=2, column=1, sticky="w", padx=4)

    tk.Label(top, text="Noise / spread:").grid(row=3, column=0, sticky="w")
    tk.Entry(top, textvariable=params["noise"], width=10).grid(row=3, column=1, sticky="w", padx=4)

    tk.Label(top, text="Imbalance (logistic):").grid(row=4, column=0, sticky="w")
    tk.Entry(top, textvariable=params["imbalance"], width=10).grid(row=4, column=1, sticky="w", padx=4)

    tk.Label(top, text="Seed:").grid(row=5, column=0, sticky="w")
    tk.Entry(top, textvariable=params["seed"], width=10).grid(row=5, column=1, sticky="w", padx=4)

    tk.Label(top, text="Filename:").grid(row=6, column=0, sticky="w")
    tk.Entry(top, textvariable=params["filename"], width=20).grid(row=6, column=1, sticky="w", padx=4)

    btns = tk.Frame(top)
    btns.grid(row=7, column=0, columnspan=3, pady=6, sticky="w")
    status_var = tk.StringVar(value="Generate a dataset to preview.")

    figure = Figure(figsize=(6, 4), dpi=100)
    ax = figure.add_subplot(111)
    canvas = FigureCanvasTkAgg(figure, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill="both", expand=True)

    def generate(_event=None):
        try:
            n_samples = int(float(params["samples"].get()))
            n_features = int(float(params["features"].get()))
            noise = float(params["noise"].get())
            imb = float(params["imbalance"].get())
            seed = int(float(params["seed"].get()))
        except ValueError:
            messagebox.showerror("Generator", "Please check numeric inputs.")
            return
        kind = params["model_type"].get()
        X, y = _make_regression(kind, n_samples, n_features, noise, imb, seed)
        df_cols = [f"f{i+1}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=df_cols)
        df["target"] = y
        win.generated_df = df  # type: ignore[attr-defined]

        ax.clear()
        if n_features >= 2:
            scatter = ax.scatter(df_cols and df[df_cols[0]], df[df_cols[1]], c=y, cmap="viridis", alpha=0.7)
            figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel(df_cols[0])
            ax.set_ylabel(df_cols[1])
        else:
            ax.scatter(df_cols and df[df_cols[0]], y, c="tab:blue", alpha=0.7)
            ax.set_xlabel(df_cols[0])
            ax.set_ylabel("target")
        ax.set_title(f"{kind} | samples={n_samples}, features={n_features}")
        canvas.draw_idle()
        status_var.set(f"Generated dataset with shape {df.shape}.")

    def save():
        if not hasattr(win, "generated_df"):
            messagebox.showinfo("Generator", "Generate a dataset first.")
            return
        df: pd.DataFrame = getattr(win, "generated_df")
        target_dir = data_dir or os.getcwd()
        os.makedirs(target_dir, exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            initialdir=target_dir,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=params["filename"].get(),
            title="Save generated dataset",
        )
        if not file_path:
            return
        df.to_csv(file_path, index=False)
        status_var.set(f"Saved to {file_path}")

    tk.Button(btns, text="Generate", command=generate).pack(side=tk.LEFT, padx=4)
    tk.Button(btns, text="Save CSV", command=save).pack(side=tk.LEFT, padx=4)
    tk.Label(top, textvariable=status_var, fg="blue").grid(row=8, column=0, columnspan=3, sticky="w")

    win.bind("<Return>", generate)
    if not parent:
        win.mainloop()


if __name__ == "__main__":
    launch_generator_window(None, os.path.join(os.getcwd(), "Data"))
