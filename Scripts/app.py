import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

from Scripts.tooltips import Tooltip
from Scripts.generator import launch_generator_window

matplotlib.use("TkAgg")
matplotlib.rcParams.update(
    {
        "axes.grid": True,
        "axes.facecolor": "#fffbfb",
        "figure.facecolor": "white",
        "font.size": 9,
    }
)


class RegressionToolApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Regression Tool")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = int(screen_w * 0.85)
        window_h = int(screen_h * 0.85)
        self.root.geometry(f"{window_w}x{window_h}")
        self.root.minsize(int(screen_w * 0.6), int(screen_h * 0.6))
        self.left_panel_width = max(480, int(window_w * 0.33))

        self.df: pd.DataFrame | None = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder: LabelEncoder | None = None
        self.training_history: list[tuple[int, float]] = []
        self.task_type: str = "regression"
        self.last_eval: dict | None = None
        self.last_train_data: dict | None = None

        self.dataset_var = tk.StringVar(value=os.path.join("Data", ""))
        self.model_var = tk.StringVar(value="Linear Regression")
        self.target_var = tk.StringVar(value="")
        self.plot_feature_var = tk.StringVar(value="")
        self.plot_type_var = tk.StringVar(value="Model Fit")

        self.epochs_var = tk.StringVar(value="40")
        self.lr_var = tk.StringVar(value="0.01")
        self.alpha_var = tk.StringVar(value="0.0001")
        self.random_state_var = tk.StringVar(value="42")
        self.n_estimators_var = tk.StringVar(value="120")
        self.max_depth_var = tk.StringVar(value="")
        self.test_size_var = tk.StringVar(value="0.2")
        self.sample_var = tk.StringVar(value="")
        self.cv_folds_var = tk.StringVar(value="0")
        self.standardize_var = tk.BooleanVar(value=True)
        self.skip_first_row_var = tk.BooleanVar(value=False)
        self.test_size_var.trace_add("write", lambda *_: self._update_train_test_info())
        self.random_state_var.trace_add("write", lambda *_: self._update_train_test_info())

        self.status_var = tk.StringVar(value="Load a dataset to begin.")
        self.test_status_var = tk.StringVar(value="Waiting for a trained model to test.")
        self.dataset_info_var = tk.StringVar(value="No dataset loaded.")
        self.train_test_info_var = tk.StringVar(value="Train/Test split: -")

        self.feature_listbox: tk.Listbox | None = None
        self.available_features: list[str] = []
        self.feature_nav_label: tk.Label | None = None
        self.metrics_text: tk.Text | None = None
        self.progress_var = tk.DoubleVar(value=0.0)
        self.colorbar = None
        self.feature_transformer = None

        self._build_layout()
        self.root.bind("<Left>", lambda event: self._shift_feature(-1))
        self.root.bind("<Right>", lambda event: self._shift_feature(1))
        self._toggle_param_visibility()
        self.update_controls_state()

    def _attach_tooltip(self, widget, text: str) -> None:
        Tooltip(widget, text)

    def update_controls_state(self) -> None:
        dataset_loaded = self.df is not None and not self.df.empty
        model_ready = self.model is not None
        train_state = tk.NORMAL if dataset_loaded else tk.DISABLED
        eval_state = tk.NORMAL if dataset_loaded and model_ready else tk.DISABLED
        predict_state = tk.NORMAL if model_ready else tk.DISABLED
        for btn in ("btn_train", "btn_min_loss"):
            if hasattr(self, btn):
                getattr(self, btn).configure(state=train_state)
        if hasattr(self, "btn_eval"):
            self.btn_eval.configure(state=eval_state)
        if hasattr(self, "btn_predict"):
            self.btn_predict.configure(state=predict_state)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1, minsize=self.left_panel_width)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        left_container = tk.Frame(self.root)
        left_container.grid(row=0, column=0, sticky="nsew")
        left_container.rowconfigure(0, weight=1)
        left_container.columnconfigure(0, weight=1)
        left_canvas = tk.Canvas(left_container, borderwidth=0, highlightthickness=0)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scrollbar = tk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_scrollbar.grid(row=0, column=1, sticky="ns")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left = tk.Frame(left_canvas, padx=10, pady=10)
        left_canvas.create_window((0, 0), window=left, anchor="nw")
        def _sync_left_width(_event=None) -> None:
            target_width = max(self.left_panel_width, left_container.winfo_width())
            left_canvas.configure(scrollregion=left_canvas.bbox("all"), width=target_width)
        left.bind("<Configure>", _sync_left_width)
        left_container.bind("<Configure>", _sync_left_width)
        _sync_left_width()
        left.columnconfigure(0, weight=1)

        right = tk.Frame(self.root, padx=10, pady=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)

        self._build_model_section(left)
        self._build_data_info_section(left)
        self._build_testing_section(left)
        self._build_training_section(left)
        self._build_graph_section(right)
        self._build_metrics_section(right)

    def _build_model_section(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Model", padx=8, pady=8)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        tk.Label(frame, text="Dataset (csv from Data):").grid(row=0, column=0, sticky="w")
        dataset_entry = tk.Entry(frame, textvariable=self.dataset_var)
        dataset_entry.grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=4)
        tk.Button(frame, text="Load", command=self.load_dataset).grid(row=0, column=3, padx=4)
        self._attach_tooltip(dataset_entry, "Path to CSV (use files under Data/)")
        tk.Button(frame, text="Generate dataset", command=self.open_generator).grid(row=0, column=4, padx=4)
        tk.Checkbutton(frame, text="Ignore first row", variable=self.skip_first_row_var).grid(
            row=1, column=1, sticky="w", pady=4
        )

        tk.Label(frame, text="Model type:").grid(row=1, column=0, sticky="w", pady=4)
        model_options = [
            "Linear Regression",
            "Logistic Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Random Forest Regression",
            "Quadratic Regression",
            "Cubic Regression",
            "Logarithmic Regression",
            "Exponential Regression",
            "Trigonometric Regression",
        ]
        model_cb = ttk.Combobox(frame, textvariable=self.model_var, values=model_options, state="readonly")
        model_cb.grid(row=1, column=1, sticky="ew", padx=4)
        model_cb.bind("<<ComboboxSelected>>", lambda _e: (self._toggle_param_visibility(), self._update_dataset_info()))

        tk.Label(frame, text="Target column:").grid(row=2, column=0, sticky="w", pady=4)
        self.target_dropdown = ttk.Combobox(frame, textvariable=self.target_var, state="readonly")
        self.target_dropdown.grid(row=2, column=1, sticky="ew", padx=4)
        self.target_dropdown.bind(
            "<<ComboboxSelected>>", lambda _e: (self._update_dataset_info(), self._update_graph(None, None, None))
        )

        tk.Label(frame, text="Feature columns:").grid(row=3, column=0, sticky="nw", pady=4)
        self.feature_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        self.feature_listbox.grid(row=3, column=1, columnspan=1, sticky="nsew", padx=4)

        tk.Label(frame, text="Plot feature:").grid(row=4, column=0, sticky="w", pady=4)
        self.plot_feature_dropdown = ttk.Combobox(frame, textvariable=self.plot_feature_var, state="readonly")
        self.plot_feature_dropdown.grid(row=4, column=1, sticky="ew", padx=4)
        self.plot_feature_dropdown.bind("<<ComboboxSelected>>", lambda _event: self._update_graph(None, None, None))
        self._attach_tooltip(self.plot_feature_dropdown, "Feature to visualize against target")

        action_frame = tk.Frame(frame)
        action_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=6)
        tk.Button(action_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=4)
        tk.Button(action_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=4)
        tk.Button(action_frame, text="Save Params", command=self.save_params).pack(side=tk.LEFT, padx=4)
        tk.Button(action_frame, text="Load Params", command=self.load_params).pack(side=tk.LEFT, padx=4)
        tk.Button(action_frame, text="Save Session", command=self.save_session).pack(side=tk.LEFT, padx=4)
        tk.Button(action_frame, text="Load Session", command=self.load_session).pack(side=tk.LEFT, padx=4)

        tk.Label(frame, textvariable=self.status_var, fg="blue").grid(
            row=6, column=0, columnspan=4, sticky="w", pady=4
        )

    def _build_testing_section(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Testing", padx=8, pady=8)
        frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        tk.Label(frame, text="Test size (0-1):").grid(row=0, column=0, sticky="w")
        entry_test = tk.Entry(frame, textvariable=self.test_size_var, width=10)
        entry_test.grid(row=0, column=1, sticky="w", padx=4)
        self._attach_tooltip(entry_test, "Fraction of data held out for testing")
        self.btn_eval = tk.Button(frame, text="Evaluate", command=self.evaluate_model, name="btn_eval")
        self.btn_eval.grid(row=0, column=2, padx=4)

        tk.Label(frame, text="Custom sample (comma-separated):").grid(row=1, column=0, sticky="w", pady=4)
        tk.Entry(frame, textvariable=self.sample_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=4)
        self.btn_predict = tk.Button(frame, text="Predict sample", command=self.predict_sample, name="btn_predict")
        self.btn_predict.grid(row=1, column=3, padx=4)
        self._attach_tooltip(self.btn_predict, "Predict using the trained or loaded model")

        tk.Label(frame, textvariable=self.test_status_var, fg="purple").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=4
        )

    def _build_data_info_section(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Data & Info", padx=8, pady=8)
        frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        tk.Label(frame, textvariable=self.dataset_info_var, justify="left").grid(row=0, column=0, sticky="w")
        tk.Label(frame, textvariable=self.train_test_info_var, justify="left").grid(row=1, column=0, sticky="w", pady=4)
        btns = tk.Frame(frame)
        btns.grid(row=2, column=0, sticky="w")
        tk.Button(btns, text="Preview data", command=self.preview_data).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Update info", command=self._update_dataset_info).pack(side=tk.LEFT, padx=4)

    def _build_training_section(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Training", padx=8, pady=8)
        frame.grid(row=3, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        tk.Label(frame, text="Epochs:").grid(row=0, column=0, sticky="w")
        self.entry_epochs = tk.Entry(frame, textvariable=self.epochs_var, width=8)
        self.entry_epochs.grid(row=0, column=1, sticky="w", padx=4)
        self._attach_tooltip(self.entry_epochs, "Number of passes over the data (SGD models)")

        tk.Label(frame, text="Learning rate:").grid(row=1, column=0, sticky="w")
        self.entry_lr = tk.Entry(frame, textvariable=self.lr_var, width=8)
        self.entry_lr.grid(row=1, column=1, sticky="w", padx=4)
        self._attach_tooltip(self.entry_lr, "Step size for SGD-based models")

        tk.Label(frame, text="Regularization (alpha):").grid(row=2, column=0, sticky="w")
        self.entry_alpha = tk.Entry(frame, textvariable=self.alpha_var, width=8)
        self.entry_alpha.grid(row=2, column=1, sticky="w", padx=4)
        self._attach_tooltip(self.entry_alpha, "Regularization strength (SGD/Lasso/Ridge)")

        tk.Label(frame, text="n_estimators (RF):").grid(row=0, column=2, sticky="w")
        self.entry_estimators = tk.Entry(frame, textvariable=self.n_estimators_var, width=10)
        self.entry_estimators.grid(row=0, column=3, sticky="w", padx=4)
        self._attach_tooltip(self.entry_estimators, "Number of trees for Random Forest")

        tk.Label(frame, text="max_depth (RF):").grid(row=1, column=2, sticky="w")
        self.entry_depth = tk.Entry(frame, textvariable=self.max_depth_var, width=10)
        self.entry_depth.grid(row=1, column=3, sticky="w", padx=4)
        self._attach_tooltip(self.entry_depth, "Maximum depth of trees (leave blank for full depth)")

        tk.Label(frame, text="Random state:").grid(row=2, column=2, sticky="w")
        self.entry_rs = tk.Entry(frame, textvariable=self.random_state_var, width=10)
        self.entry_rs.grid(row=2, column=3, sticky="w", padx=4)
        self._attach_tooltip(self.entry_rs, "Seed for reproducibility")

        tk.Label(frame, text="CV folds (0=off):").grid(row=3, column=0, sticky="w")
        self.entry_cv = tk.Entry(frame, textvariable=self.cv_folds_var, width=8)
        self.entry_cv.grid(row=3, column=1, sticky="w", padx=4)
        self._attach_tooltip(self.entry_cv, "Cross-validation folds; 0 disables CV summary")

        tk.Checkbutton(frame, text="Standardize features", variable=self.standardize_var).grid(
            row=3, column=2, sticky="w"
        )

        btn_reset = tk.Button(frame, text="Reset to defaults", command=self.reset_defaults)
        btn_reset.grid(row=4, column=0, sticky="w", pady=4)
        self._attach_tooltip(btn_reset, "Restore training parameters to default values")

        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=4, pady=6)
        self.btn_min_loss = tk.Button(btn_frame, text="Calculate minimal loss", command=self.calculate_min_loss, name="btn_min_loss")
        self.btn_min_loss.pack(side=tk.LEFT, padx=4)
        self.btn_train = tk.Button(btn_frame, text="Train", command=self.train_model, name="btn_train")
        self.btn_train.pack(side=tk.LEFT, padx=4)
        self._attach_tooltip(self.btn_min_loss, "Run a quick grid search over lr/alpha or RF params")
        self._attach_tooltip(self.btn_train, "Train the selected model with current parameters")

        progress = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        progress.grid(row=6, column=0, columnspan=4, sticky="ew", pady=4)

    def _build_graph_section(self, parent: tk.Frame) -> None:
        header = tk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header.columnconfigure(2, weight=1)
        tk.Label(header, text="Graphs", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w")
        nav = tk.Frame(header)
        nav.grid(row=0, column=1, sticky="e")
        tk.Button(nav, text="◀", width=3, command=lambda: self._shift_feature(-1)).pack(side=tk.LEFT, padx=2)
        self.feature_nav_label = tk.Label(nav, textvariable=self.plot_feature_var, width=20, anchor="center")
        self.feature_nav_label.pack(side=tk.LEFT, padx=4)
        tk.Button(nav, text="▶", width=3, command=lambda: self._shift_feature(1)).pack(side=tk.LEFT, padx=2)

        plot_selector = tk.Frame(header)
        plot_selector.grid(row=0, column=2, sticky="e")
        tk.Label(plot_selector, text="Plot:").pack(side=tk.LEFT)
        plot_cb = ttk.Combobox(
            plot_selector,
            textvariable=self.plot_type_var,
            state="readonly",
            values=[
                "Model Fit",
                "Loss Curve",
                "Residuals",
                "Confusion Matrix",
                "ROC Curve",
                "Precision-Recall",
                "Feature Importance",
            ],
            width=18,
        )
        plot_cb.pack(side=tk.LEFT, padx=4)
        plot_cb.bind("<<ComboboxSelected>>", lambda _e: self._update_graph(None, None, None))
        tk.Button(plot_selector, text="Save plot", command=self.save_plot).pack(side=tk.LEFT, padx=2)

        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.ax_reg = self.figure.add_subplot(121)
        self.ax_loss = self.figure.add_subplot(122)
        self.figure.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    def _build_metrics_section(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Metrics", padx=8, pady=8)
        frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        frame.columnconfigure(0, weight=1)
        self.metrics_text = tk.Text(frame, height=10, wrap="word")
        self.metrics_text.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        tk.Button(frame, text="Export classification report", command=self.export_classification_report).grid(
            row=1, column=0, sticky="w", pady=4
        )

    def browse_dataset(self) -> None:
        initial_dir = os.path.join(os.getcwd(), "Data")
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir, title="Select dataset", filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            self.dataset_var.set(os.path.relpath(file_path))

    def load_dataset(self) -> None:
        path = self.dataset_var.get().strip()
        if not path:
            messagebox.showwarning("Dataset", "Please provide a dataset path.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Dataset", f"Dataset not found: {path}")
            return
        try:
            skip_rows = 1 if self.skip_first_row_var.get() else 0
            self.df = pd.read_csv(path, skiprows=skip_rows)
        except Exception as exc:
            messagebox.showerror("Dataset", f"Could not load dataset: {exc}")
            return

        if self.df.empty:
            messagebox.showerror("Dataset", "Dataset is empty.")
            return

        columns = list(self.df.columns)
        self.target_dropdown["values"] = columns
        self.target_var.set(columns[-1])
        self._refresh_feature_list(columns)
        self.plot_feature_dropdown["values"] = columns[:-1]
        self.plot_feature_var.set(columns[0] if len(columns) > 1 else columns[-1])
        self.available_features = [c for c in columns if c != self.target_var.get()]

        self.status_var.set(f"Loaded {path} with {len(self.df)} rows and {len(columns)} columns.")
        self.training_history = []
        self.model = None
        self.progress_var.set(0)
        self.feature_transformer = None
        self._update_graph(self.available_features, None, None)
        self._update_dataset_info()
        self._update_train_test_info()
        self.update_controls_state()

    def _refresh_feature_list(self, columns: list[str]) -> None:
        if not self.feature_listbox:
            return
        self.feature_listbox.delete(0, tk.END)
        for col in columns:
            self.feature_listbox.insert(tk.END, col)
        # Preselect all except target
        target = self.target_var.get()
        for idx, col in enumerate(columns):
            if col != target:
                self.feature_listbox.selection_set(idx)
        self.available_features = [c for c in columns if c != target]

    def _update_train_test_info(self) -> None:
        try:
            test_size = float(self.test_size_var.get())
            train_size = 1 - test_size
            rs = self.random_state_var.get()
            self.train_test_info_var.set(f"Train/Test split: {train_size:.2f}/{test_size:.2f} | random_state={rs}")
        except Exception:
            self.train_test_info_var.set("Train/Test split: -")

    def _update_dataset_info(self) -> None:
        if self.df is None:
            self.dataset_info_var.set("No dataset loaded.")
            return
        rows, cols = self.df.shape
        info_lines = [f"Samples: {rows}", f"Columns: {cols}"]
        target = self.target_var.get()
        if target in self.df.columns:
            info_lines.append(f"Target: {target}")
            info_lines.append(f"Features: {max(cols-1, 0)}")
            if self.model_var.get() == "Logistic Regression":
                counts = self.df[target].value_counts()
                dist = ", ".join([f"{idx}:{cnt}" for idx, cnt in counts.items()])
                info_lines.append(f"Class dist: {dist}")
        self.dataset_info_var.set(" | ".join(info_lines))

    def _get_selected_features(self) -> list[str]:
        if not self.feature_listbox:
            return []
        selected_indices = self.feature_listbox.curselection()
        features = [self.feature_listbox.get(i) for i in selected_indices]
        if not features:
            # default to all except target
            features = [col for col in self.df.columns if col != self.target_var.get()]
        features = [f for f in features if f != self.target_var.get()]
        return features

    def _prepare_data(self):
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        target = self.target_var.get()
        if target not in self.df.columns:
            raise ValueError("Please select a target column.")
        features = self._get_selected_features()
        missing = [f for f in features if f not in self.df.columns]
        if missing:
            raise ValueError(f"Missing selected features: {missing}")
        data = self.df[features + [target]].copy()
        data = data.dropna()
        if data.empty:
            raise ValueError("No data left after dropping missing values.")

        for col in features:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna()
        if data.empty:
            raise ValueError("Features could not be converted to numeric.")

        X = data[features].values
        raw_y = data[target]

        model_type = self.model_var.get()
        if model_type == "Logistic Regression":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(raw_y.astype(str))
            self.task_type = "classification"
        else:
            y = pd.to_numeric(raw_y, errors="coerce")
            if y.isna().any():
                raise ValueError("Target could not be converted to numeric.")
            y = y.values
            self.label_encoder = None
            self.task_type = "regression"

        return X, y, features

    def _model_specific_transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        model_type = self.model_var.get()
        if model_type in ("Quadratic Regression", "Cubic Regression"):
            degree = 2 if model_type == "Quadratic Regression" else 3
            if (
                self.feature_transformer is None
                or not isinstance(self.feature_transformer, PolynomialFeatures)
                or getattr(self.feature_transformer, "degree", None) != degree
            ):
                transformer = PolynomialFeatures(degree=degree, include_bias=False)
                if fit:
                    X_transformed = transformer.fit_transform(X)
                    self.feature_transformer = transformer
                else:
                    X_transformed = transformer.fit_transform(X)
                    self.feature_transformer = transformer
            else:
                X_transformed = self.feature_transformer.transform(X)
            return X_transformed.astype(float)
        self.feature_transformer = None if model_type not in ("Quadratic Regression", "Cubic Regression") else self.feature_transformer
        if model_type == "Logarithmic Regression":
            return (np.sign(X) * np.log1p(np.abs(X))).astype(float)
        if model_type == "Exponential Regression":
            return np.exp(np.clip(X, -5, 5)).astype(float)
        if model_type == "Trigonometric Regression":
            sin_part = np.sin(X)
            cos_part = np.cos(X)
            return np.concatenate([sin_part, cos_part], axis=1).astype(float)
        return X.astype(float)

    def _pipeline_transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X_t = self._model_specific_transform(X, fit=fit)
        if not self.standardize_var.get():
            if fit:
                self.scaler = StandardScaler()
            return X_t
        if fit:
            self.scaler.fit(X_t)
        return self.scaler.transform(X_t)

    def _update_training_plot(self, axis=None) -> None:
        ax = axis or self.ax_loss
        ax.clear()
        if self.training_history:
            epochs, losses = zip(*self.training_history)
            ax.plot(epochs, losses, marker="o", color="tab:red")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss curve")
        else:
            ax.set_title("Loss curve")
        self.canvas.draw_idle()

    def train_model(self) -> None:
        if self.df is None:
            messagebox.showwarning("Training", "Load a dataset first.")
            return
        try:
            epochs = int(float(self.epochs_var.get()))
            lr = float(self.lr_var.get())
            alpha = float(self.alpha_var.get())
            random_state = int(float(self.random_state_var.get()))
            n_estimators = int(float(self.n_estimators_var.get()))
            max_depth = self.max_depth_var.get().strip()
            max_depth_int = int(float(max_depth)) if max_depth else None
        except ValueError:
            messagebox.showerror("Training", "Check training parameter formats.")
            return

        try:
            X, y, features = self._prepare_data()
        except Exception as exc:
            messagebox.showerror("Training", str(exc))
            return

        X_scaled = self._pipeline_transform(X, fit=True)
        model_type = self.model_var.get()
        self.training_history = []
        self.progress_var.set(0)
        total_steps = max(epochs, 1)
        self.last_eval = None

        try:
            if model_type in (
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Quadratic Regression",
                "Cubic Regression",
                "Logarithmic Regression",
                "Exponential Regression",
                "Trigonometric Regression",
            ):
                penalty = "l2" if model_type != "Lasso Regression" else "l1"
                adjusted_alpha = alpha if model_type != "Linear Regression" else max(alpha * 0.1, 1e-6)
                model = SGDRegressor(
                    penalty=penalty,
                    alpha=adjusted_alpha,
                    learning_rate="constant",
                    eta0=lr,
                    random_state=random_state,
                    max_iter=1,
                    tol=None,
                )
                self.model = model
                for epoch in range(epochs):
                    model.partial_fit(X_scaled, y)
                    preds = model.predict(X_scaled)
                    loss = mean_squared_error(y, preds)
                    self.training_history.append((epoch + 1, float(loss)))
                    self._update_training_plot()
                    self._update_graph(features, X_scaled, y)
                    self.progress_var.set(((epoch + 1) / total_steps) * 100)
                    self.root.update_idletasks()
            elif model_type == "Logistic Regression":
                classes = np.unique(y)
                if classes.size < 2:
                    raise ValueError("Logistic Regression needs at least 2 classes.")
                model = SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=alpha,
                    learning_rate="constant",
                    eta0=lr,
                    random_state=random_state,
                    max_iter=1,
                    tol=None,
                )
                self.model = model
                for epoch in range(epochs):
                    model.partial_fit(X_scaled, y, classes=classes)
                    prob = model.predict_proba(X_scaled)
                    loss = log_loss(y, prob, labels=classes)
                    self.training_history.append((epoch + 1, float(loss)))
                    self._update_training_plot()
                    self._update_graph(features, X_scaled, y)
                    self.progress_var.set(((epoch + 1) / total_steps) * 100)
                    self.root.update_idletasks()
            elif model_type == "Random Forest Regression":
                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth_int, random_state=random_state
                )
                model.fit(X_scaled, y)
                preds = model.predict(X_scaled)
                loss = mean_squared_error(y, preds)
                self.training_history.append((1, float(loss)))
                self.model = model
            self.status_var.set("Training complete.")
            self.last_train_data = {"features": features, "X": X_scaled, "y": y}
            self.progress_var.set(100)
            self._update_graph(features, X_scaled, y)
        except Exception as exc:
            messagebox.showerror("Training", f"Training failed: {exc}")
        self.update_controls_state()

    def evaluate_model(self) -> None:
        if self.model is None:
            messagebox.showinfo("Testing", "Train or load a model first.")
            return
        self._update_train_test_info()
        try:
            test_size = float(self.test_size_var.get())
        except ValueError:
            messagebox.showerror("Testing", "Test size must be a number between 0 and 1.")
            return
        try:
            X, y, features = self._prepare_data()
        except Exception as exc:
            messagebox.showerror("Testing", str(exc))
            return

        try:
            X_scaled = self._pipeline_transform(X, fit=False)
        except Exception:
            X_scaled = self._pipeline_transform(X, fit=True)

        stratify = y if self.task_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=int(float(self.random_state_var.get())), stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=int(float(self.random_state_var.get()))
            )

        metrics_lines: list[str] = []
        try:
            if self.task_type == "classification":
                y_pred = self.model.predict(X_test)
                probas = None
                if hasattr(self.model, "predict_proba"):
                    probas = self.model.predict_proba(X_test)
                acc = accuracy_score(y_test, y_pred)
                average_mode = "binary" if len(np.unique(y_test)) == 2 else "weighted"
                prec = precision_score(y_test, y_pred, average=average_mode, zero_division=0)
                rec = recall_score(y_test, y_pred, average=average_mode, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=average_mode, zero_division=0)
                metrics_lines.append(f"Accuracy: {acc:.3f}")
                metrics_lines.append(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
                if probas is not None and len(np.unique(y_test)) == 2:
                    auc = roc_auc_score(y_test, probas[:, 1])
                    metrics_lines.append(f"ROC-AUC: {auc:.3f}")
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(
                    y_test, y_pred, target_names=[str(c) for c in np.unique(y_test)], zero_division=0
                )
                metrics_lines.append("Confusion Matrix:\n" + str(cm))
                metrics_lines.append("Classification report available below.")
                cv_summary = self._run_cross_validation(X, y)
                if cv_summary:
                    metrics_lines.append(cv_summary)
                if self.metrics_text:
                    self.metrics_text.delete("1.0", tk.END)
                    self.metrics_text.insert(tk.END, "\n".join(metrics_lines) + "\n\n" + report)
                self.test_status_var.set(f"Accuracy: {acc:.3f} on {len(y_test)} samples.")
                self.last_eval = {
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "probas": probas,
                    "cm": cm,
                    "report": report,
                }
            else:
                preds = self.model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                metrics_lines.append(f"MSE: {mse:.4f}")
                metrics_lines.append(f"MAE: {mae:.4f}")
                metrics_lines.append(f"R²: {r2:.3f}")
                cv_summary = self._run_cross_validation(X, y)
                if cv_summary:
                    metrics_lines.append(cv_summary)
                if self.metrics_text:
                    self.metrics_text.delete("1.0", tk.END)
                    self.metrics_text.insert(tk.END, "\n".join(metrics_lines))
                self.test_status_var.set(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.3f} on {len(y_test)} samples.")
                residuals = y_test - preds
                self.last_eval = {"X_test": X_test, "y_test": y_test, "y_pred": preds, "residuals": residuals}
        except Exception as exc:
            messagebox.showerror("Testing", f"Evaluation failed: {exc}")
            return

        self._update_graph(features, X_scaled, y)
        self._update_train_test_info()
        self.update_controls_state()

    def predict_sample(self) -> None:
        if self.model is None:
            messagebox.showinfo("Predict", "Train or load a model first.")
            return
        try:
            X, y, features = self._prepare_data()
        except Exception as exc:
            messagebox.showerror("Predict", str(exc))
            return

        raw = self.sample_var.get()
        try:
            values = [float(v.strip()) for v in raw.split(",") if v.strip() != ""]
        except ValueError:
            messagebox.showerror("Predict", "Please enter numeric values separated by commas.")
            return
        if len(values) != len(features):
            messagebox.showerror("Predict", f"Expected {len(features)} values for features {features}.")
            return

        try:
            sample_scaled = self._pipeline_transform(np.array([values]), fit=False)
        except Exception:
            sample_scaled = self._pipeline_transform(np.array([values]), fit=True)
        try:
            pred = self.model.predict(sample_scaled)
        except Exception as exc:
            messagebox.showerror("Predict", f"Prediction failed: {exc}")
            return

        if self.task_type == "classification" and self.label_encoder is not None:
            decoded = self.label_encoder.inverse_transform(pred.astype(int))[0]
            self.test_status_var.set(f"Sample prediction: {decoded}")
        else:
            self.test_status_var.set(f"Sample prediction: {pred[0]:.4f}")

    def calculate_min_loss(self) -> None:
        if self.df is None:
            messagebox.showwarning("Search", "Load a dataset first.")
            return
        self._update_train_test_info()
        try:
            X, y, _ = self._prepare_data()
        except Exception as exc:
            messagebox.showerror("Search", str(exc))
            return

        try:
            test_size = float(self.test_size_var.get())
            random_state = int(float(self.random_state_var.get()))
        except ValueError:
            messagebox.showerror("Search", "Check test size/random state values.")
            return

        stratify = y if self.task_type == "classification" else None
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        X_train_scaled = self._pipeline_transform(X_train, fit=True)
        X_val_scaled = self._pipeline_transform(X_val, fit=False)

        model_type = self.model_var.get()
        best_loss = float("inf")
        best_params = {}

        if model_type == "Random Forest Regression":
            n_est_grid = [80, 120, 180]
            depth_grid = [None, 5, 10]
            for n in n_est_grid:
                for d in depth_grid:
                    model = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=random_state)
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_val_scaled)
                    loss = mean_squared_error(y_val, preds)
                    if loss < best_loss:
                        best_loss = loss
                        best_params = {"n_estimators": n, "max_depth": d}
            self.n_estimators_var.set(str(best_params["n_estimators"]))
            self.max_depth_var.set("" if best_params["max_depth"] is None else str(best_params["max_depth"]))
        else:
            lr_grid = [0.001, 0.01, 0.05]
            alpha_grid = [0.0001, 0.001, 0.01]
            for lr in lr_grid:
                for alpha in alpha_grid:
                    if model_type == "Logistic Regression":
                        classes = np.unique(y_train)
                        model = SGDClassifier(
                            loss="log_loss",
                            penalty="l2" if model_type == "Logistic Regression" else "l2",
                            alpha=alpha,
                            learning_rate="constant",
                            eta0=lr,
                            random_state=random_state,
                            max_iter=1,
                            tol=None,
                        )
                        for _ in range(10):
                            model.partial_fit(X_train_scaled, y_train, classes=classes)
                        prob = model.predict_proba(X_val_scaled)
                        loss = log_loss(y_val, prob, labels=classes)
                    else:
                        penalty = "l2" if model_type != "Lasso Regression" else "l1"
                        model = SGDRegressor(
                            penalty=penalty,
                            alpha=alpha if model_type != "Linear Regression" else max(alpha * 0.1, 1e-6),
                            learning_rate="constant",
                            eta0=lr,
                            random_state=random_state,
                            max_iter=1,
                            tol=None,
                        )
                        for _ in range(10):
                            model.partial_fit(X_train_scaled, y_train)
                        preds = model.predict(X_val_scaled)
                        loss = mean_squared_error(y_val, preds)
                    if loss < best_loss:
                        best_loss = loss
                        best_params = {"lr": lr, "alpha": alpha}
            self.lr_var.set(str(best_params.get("lr", self.lr_var.get())))
        self.alpha_var.set(str(best_params.get("alpha", self.alpha_var.get())))

        self.status_var.set(f"Best validation loss: {best_loss:.4f}")

    def _shift_feature(self, step: int) -> None:
        features = self._get_selected_features()
        if not features and self.available_features:
            features = self.available_features
        if not features and self.df is not None:
            features = [c for c in self.df.columns if c != self.target_var.get()]
        if not features:
            return
        current = self.plot_feature_var.get()
        if current not in features:
            current = features[0]
        new_idx = (features.index(current) + step) % len(features)
        self.plot_feature_var.set(features[new_idx])
        self._update_graph(features, None, None)

    def _toggle_param_visibility(self) -> None:
        model_type = self.model_var.get()
        rf = model_type == "Random Forest Regression"
        state_rf = tk.NORMAL if rf else tk.DISABLED
        state_sgd = tk.NORMAL if not rf else tk.DISABLED
        # reset transformer when model type changes
        self.feature_transformer = None
        for widget in (
            self.entry_estimators,
            self.entry_depth,
        ):
            widget.configure(state=state_rf)
        for widget in (
            self.entry_epochs,
            self.entry_lr,
            self.entry_alpha,
        ):
            widget.configure(state=state_sgd)

    def _update_graph(
        self, features: list[str] | None = None, X_scaled: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> None:
        self.ax_reg.clear()
        self.ax_loss.clear()
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        if self.df is None or self.df.empty:
            self.ax_reg.set_title("Regression plot")
            self.canvas.draw_idle()
            return

        plot_type = self.plot_type_var.get()
        if not features:
            features = self._get_selected_features()
        if not features:
            features = [c for c in self.df.columns if c != self.target_var.get()]
        if not features:
            self.ax_reg.set_title("Select at least one feature")
            self.canvas.draw_idle()
            return

        feature_for_plot = self.plot_feature_var.get()
        if feature_for_plot not in features:
            feature_for_plot = features[0]
            self.plot_feature_var.set(feature_for_plot)
        feature_idx = features.index(feature_for_plot)

        x_vals = self.df[feature_for_plot]
        try:
            x_vals = pd.to_numeric(x_vals, errors="coerce")
        except Exception:
            self.ax_reg.set_title("Selected feature not numeric")
            self.canvas.draw_idle()
            return

        if plot_type in ("Model Fit", "Residuals"):
            if x_vals.isna().all():
                self.ax_reg.set_title("Selected feature has no numeric values")
                self.canvas.draw_idle()
                return
            y_vals = pd.to_numeric(self.df[self.target_var.get()], errors="coerce")
            if y_vals.isna().all():
                self.ax_reg.set_title("Target is not numeric; model fit plot may be limited.")
            self.ax_reg.scatter(x_vals, y_vals, alpha=0.4, label="Data", color="tab:blue")

        if plot_type == "Model Fit":
            line_drawn = False
            if self.model is not None:
                xs = np.linspace(x_vals.min(), x_vals.max(), 80)
                numeric_feats = self.df[features].apply(pd.to_numeric, errors="coerce").dropna()
                if not numeric_feats.empty:
                    base = numeric_feats.mean().values
                    line_features = np.tile(base, (len(xs), 1))
                    line_features[:, feature_idx] = xs
                    try:
                        transformed = self._pipeline_transform(line_features, fit=False)
                        preds = self.model.predict(transformed)
                        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
                            prob = self.model.predict_proba(transformed)
                            target_col = 1 if prob.shape[1] > 1 else 0
                            preds = prob[:, target_col]
                        self.ax_reg.plot(xs, preds, color="tab:orange", label="Model prediction")
                        line_drawn = True
                    except Exception:
                        pass

            self.ax_reg.set_xlabel(feature_for_plot)
            self.ax_reg.set_ylabel(self.target_var.get())
            self.ax_reg.set_title("Model fit" if line_drawn else "Data preview")
            self.ax_reg.legend()
            self._update_training_plot()

        elif plot_type == "Residuals":
            residuals = None
            preds = None
            if self.last_eval and "residuals" in self.last_eval:
                residuals = self.last_eval["residuals"]
                preds = self.last_eval["y_pred"]
            elif self.last_train_data:
                preds = self.model.predict(self.last_train_data["X"])
                residuals = self.last_train_data["y"] - preds
            if residuals is not None and preds is not None:
                self.ax_reg.scatter(preds, residuals, alpha=0.6, color="tab:orange")
                self.ax_reg.axhline(0, color="gray", linestyle="--")
                self.ax_reg.set_xlabel("Predicted")
                self.ax_reg.set_ylabel("Residuals")
                self.ax_reg.set_title("Residual plot")
            else:
                self.ax_reg.set_title("Residuals unavailable. Evaluate the model first.")
            self._update_training_plot()

        elif plot_type == "Confusion Matrix":
            if self.last_eval and "cm" in self.last_eval:
                cm = self.last_eval["cm"]
                im = self.ax_reg.imshow(cm, cmap="Blues")
                self.colorbar = self.figure.colorbar(im, ax=self.ax_reg, fraction=0.046, pad=0.04)
                self.ax_reg.set_title("Confusion Matrix")
                classes = list(range(cm.shape[0]))
                self.ax_reg.set_xticks(classes)
                self.ax_reg.set_yticks(classes)
                self.ax_reg.set_xlabel("Predicted")
                self.ax_reg.set_ylabel("Actual")
                for (i, j), val in np.ndenumerate(cm):
                    self.ax_reg.text(j, i, f"{val}", ha="center", va="center")
            else:
                self.ax_reg.set_title("Run evaluation to see confusion matrix.")
            self._update_training_plot()

        elif plot_type == "ROC Curve":
            if self.last_eval and self.last_eval.get("probas") is not None and len(np.unique(self.last_eval["y_test"])) == 2:
                y_test = self.last_eval["y_test"]
                probas = self.last_eval["probas"][:, 1] if self.last_eval["probas"].shape[1] > 1 else self.last_eval["probas"][:, 0]
                fpr, tpr, _ = roc_curve(y_test, probas)
                auc = roc_auc_score(y_test, probas)
                self.ax_reg.plot(fpr, tpr, label=f"ROC AUC {auc:.3f}")
                self.ax_reg.plot([0, 1], [0, 1], linestyle="--", color="gray")
                self.ax_reg.set_xlabel("False Positive Rate")
                self.ax_reg.set_ylabel("True Positive Rate")
                self.ax_reg.legend()
                self.ax_reg.set_title("ROC Curve")
            else:
                self.ax_reg.set_title("Run classification evaluation to see ROC curve.")
            self._update_training_plot()

        elif plot_type == "Precision-Recall":
            if self.last_eval and self.last_eval.get("probas") is not None and len(np.unique(self.last_eval["y_test"])) == 2:
                y_test = self.last_eval["y_test"]
                probas = self.last_eval["probas"][:, 1] if self.last_eval["probas"].shape[1] > 1 else self.last_eval["probas"][:, 0]
                precision, recall, _ = precision_recall_curve(y_test, probas)
                self.ax_reg.plot(recall, precision, color="tab:green")
                self.ax_reg.set_xlabel("Recall")
                self.ax_reg.set_ylabel("Precision")
                self.ax_reg.set_title("Precision-Recall Curve")
            else:
                self.ax_reg.set_title("Run classification evaluation to see PR curve.")
            self._update_training_plot()

        elif plot_type == "Feature Importance":
            if self.model is not None and hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                idx = np.argsort(importances)[::-1]
                sorted_feats = np.array(features)[idx]
                sorted_imps = importances[idx]
                self.ax_reg.barh(sorted_feats, sorted_imps, color="tab:purple")
                self.ax_reg.set_title("Feature importance")
                self.ax_reg.invert_yaxis()
            else:
                self.ax_reg.set_title("Feature importance available for Random Forest models.")
            self._update_training_plot()

        elif plot_type == "Loss Curve":
            self._update_training_plot(axis=self.ax_reg)
            self.ax_loss.set_title("")
        else:
            self._update_training_plot()

        self.canvas.draw_idle()

    def preview_data(self) -> None:
        if self.df is None:
            messagebox.showinfo("Preview", "Load a dataset first.")
            return
        top = tk.Toplevel(self.root)
        top.title("Data preview")
        text = tk.Text(top, wrap="none", width=100, height=20)
        text.pack(fill="both", expand=True)
        text.insert(tk.END, self.df.head(20).to_string(index=False))
        text.configure(state="disabled")

    def reset_defaults(self) -> None:
        self.epochs_var.set("40")
        self.lr_var.set("0.01")
        self.alpha_var.set("0.0001")
        self.random_state_var.set("42")
        self.n_estimators_var.set("120")
        self.max_depth_var.set("")
        self.test_size_var.set("0.2")
        self.cv_folds_var.set("0")
        self.standardize_var.set(True)
        self.progress_var.set(0)
        self._toggle_param_visibility()
        self._update_train_test_info()

    def save_plot(self) -> None:
        os.makedirs("Models", exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            initialdir=os.path.join(os.getcwd(), "Models"),
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            title="Save current plot",
        )
        if not file_path:
            return
        try:
            self.figure.savefig(file_path, dpi=200, bbox_inches="tight")
            self.status_var.set(f"Plot saved to {file_path}")
        except Exception as exc:
            messagebox.showerror("Save plot", f"Could not save plot: {exc}")

    def export_classification_report(self) -> None:
        if not self.last_eval or "report" not in self.last_eval:
            messagebox.showinfo("Export", "Run a classification evaluation first.")
            return
        os.makedirs("Models", exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            initialdir=os.path.join(os.getcwd(), "Models"),
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
            title="Save classification report",
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.last_eval["report"])
            self.status_var.set(f"Classification report saved to {file_path}")
        except Exception as exc:
            messagebox.showerror("Export", f"Could not save report: {exc}")

    def _run_cross_validation(self, X: np.ndarray, y: np.ndarray) -> str | None:
        try:
            folds = int(float(self.cv_folds_var.get()))
        except ValueError:
            return None
        if folds <= 1:
            return None
        try:
            from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

            model_type = self.model_var.get()
            random_state = int(float(self.random_state_var.get()))
            lr = float(self.lr_var.get())
            alpha = float(self.alpha_var.get())
            n_estimators = int(float(self.n_estimators_var.get()))
            max_depth = self.max_depth_var.get().strip()
            max_depth_int = int(float(max_depth)) if max_depth else None

            def build_estimator():
                if model_type == "Random Forest Regression":
                    return RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth_int, random_state=random_state
                    )
                if model_type == "Logistic Regression":
                    return SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=alpha,
                        learning_rate="constant",
                        eta0=lr,
                        random_state=random_state,
                        max_iter=1000,
                        tol=1e-3,
                    )
                penalty = "l2" if model_type != "Lasso Regression" else "l1"
                adjusted_alpha = alpha if model_type != "Linear Regression" else max(alpha * 0.1, 1e-6)
                return SGDRegressor(
                    penalty=penalty,
                    alpha=adjusted_alpha,
                    learning_rate="constant",
                    eta0=lr,
                    random_state=random_state,
                    max_iter=1000,
                    tol=1e-3,
                )

            estimator = build_estimator()
            if self.standardize_var.get():
                from sklearn.pipeline import Pipeline

                estimator = Pipeline([("scaler", StandardScaler()), ("model", estimator)])

            # apply model-specific transform for CV without altering main transformer
            def transform_for_cv(X_in: np.ndarray) -> np.ndarray:
                mt = self.model_var.get()
                if mt in ("Quadratic Regression", "Cubic Regression"):
                    degree = 2 if mt == "Quadratic Regression" else 3
                    transformer = PolynomialFeatures(degree=degree, include_bias=False)
                    return transformer.fit_transform(X_in)
                if mt == "Logarithmic Regression":
                    return (np.sign(X_in) * np.log1p(np.abs(X_in))).astype(float)
                if mt == "Exponential Regression":
                    return np.exp(np.clip(X_in, -5, 5)).astype(float)
                if mt == "Trigonometric Regression":
                    sin_part = np.sin(X_in)
                    cos_part = np.cos(X_in)
                    return np.concatenate([sin_part, cos_part], axis=1).astype(float)
                return X_in.astype(float)

            X_cv = transform_for_cv(X)
            if self.task_type == "classification":
                cv = StratifiedKFold(folds, shuffle=True, random_state=random_state)
                scores = cross_val_score(estimator, X_cv, y, cv=cv, scoring="accuracy")
                return f"CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}"
            cv = KFold(folds, shuffle=True, random_state=random_state)
            scores = cross_val_score(estimator, X_cv, y, cv=cv, scoring="neg_mean_squared_error")
            scores = -scores
            return f"CV MSE: {np.mean(scores):.3f} ± {np.std(scores):.3f}"
        except Exception as exc:
            self.status_var.set(f"CV failed: {exc}")
            return None

    def save_session(self) -> None:
        session = {
            "dataset": self.dataset_var.get(),
            "model_type": self.model_var.get(),
            "target": self.target_var.get(),
            "features": self._get_selected_features(),
            "plot_feature": self.plot_feature_var.get(),
            "plot_type": self.plot_type_var.get(),
            "epochs": self.epochs_var.get(),
            "learning_rate": self.lr_var.get(),
            "alpha": self.alpha_var.get(),
            "n_estimators": self.n_estimators_var.get(),
            "max_depth": self.max_depth_var.get(),
            "random_state": self.random_state_var.get(),
            "test_size": self.test_size_var.get(),
            "cv_folds": self.cv_folds_var.get(),
            "standardize": self.standardize_var.get(),
        }
        os.makedirs("Models", exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            initialdir=os.path.join(os.getcwd(), "Models"),
            title="Save session",
            defaultextension=".json",
            filetypes=[("JSON file", "*.json")],
        )
        if not file_path:
            return
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        self.status_var.set(f"Session saved to {file_path}")

    def load_session(self) -> None:
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "Models"),
            title="Load session",
            filetypes=[("JSON file", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                session = json.load(f)
            dataset_path = session.get("dataset", self.dataset_var.get())
            self.dataset_var.set(dataset_path)
            self.model_var.set(session.get("model_type", self.model_var.get()))
            target_from_session = session.get("target", self.target_var.get())
            self.plot_feature_var.set(session.get("plot_feature", self.plot_feature_var.get()))
            self.plot_type_var.set(session.get("plot_type", self.plot_type_var.get()))
            self.epochs_var.set(session.get("epochs", self.epochs_var.get()))
            self.lr_var.set(session.get("learning_rate", self.lr_var.get()))
            self.alpha_var.set(session.get("alpha", self.alpha_var.get()))
            self.n_estimators_var.set(session.get("n_estimators", self.n_estimators_var.get()))
            self.max_depth_var.set(session.get("max_depth", self.max_depth_var.get()))
            self.random_state_var.set(session.get("random_state", self.random_state_var.get()))
            self.test_size_var.set(session.get("test_size", self.test_size_var.get()))
            self.cv_folds_var.set(session.get("cv_folds", self.cv_folds_var.get()))
            self.standardize_var.set(session.get("standardize", self.standardize_var.get()))
            # Load dataset after setting variables so selections can be applied
            if dataset_path and os.path.exists(dataset_path):
                self.load_dataset()
            self.target_var.set(target_from_session)
            if self.df is not None:
                cols = list(self.df.columns)
                self._refresh_feature_list(cols)
                saved_features = session.get("features", [])
                if saved_features and self.feature_listbox:
                    self.feature_listbox.selection_clear(0, tk.END)
                    for feat in saved_features:
                        if feat in cols:
                            idx = cols.index(feat)
                            self.feature_listbox.selection_set(idx)
                self._update_dataset_info()
            self._toggle_param_visibility()
            self._update_train_test_info()
            self.status_var.set(f"Session loaded from {file_path}")
        except Exception as exc:
            messagebox.showerror("Load session", f"Could not load session: {exc}")

    def open_generator(self) -> None:
        try:
            launch_generator_window(self.root, os.path.join(os.getcwd(), "Data"))
        except Exception as exc:
            messagebox.showerror("Generator", f"Could not open generator: {exc}")

    def save_model(self) -> None:
        if self.model is None:
            messagebox.showinfo("Save model", "No trained model to save.")
            return
        os.makedirs("Models", exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            initialdir=os.path.join(os.getcwd(), "Models"),
            title="Save model",
            defaultextension=".joblib",
            filetypes=[("Joblib file", "*.joblib")],
        )
        if not file_path:
            return
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "model_type": self.model_var.get(),
            "target": self.target_var.get(),
            "features": self._get_selected_features(),
            "standardize": self.standardize_var.get(),
            "feature_transformer": self.feature_transformer,
        }
        try:
            joblib.dump(bundle, file_path)
            self.status_var.set(f"Model saved to {file_path}")
        except Exception as exc:
            messagebox.showerror("Save model", f"Could not save model: {exc}")

    def load_model(self) -> None:
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "Models"),
            title="Load model",
            filetypes=[("Joblib file", "*.joblib"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            bundle = joblib.load(file_path)
            self.model = bundle.get("model")
            self.scaler = bundle.get("scaler", StandardScaler())
            self.label_encoder = bundle.get("label_encoder")
            self.model_var.set(bundle.get("model_type", self.model_var.get()))
            self.target_var.set(bundle.get("target", self.target_var.get()))
            self.standardize_var.set(bundle.get("standardize", self.standardize_var.get()))
            self.feature_transformer = bundle.get("feature_transformer")
            self.status_var.set(f"Loaded model from {file_path}")
            self.training_history = []
            self._toggle_param_visibility()
            self.update_controls_state()
        except Exception as exc:
            messagebox.showerror("Load model", f"Could not load model: {exc}")

    def save_params(self) -> None:
        os.makedirs("Models", exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            initialdir=os.path.join(os.getcwd(), "Models"),
            title="Save params",
            defaultextension=".json",
            filetypes=[("JSON file", "*.json")],
        )
        if not file_path:
            return
        params = {
            "dataset": self.dataset_var.get(),
            "model_type": self.model_var.get(),
            "target": self.target_var.get(),
            "features": self._get_selected_features(),
            "epochs": self.epochs_var.get(),
            "learning_rate": self.lr_var.get(),
            "alpha": self.alpha_var.get(),
            "n_estimators": self.n_estimators_var.get(),
            "max_depth": self.max_depth_var.get(),
            "random_state": self.random_state_var.get(),
            "test_size": self.test_size_var.get(),
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)
            self.status_var.set(f"Parameters saved to {file_path}")
        except Exception as exc:
            messagebox.showerror("Save params", f"Could not save params: {exc}")

    def load_params(self) -> None:
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "Models"),
            title="Load params",
            filetypes=[("JSON file", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            self.dataset_var.set(params.get("dataset", self.dataset_var.get()))
            self.model_var.set(params.get("model_type", self.model_var.get()))
            self.target_var.set(params.get("target", self.target_var.get()))
            self.epochs_var.set(params.get("epochs", self.epochs_var.get()))
            self.lr_var.set(params.get("learning_rate", self.lr_var.get()))
            self.alpha_var.set(params.get("alpha", self.alpha_var.get()))
            self.n_estimators_var.set(params.get("n_estimators", self.n_estimators_var.get()))
            self.max_depth_var.set(params.get("max_depth", self.max_depth_var.get()))
            self.random_state_var.set(params.get("random_state", self.random_state_var.get()))
            self.test_size_var.set(params.get("test_size", self.test_size_var.get()))
            if self.df is not None:
                cols = list(self.df.columns)
                self._refresh_feature_list(cols)
            self._update_train_test_info()
            self._toggle_param_visibility()
            self.status_var.set(f"Parameters loaded from {file_path}")
        except Exception as exc:
            messagebox.showerror("Load params", f"Could not load params: {exc}")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = RegressionToolApp()
    app.run()
