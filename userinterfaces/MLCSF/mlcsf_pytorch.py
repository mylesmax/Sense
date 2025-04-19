"""
adapted from https://github.com/NMIL230/nmil-p-ml-csf-JVis2024

citations
1) Active mutual conjoint estimation of multiple contrast sensitivity functions. Marticorena, D. C. P., Wong, Q. W., Browning, J., Wilbur, K., Davey, P. G., Seitz, A. R., Gardner, J. R., & Barbour, D. L. Journal of Vision, 24(8):6, August, 2024.
2) Contrast response function estimation with nonparametric Bayesian active learning Dom C P Marticorena 1 2, Quinn Wai Wong 1 3, Jake Browning 4 5, Ken Wilbur 4 6, Samyukta Jayakumar 7 8, Pinakin Gunvant Davey 9 10, Aaron R Seitz 11 12, Jacob R Gardner 13 14, Dennis L Barbour 1 15
3) Bayesian active probabilistic classification for psychometric field estimation Xinyu D Song 1, Kiron A Sukesan 1, Dennis L Barbour 2 PMID: 29256098 PMCID: PMC5839980 DOI: 10.3758/s13414-017-1460-0

"""

from __future__ import annotations
import os, sys, csv, math
from pathlib import Path
from itertools import product
import warnings
import time

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary, set_trainable
from gpflow.ci_utils import reduce_in_tests # For setting optimization iterations

from scipy.stats import qmc # Halton sequence
from scipy.stats import norm as scipy_norm # Gaussian CDF for BALD calculation
from scipy.ndimage import gaussian_filter # Added for smoothing

warnings.filterwarnings("ignore", message=".*Converting sparse IndexedSlices.*")
warnings.filterwarnings("ignore", message=".*Consider increasing the number of epochs.*")
tf.get_logger().setLevel('INFO')

os.environ["QT_API"] = "pyqt6"
import matplotlib; matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QInputDialog,
QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
QSizePolicy
)
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt, QTimer
import seaborn as sns
import matplotlib.font_manager as fm


# --- GPflow Configuration ---
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

# --- Constants ---
GAMMA = 0.01 # Guess rate
LAMBDA = 0.01 # Lapse rate
BALD_C = np.sqrt(np.pi / 2.0)
MAX_OPT_ITER = reduce_in_tests(150)


# ────────────────────────────── Plot canvas ────────────────────────────── #
class PosteriorPlot(FigureCanvas):
    """Square heat‑map of posterior mean with the next sample marked."""
    def __init__(self, parent=None, grid_res: int = 60, cmap: str = "inferno",
                 figsize=(7, 7)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        super().__init__(self.fig); self.setParent(parent)
        self.grid_res = grid_res; self.cmap_name = cmap
        self.im = self.next_artist = self.cbar = None
        self.det_scatter = self.miss_scatter = self.contour = None
        self.scientific_style = False
        self.colorblind_style = False
        self.setMinimumSize(400, 400); self.setMaximumSize(500, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.ax.set_aspect("equal", adjustable="box")

        self.cb_colors = sns.color_palette('colorblind')
        self.cb_colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r,g,b in self.cb_colors]


    def set_cmap(self, name): self.cmap_name = name

    def set_scientific_style(self, scientific=True):
        self.scientific_style = scientific
        if scientific: self.colorblind_style = False

    def set_colorblind_style(self, colorblind=True):
        self.colorblind_style = colorblind
        if colorblind: self.scientific_style = False

    def refresh(self, X_grid, Z_mean_prob, next_pt, x_label, y_label,
                X_samples=None, y_samples=None, display_mean_prob=None):
        """
        Updates the plot. Z_mean_prob is the mean probability used for contours.
        display_mean_prob is optionally sharpened for heatmap display.
        X_grid and next_pt now use log contrast values for y-axis.
        """
        # --- Data Preparation ---
        Z_heatmap_data = display_mean_prob if display_mean_prob is not None else Z_mean_prob
        if Z_heatmap_data is None:
             Z_heatmap = np.full((self.grid_res, self.grid_res), 0.5)
        else:
            Z_heatmap = Z_heatmap_data.reshape(self.grid_res, self.grid_res)

        # Data for the contour: ALWAYS use the raw posterior mean probability ('Z_mean_prob')
        Z_contour_data = Z_mean_prob
        if Z_contour_data is None:
             Z_contour = np.full((self.grid_res, self.grid_res), 0.5)
        else:
            Z_contour = Z_contour_data.reshape(self.grid_res, self.grid_res)

        # Get bounds from the real-space grid (now includes log contrast for y)
        x_min, x_max = X_grid[:, 0].min(), X_grid[:, 0].max()
        y_min_log, y_max_log = X_grid[:, 1].min(), X_grid[:, 1].max() # These are log bounds
        xs = np.linspace(x_min, x_max, self.grid_res)
        ys_log = np.linspace(y_min_log, y_max_log, self.grid_res) # y axis is log

        # --- Styling Setup ---
        if self.colorblind_style:
            # ... (colorblind style settings) ...
            plt.rcParams['font.family'] = 'sans-serif'
            marker_size = 20
            marker_line_width = 1.0
            line_width = 1.2
            tick_font_size = 8
            label_font_size = 8
            title_font_size = 10
            font_weight = 'normal'
            success_color = self.cb_colors[0]
            failure_color = self.cb_colors[1]
            contour_color = self.cb_colors[4]
            contour_style = '-'
            next_point_color = 'k'
            plt.rcParams['font.weight'] = font_weight
            plt.rcParams['axes.labelweight'] = font_weight
            plt.rcParams['axes.titleweight'] = font_weight
        elif self.scientific_style:
            plt.rcParams['font.family'] = 'sans-serif'
            marker_size = 35
            marker_line_width = 0.65
            line_width = 1.5
            tick_font_size = 8
            label_font_size = 8
            title_font_size = 10
            font_weight = 'normal'
            success_color = 'blue'
            failure_color = 'red'
            contour_color = '#40E1D0'
            contour_style = 'dashed'
            next_point_color = 'black'
        else: # Default style
            plt.rcParams['font.family'] = 'sans-serif' # Ensure default also has sans-serif
            marker_size = 30
            marker_line_width = 1.3
            line_width = 1.3
            tick_font_size = plt.rcParams['xtick.labelsize'] # Use default tick size
            label_font_size = plt.rcParams['axes.labelsize'] # Use default label size
            title_font_size = plt.rcParams['axes.titlesize'] # Use default title size
            font_weight = plt.rcParams['font.weight'] # Use default weight
            success_color = 'lime'
            failure_color = 'red'
            contour_color = 'white'
            contour_style = '--'
            next_point_color = 'white'


        # --- Plotting Elements ---
        # Initialize or update heatmap
        if self.im is None:
            self.im = self.ax.imshow(
                Z_heatmap, origin="lower", cmap=self.cmap_name,
                vmin=0, vmax=1, extent=[x_min, x_max, y_min_log, y_max_log], # Use log contrast extent
                aspect="auto"
            )
            self.cbar = self.fig.colorbar(self.im, ax=self.ax)
            self.cbar.set_label("P(detect)")
            if self.colorblind_style:
                self.cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
                self.cbar.ax.tick_params(labelsize=tick_font_size)
                self.cbar.ax.set_ylabel("P(detect)", fontsize=label_font_size, weight=font_weight)
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label) # y_label now indicates log contrast
        else:
            self.im.set_data(Z_heatmap)
            self.im.set_cmap(self.cmap_name)
            self.im.set_extent([x_min, x_max, y_min_log, y_max_log]) # Update with log contrast extent
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label) # y_label now indicates log contrast

        # Update 50% threshold contour (using Z_contour)
        if self.contour is not None:
            # Try to remove the contour in a way that works with different matplotlib versions
            try:
                # For older matplotlib versions
                for coll in self.contour.collections:
                    coll.remove()
            except (AttributeError, TypeError):
                # For newer matplotlib versions
                try:
                    self.contour.remove()
                except:
                    # Last resort: clear the axes and redraw everything except contour
                    print("Warning: Could not remove contour normally. Clearing axes.") # Added warning
                    # Store necessary elements before clearing
                    current_title = self.ax.get_title()
                    current_xlabel = self.ax.get_xlabel()
                    current_ylabel = self.ax.get_ylabel()
                    current_xlim = self.ax.get_xlim()
                    current_ylim = self.ax.get_ylim()
                    # Clear axes but keep the image reference if possible
                    self.ax.clear()
                    # Redraw the heatmap
                    if self.im is not None:
                        self.im = self.ax.imshow(
                            Z_heatmap, origin="lower", cmap=self.cmap_name,
                            vmin=0, vmax=1, extent=[x_min, x_max, y_min_log, y_max_log], # Use log contrast extent
                            aspect="auto"
                        )
                        # Re-add colorbar if it existed
                        if self.cbar:
                            self.cbar = self.fig.colorbar(self.im, ax=self.ax)
                            self.cbar.set_label("P(detect)")
                            if self.colorblind_style:
                                self.cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
                                self.cbar.ax.tick_params(labelsize=tick_font_size)
                                self.cbar.ax.set_ylabel("P(detect)", fontsize=label_font_size, weight=font_weight)
                    # Restore labels, title, limits
                    self.ax.set_title(current_title)
                    self.ax.set_xlabel(current_xlabel)
                    self.ax.set_ylabel(current_ylabel)
                    self.ax.set_xlim(current_xlim)
                    self.ax.set_ylim(current_ylim)
                    # Reset other artists that might have been cleared
                    self.det_scatter = None
                    self.miss_scatter = None
                    self.next_artist = None


            self.contour = None


        if Z_contour_data is not None and np.all(np.isfinite(Z_contour)):
            self.contour = self.ax.contour(xs, ys_log, Z_contour, levels=[0.5],
                                           colors=[contour_color],
                                           linestyles=contour_style,
                                           linewidths=line_width)
        else:
             self.contour = None

        if self.det_scatter is None or self.det_scatter.axes != self.ax:
            if self.det_scatter and self.det_scatter.axes != self.ax: self.det_scatter = None
        elif self.det_scatter: self.det_scatter.remove()

        if self.miss_scatter is None or self.miss_scatter.axes != self.ax:
             if self.miss_scatter and self.miss_scatter.axes != self.ax: self.miss_scatter = None
        elif self.miss_scatter: self.miss_scatter.remove()


        if X_samples and y_samples and len(X_samples) > 0:
            X_arr = np.vstack(X_samples) # X_arr now includes log contrast for y-axis
            y_arr = np.array(y_samples)
            detected_mask = (y_arr == 1)
            missed_mask = (y_arr == 0)

            # Detections
            if np.any(detected_mask):
                 marker = 'o' if not self.scientific_style else '+'
                 scatter_kwargs = {'s': marker_size, 'alpha': .9 if self.colorblind_style else .7, 'linewidths': marker_line_width, 'zorder': 4}
                 if self.colorblind_style:
                     self.det_scatter = self.ax.scatter(X_arr[detected_mask, 0], X_arr[detected_mask, 1], c=success_color, marker=marker, **scatter_kwargs)
                 elif self.scientific_style:
                      self.det_scatter = self.ax.scatter(X_arr[detected_mask, 0], X_arr[detected_mask, 1], c=success_color, marker=marker, **scatter_kwargs)
                 else: # Default
                      self.det_scatter = self.ax.scatter(X_arr[detected_mask, 0], X_arr[detected_mask, 1], c=success_color, marker=marker, edgecolors='darkgreen', **scatter_kwargs)


            # Misses
            if np.any(missed_mask):
                 marker = 's' if self.colorblind_style else ('d' if self.scientific_style else 'x')
                 scatter_kwargs = {'s': marker_size, 'alpha': .9 if self.colorblind_style else .7, 'linewidths': marker_line_width, 'zorder': 4}
                 if self.colorblind_style:
                      self.miss_scatter = self.ax.scatter(X_arr[missed_mask, 0], X_arr[missed_mask, 1], c=failure_color, marker=marker, **scatter_kwargs)
                 elif self.scientific_style:
                      self.miss_scatter = self.ax.scatter(X_arr[missed_mask, 0], X_arr[missed_mask, 1], marker=marker, facecolor='none', edgecolors=failure_color, **scatter_kwargs)
                 else: # Default
                      self.miss_scatter = self.ax.scatter(X_arr[missed_mask, 0], X_arr[missed_mask, 1], c=failure_color, marker=marker, **scatter_kwargs)

        # Mark candidate next point (now includes log contrast for y-axis)
        # Check if next_artist needs redrawing
        if self.next_artist is None or self.next_artist.axes != self.ax:
            if self.next_artist and self.next_artist.axes != self.ax: self.next_artist = None
        elif self.next_artist:
             # If it exists and belongs to the current axes, just update it or hide it
             pass # Handled below
        else: # If it was removed by ax.clear() but reference still exists
             self.next_artist = None


        if next_pt is not None: # Check if next_pt is valid
            if self.next_artist is None:
                (self.next_artist,) = self.ax.plot(*next_pt, "x",
                                                   color=next_point_color, ms=10, mew=2,
                                                   zorder=6)
            else:
                self.next_artist.set_data([next_pt[0]], [next_pt[1]])
                self.next_artist.set_color(next_point_color)
                self.next_artist.set_visible(True)
        elif self.next_artist: # If next_pt is None (e.g., converged), hide marker
             self.next_artist.set_visible(False)


        # --- Final Touches ---
        if self.colorblind_style or self.scientific_style:
            self.ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
            self.ax.xaxis.label.set_fontsize(label_font_size)
            self.ax.xaxis.label.set_fontweight(font_weight)
            self.ax.yaxis.label.set_fontsize(label_font_size)
            self.ax.yaxis.label.set_fontweight(font_weight)

        title = "Posterior mean – × = next sample"
        if self.colorblind_style:
            title = "Posterior mean (○ = detect, □ = no detect)"
        elif self.scientific_style:
             title = "Posterior mean (+ = detect, ⋄ = no detect)"
        else:
            title = "Posterior mean – white × = next sample"

        # Ensure title is set even if axes were cleared
        if self.colorblind_style or self.scientific_style:
             self.ax.set_title(title, fontsize=title_font_size, weight=font_weight)
        else:
             self.ax.set_title(title)


        # Ensure limits are set even if axes were cleared
        self.ax.set_xlim(x_min, x_max); self.ax.set_ylim(y_min_log, y_max_log)
        try:
            self.fig.tight_layout()
        except ValueError:
            print("Warning: tight_layout failed.")
        self.draw_idle()


    def export_png(self, path, n):
        vis = False
        if self.next_artist is not None:
            vis = self.next_artist.get_visible()
            self.next_artist.set_visible(False)
        old = self.ax.get_title(); self.ax.set_title(f"Posterior (n={n})")
        self.fig.savefig(path, dpi=350, bbox_inches="tight")
        self.ax.set_title(old);
        if self.next_artist is not None:
            self.next_artist.set_visible(vis)
        self.draw_idle()

# ──────────────────────────────── GUI ─────────────────────────────────── #
class MainWindow(QMainWindow):
    MAX_TRIALS = 100
    BALD_STOP_BITS = 0.05

    def __init__(self):
        super().__init__(); self.setWindowTitle("ML‑CSF Active Learner (GPflow VI+BALD v5)")
        self._dark(); self.grid_res = 60
        self.initialized = False
        self._build_ui()
        self.X_samples = [] # Real space samples
        self.X_samples_norm = [] # Normalized [0,1] samples (for GP)
        self.y_samples = [] # Outcomes (0 or 1)
        self.gp_model = None # GPflow model
        self.gp_optimizer = gpflow.optimizers.Scipy() # GPflow optimizer
        self.next_pt = None # Real space next point
        self.next_pt_norm = None # Normalized next point
        self.prob_mean_on_grid = None # Mean probability p(detect=1) on grid
        self.latent_mean_on_grid = None # Latent mean f on grid
        self.latent_var_on_grid = None # Latent variance f on grid
        self.acq_on_grid = None # Acquisition function value on grid
        self.last_log_marginal_likelihood = -np.inf
        self.scientific_style = False # Track if scientific style is active
        self.colorblind_style = False # Track if colorblind style is active

        self.update_timer = QTimer(self)
        self.update_timer.setInterval(50)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._update_plot_and_status)


    def _dark(self):
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#121212"))
        pal.setColor(QPalette.ColorRole.WindowText, QColor("white"))
        self.setPalette(pal)
        self.setStyleSheet("QPushButton{padding:4px;} QLabel{color:white;}")

    def _check_y_bounds(self):
        # Ensure ymin corresponds to a positive contrast
        min_contrast = 10**self.ymin.value()
        if min_contrast <= 0:
            # Suggest a minimum value, e.g., log10(0.1) = -1
            QMessageBox.warning(self, "Invalid Y Min",
                                f"Minimum Y (log10 contrast) value {self.ymin.value():.2f} corresponds to non-positive contrast ({min_contrast:.2e}).\nPlease set it higher (e.g., -1 for 10%).")
            return False
        if self.ymax.value() <= self.ymin.value():
             QMessageBox.warning(self, "Invalid Y Bounds", "Y max must be greater than Y min.")
             return False
        return True

    def _normalize_point(self, point):
        if not self._check_y_bounds(): # Add check here
             # Handle error or return default normalization
             return np.array([0.5, 0.5]) # Or raise error

        x_range = self.xmax.value() - self.xmin.value()
        # Use LOG contrast range for y
        y_log_range = self.ymax.value() - self.ymin.value()

        x_norm = (point[0] - self.xmin.value()) / x_range if x_range != 0 else 0.5
        # Normalize the LOG contrast value
        y_norm = (point[1] - self.ymin.value()) / y_log_range if y_log_range != 0 else 0.5
        return np.clip(np.array([x_norm, y_norm]), 0.0, 1.0)

    def _denormalize_point(self, norm_point):
        if not self._check_y_bounds(): # Add check here
             # Handle error or return default point
             return np.array([self.xmin.value(), self.ymin.value()]) # Or raise error

        x = norm_point[0] * (self.xmax.value() - self.xmin.value()) + self.xmin.value()
        # De-normalize to LOG contrast first
        y_log = norm_point[1] * (self.ymax.value() - self.ymin.value()) + self.ymin.value()
        return np.array([x, y_log]) # Return log contrast

    # ─── UI layout ─── #
    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c); v = QVBoxLayout(c)

        # row 1 – labels, units, colormap
        r1 = QHBoxLayout()
        for txt, attr, default in (
            ("Var 1 name:", "var1", "Test1"),
            ("Unit:", "unit1", "Unit1"),
            ("Var 2 name:", "var2", "Test2"),
            ("Unit:", "unit2", "Unit2"),
        ):
            r1.addWidget(QLabel(txt)); le = QLineEdit(default); r1.addWidget(le)
            setattr(self, attr, le)
        r1.addWidget(QLabel("Colormap:"))
        self.cmap = QComboBox(); self.cmap.addItems(
            ["inferno", "viridis", "magma", "cividis", "gist_gray"]
        ); r1.addWidget(self.cmap)

        # Add style dropdown
        r1.addWidget(QLabel("Style:"))
        self.style = QComboBox()
        self.style.addItems(["Default", "Scientific", "Colorblind"])
        r1.addWidget(self.style)

        v.addLayout(r1)

        # row 2 – bounds in cleaner format with brackets
        r2 = QHBoxLayout()
        
        # X bounds with brackets
        r2.addWidget(QLabel("X bounds: ["))
        self.xmin = QDoubleSpinBox(minimum=-1e6, maximum=1e6,
                                singleStep=0.5, value=0.5, decimals=3)
        r2.addWidget(self.xmin)
        r2.addWidget(QLabel(","))
        self.xmax = QDoubleSpinBox(minimum=-1e6, maximum=1e6,
                                singleStep=0.5, value=20, decimals=3)
        r2.addWidget(self.xmax)
        r2.addWidget(QLabel("]"))
        
        # Add some spacing
        r2.addSpacing(20)
        
        # Y bounds with brackets
        r2.addWidget(QLabel("Y bounds: ["))
        self.ymin = QDoubleSpinBox(minimum=-1e6, maximum=1e6,
                                singleStep=0.2, value=-1.0, decimals=3)
        r2.addWidget(self.ymin)
        r2.addWidget(QLabel(","))
        self.ymax = QDoubleSpinBox(minimum=-1e6, maximum=1e6,
                                singleStep=0.2, value=1.7, decimals=3)
        r2.addWidget(self.ymax)
        r2.addWidget(QLabel("]"))
        
        # Connect value change events
        self.xmin.valueChanged.connect(self._bounds_changed)
        self.xmax.valueChanged.connect(self._bounds_changed)
        self.ymin.valueChanged.connect(self._bounds_changed)
        self.ymax.valueChanged.connect(self._bounds_changed)
        
        v.addLayout(r2)

        # row 3 – initialise with compact layout
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Initial point: ["))
        self.x0 = QDoubleSpinBox(singleStep=.1, value=2.0)
        r3.addWidget(self.x0)
        r3.addWidget(QLabel(","))
        self.y0 = QDoubleSpinBox(singleStep=.1, value=1.0)
        r3.addWidget(self.y0)
        r3.addWidget(QLabel("]"))
        
        # Add some spacing
        r3.addSpacing(20)
        
        self.autoSeed = QCheckBox("Auto‑seed 7 pts")
        self.autoSeed.setChecked(False)
        r3.addWidget(self.autoSeed)
        
        self.btnInitDet = QPushButton("Detect at init")
        self.btnInitDet.clicked.connect(lambda: self._initialize(1))
        r3.addWidget(self.btnInitDet)
        
        self.btnInitMiss = QPushButton("No detect at init")
        self.btnInitMiss.clicked.connect(lambda: self._initialize(0))
        r3.addWidget(self.btnInitMiss)
        
        v.addLayout(r3)

        # row 4 – actions with better formatting for next point and colored buttons
        r4 = QHBoxLayout()
        self.nextLab = QLabel("Sample: n/a")
        r4.addWidget(self.nextLab)
        
        # Colored detect button
        self.btnDet = QPushButton("Detect")
        self.btnDet.setEnabled(False)
        self.btnDet.clicked.connect(lambda: self._record(1))
        self.btnDet.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        r4.addWidget(self.btnDet)
        
        # Colored no detect button
        self.btnMiss = QPushButton("No detect")
        self.btnMiss.setEnabled(False)
        self.btnMiss.clicked.connect(lambda: self._record(0))
        self.btnMiss.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        r4.addWidget(self.btnMiss)
        
        self.btnSave = QPushButton("Save…")
        self.btnSave.setEnabled(False)
        self.btnSave.clicked.connect(self._save)
        r4.addWidget(self.btnSave)
        
        v.addLayout(r4)

        # plot
        self.plot = PosteriorPlot(self, grid_res=self.grid_res, cmap="inferno")
        plot_container = QWidget(); pl = QVBoxLayout(plot_container); pl.setContentsMargins(10,10,10,10)
        pl.addWidget(self.plot, alignment=Qt.AlignmentFlag.AlignCenter)
        v.addWidget(plot_container, stretch=1)

        self.cmap.currentTextChanged.connect(
            lambda name: self.plot.set_cmap(name)
            or (self._trigger_update() if self.initialized else None) # Use trigger
        )
        self.style.currentTextChanged.connect(self._update_style)
        self.statusBar().showMessage("Set bounds, pick initial point & outcome.")

    # ─── helpers ─── #
    def _make_grid(self):
        # Real space grid
        xs = np.linspace(self.xmin.value(), self.xmax.value(), self.grid_res)
        # Grid for LOG contrast
        ys_log = np.linspace(self.ymin.value(), self.ymax.value(), self.grid_res)
        xx, yy_log = np.meshgrid(xs, ys_log, indexing='ij')
        # Store grid with log contrast
        self.X_grid = np.vstack([xx.ravel(order='F'), yy_log.ravel(order='F')]).T

        # Normalized grid for GP model (still [0,1]x[0,1])
        xs_norm = np.linspace(0, 1, self.grid_res)
        ys_norm = np.linspace(0, 1, self.grid_res)
        xx_norm, yy_norm = np.meshgrid(xs_norm, ys_norm, indexing='ij')
        self.X_grid_norm = np.vstack([xx_norm.ravel(order='F'), yy_norm.ravel(order='F')]).T
        self.X_grid_norm_tf = tf.convert_to_tensor(self.X_grid_norm, dtype=gpflow.config.default_float())

        # Store axis vectors (store log axis for y)
        self.xs = xs
        self.ys = ys_log # Store the log contrast axis values
        self.xs_norm = xs_norm
        self.ys_norm = ys_norm

    # ─── initialisation ─── #
    def _initialize(self, outcome: int):
        if self.initialized: return
        if not self._check_y_bounds(): return # Check bounds before proceeding
        
        self.var1_lbl = f"{self.var1.text() or 'X1'} ({self.unit1.text()})"
        self.var2_lbl = f"{self.var2.text() or 'X2'} ({self.unit2.text()})"
        self._make_grid()

        # Reset sample lists
        self.X_samples = []
        self.X_samples_norm = []
        self.y_samples = []

        # Store initial point (real and normalized)
        initial_point = np.array([self.x0.value(), self.y0.value()])
        self.X_samples.append(initial_point)
        self.X_samples_norm.append(self._normalize_point(initial_point))
        self.y_samples.append(outcome)

        # Optional Halton seeding
        if self.autoSeed.isChecked():
            sampler = qmc.Halton(d=2, scramble=False)
            seeds_norm = sampler.random(n=7) # Already in [0,1] space
            for seed_norm in seeds_norm:
                self.X_samples_norm.append(seed_norm)
                seed_real = self._denormalize_point(seed_norm)
                self.X_samples.append(seed_real)
                # Ask the user for the actual outcome for each seed point
                outcome, ok = QInputDialog.getInt(
                    self, "Seed outcome",
                    f"Seen stimulus at ({seed_real[0]:.2f}, {seed_real[1]:.2f})? (1=detect, 0=miss)",
                    1, 0, 1)
                if ok:
                    self.y_samples.append(int(outcome))
                else:
                    # If user cancels, remove the point
                    self.X_samples_norm.pop()
                    self.X_samples.pop()

        # Create GP model structure (will be finalized in _fit)
        self.gp_model = None # Force recreation in _fit

        self.initialized = True
        self.btnInitDet.setEnabled(False); self.btnInitMiss.setEnabled(False)
        self.btnDet.setEnabled(True); self.btnMiss.setEnabled(True); self.btnSave.setEnabled(True)
        for sb in [self.xmin, self.xmax, self.ymin, self.ymax]: sb.setEnabled(True)

        self.statusBar().showMessage("Initializing GP model...")
        QApplication.processEvents() # Allow UI to update
        self._fit_and_update() # Initial fit and update
        self.statusBar().showMessage("Initialized. Select Detect/No Detect for next point.")

    # ─── GP + BALD core ─── #
    @staticmethod
    def _create_gpflow_model(X_norm_tf, y_tf):
        """Creates the GPflow VGP model using an anisotropic RBF kernel."""
        n_samples = X_norm_tf.shape[0]
        if n_samples == 0: return None

        # Define minimum lengthscale constraint to prevent overfitting
        min_lengthscale = 0.2
        
        # We'll try different approaches based on the GPflow version
        try:
            # First approach: Create the kernel with increased initial lengthscales
            rbf_kernel = gpflow.kernels.SquaredExponential(
                lengthscales=[0.5, 0.5], # Increased initial values for smoother fit
                variance=1.0
            )
            
            # Add a log-normal prior for better length-scale values
            try:
                # Log-normal prior centers around shorter length-scales
                lognormal = gpflow.priors.LogNormal(loc=np.log(0.1), scale=0.5)  # lowered median from 0.2 to 0.1
                rbf_kernel.lengthscales.prior = lognormal
                
                # Only constrain the lower end to avoid numerical collapse
                rbf_kernel.lengthscales.transform = gpflow.utilities.positive(lower=0.05)
                print("Successfully set log-normal prior on lengthscales")
            except Exception as e:
                print(f"Could not set log-normal prior: {e}")
                # Fall back to legacy methods if the direct approach fails
                try:
                    # Method 1: Set lengthscale_constraint directly (only lower bound)
                    lengthscale_constraint = gpflow.utilities.positive(lower=0.05)
                    if hasattr(rbf_kernel, 'lengthscale_constraint'):
                        rbf_kernel.lengthscale_constraint = lengthscale_constraint
                    elif hasattr(rbf_kernel, 'lengthscales_constraint'):
                        rbf_kernel.lengthscales_constraint = lengthscale_constraint
                except Exception as e:
                    print(f"Could not set constraint method 1: {e}")
                    
                    try:
                        # Method 2: Set transform directly on the parameter
                        if hasattr(rbf_kernel, 'lengthscales'):
                            if hasattr(rbf_kernel.lengthscales, 'transform'):
                                rbf_kernel.lengthscales.transform = gpflow.utilities.positive(lower=0.05)
                    except Exception as e:
                        print(f"Could not set constraint method 2: {e}")
            
            # Try to set priors if available in this version (fallback to old method)
            try:
                if hasattr(rbf_kernel, 'lengthscales') and hasattr(rbf_kernel.lengthscales, 'prior') and rbf_kernel.lengthscales.prior is None:
                    gamma_prior = gpflow.priors.Gamma(2.0, 3.0)
                    rbf_kernel.lengthscales.prior = gamma_prior  # Only if log-normal failed
                    rbf_kernel.variance.prior = gpflow.priors.Gamma(2.0, 2.0)
            except Exception as e:
                print(f"Could not set fallback priors: {e}")
                
        except Exception as e:
            print(f"Error in kernel creation: {e}")
            # Fallback to simpler kernel without constraints
            rbf_kernel = gpflow.kernels.SquaredExponential(
                lengthscales=[0.5, 0.5],
                variance=1.0
            )
        
        # Small white noise kernel for numerical stability
        white_kernel = gpflow.kernels.White(variance=1e-6)
        
        # Keep white noise variance frozen (non-trainable)
        try:
            set_trainable(white_kernel.variance, False)  # Keep it fixed for jitter only
            print("Successfully set white noise kernel to non-trainable")
        except Exception as e:
            print(f"Could not set white noise as non-trainable: {e}")
        
        # Combine RBF and white noise (no linear component)
        kernel = rbf_kernel + white_kernel

        # Likelihood (Bernoulli with Probit link is default)
        likelihood = gpflow.likelihoods.Bernoulli()

        # Choose model: VGP for exact VI (good for < ~1000 points)
        model = gpflow.models.VGP(
                    (X_norm_tf, y_tf), # Data tuple
                    kernel=kernel,
                    likelihood=likelihood,
                    num_latent_gps=1 # Standard single-output classification
                )

        return model

    @staticmethod
    def _calculate_bald_acquisition(latent_mean, latent_var, C=BALD_C, eps=1e-9):
        """Calculates BALD acquisition using Houlsby 2011 eq 15 formula."""
        # Convert to numpy arrays for simpler calculation
        mean_np = latent_mean.numpy().flatten()
        var_np = latent_var.numpy().flatten()
        
        # Only clip variance from below to ensure stability
        sigma2 = np.maximum(var_np, 1e-6)
        
        # Calculate kappa from Houlsby formula
        kappa = 1.0 / np.sqrt(1.0 + np.pi * sigma2 / 8.0)
        
        # Calculate p = Φ(κ·μ)
        p = scipy_norm.cdf(kappa * mean_np)
        
        # Ensure probability is in range to avoid log(0)
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        
        # Calculate BALD = binary entropy of p
        bald = -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        # Convert back to TensorFlow tensor for consistency
        bald_tf = tf.convert_to_tensor(bald, dtype=gpflow.config.default_float())
        
        return bald_tf

    def _fit(self):
        """Fit GPflow model using VI, compute posterior, and BALD utility."""
        n_samples = len(self.y_samples)
        n_classes = len(set(self.y_samples))

        # --- DEBUG PRINT ---
        print(f"\n--- Entering _fit ---")
        print(f"n_samples: {n_samples}, n_classes: {n_classes}, y_samples: {self.y_samples[:10]}...") # Show first few outcomes
        # --- END DEBUG PRINT ---

        if n_samples < 2 or n_classes < 2:
            print("Using distance heuristic (not enough data or classes for GP)")
            self._distance_heuristic()
            # Ensure next_pt is set if using heuristic
            if not hasattr(self, 'next_pt') or self.next_pt is None:
                 if hasattr(self, 'next_pt_norm') and self.next_pt_norm is not None:
                     self.next_pt = self._denormalize_point(self.next_pt_norm)
                 else: # Fallback if even heuristic failed somehow
                      self.next_pt = self._denormalize_point(np.array([0.5, 0.5]))
            # Clear GP-related results
            self.prob_mean_on_grid = None
            self.latent_mean_on_grid = None
            self.latent_var_on_grid = None
            self.acq_on_grid = getattr(self, 'acq_on_grid', None) # Keep acq if set by heuristic
            return True # Indicate heuristic was used

        # --- If we reach here, we should be trying the GP ---
        print("Attempting GP fitting...") # DEBUG PRINT

        try:
            # --- Prepare Data ---
            X_arr_norm = np.vstack(self.X_samples_norm)
            # Ensure y_arr is float64 {0., 1.} which Bernoulli likelihood often prefers
            y_arr = np.array(self.y_samples, dtype=np.float64).reshape(-1, 1)

            # Convert to Tensors
            X_tf = tf.convert_to_tensor(X_arr_norm, dtype=gpflow.config.default_float())
            y_tf = tf.convert_to_tensor(y_arr, dtype=gpflow.config.default_float())

            # --- DEBUG PRINT ---
            print(f"Data Shapes - X_tf: {X_tf.shape}, y_tf: {y_tf.shape}")
            print(f"Data Types - X_tf: {X_tf.dtype}, y_tf: {y_tf.dtype}")
            # --- END DEBUG PRINT ---

            # --- Create or Update Model ---
            print("Creating GPflow model...") # DEBUG PRINT
            self.gp_model = self._create_gpflow_model(X_tf, y_tf)
            if self.gp_model is None: raise ValueError("Model creation failed.")
            print("GPflow model created.") # DEBUG PRINT

            # --- Optimize Model (VI) ---
            print("Optimizing GPflow model...") # DEBUG PRINT
            
            # Store initial loss for monitoring
            try:
                initial_loss = self.gp_model.training_loss().numpy()
                print(f"Initial loss: {initial_loss:.2f}")
            except Exception as e:
                print(f"Could not compute initial loss: {e}")
                initial_loss = None

            # --- Two-loop optimization scheme ---
            # First loop: Natural gradient on variational parameters
            print("Starting natural gradient optimization on variational parameters...")
            nat_opt = gpflow.optimizers.NaturalGradient(gamma=0.1)
            var_list = [(self.gp_model.q_mu, self.gp_model.q_sqrt)]
            
            try:
                for i in range(40):  # 40 iterations of natural gradient
                    nat_opt.minimize(self.gp_model.training_loss, var_list)
                    if i % 10 == 0:  # Print progress less frequently
                        current_loss = self.gp_model.training_loss().numpy()
                        print(f"  Natural gradient step {i}: loss = {current_loss:.2f}")
            except Exception as e:
                print(f"Natural gradient optimization error: {e}")
                print("Continuing with hyperparameter optimization...")
            
            # Second loop: Scipy on kernel hyperparameters
            print("Starting Scipy optimization on hyperparameters...")
            scipy_opt = gpflow.optimizers.Scipy()
            opt_config = {'maxiter': 200, 'disp': False, 'ftol': 1e-4, 'gtol': 1e-4}
            
            try:
                optimization_result = scipy_opt.minimize(
                    self.gp_model.training_loss,
                    self.gp_model.trainable_variables,
                    options=opt_config
                )
                print(f"Optimization finished. Success: {optimization_result.success}, Message: {optimization_result.message}")
            except Exception as e:
                print(f"Scipy optimization error: {e}")
                print("Continuing with existing model parameters...")
                
            # Check if optimization has improved loss
            try:
                final_loss = self.gp_model.training_loss().numpy()
                print(f"Final loss: {final_loss:.2f}")
                
                # Print kernel parameters to diagnose length scales
                print("--- Kernel parameters after optimization ---")
                print_summary(self.gp_model.kernel)
                
                # Check if optimization made things worse
                if initial_loss is not None and final_loss > initial_loss * 1.5:
                    print("Warning: Optimization significantly worsened the loss. Consider resetting parameters.")
            except Exception as e:
                print(f"Could not compute final loss: {e}")
                
            # Store log likelihood for status
            try:
                self.last_log_marginal_likelihood = self.gp_model.elbo().numpy()
                print(f"Log Marginal Likelihood: {self.last_log_marginal_likelihood:.2f}")
            except Exception as e:
                print(f"Error computing ELBO: {e}")
                self.last_log_marginal_likelihood = -np.inf

            # --- Predict on Grid ---
            print("Predicting on grid...") # DEBUG PRINT
            try:
                # Get latent function predictions (mean and variance)
                latent_mean_tf, latent_var_tf = self.gp_model.predict_f(self.X_grid_norm_tf)
                
                # Convert to numpy and flatten
                self.latent_mean_on_grid = latent_mean_tf.numpy().flatten()
                self.latent_var_on_grid = latent_var_tf.numpy().flatten()
                
                # Handle potential extreme values by clipping
                max_abs_latent = 10.0  # Maximum absolute latent value
                min_var, max_var = 0.01, 50.0  # Min/max variance
                
                # Apply clipping
                self.latent_mean_on_grid = np.clip(self.latent_mean_on_grid, -max_abs_latent, max_abs_latent)
                self.latent_var_on_grid = np.clip(self.latent_var_on_grid, min_var, max_var)
                
                print(f"Latent mean range: {np.min(self.latent_mean_on_grid):.2f} to {np.max(self.latent_mean_on_grid):.2f}") # DEBUG
                print(f"Latent var range: {np.min(self.latent_var_on_grid):.2f} to {np.max(self.latent_var_on_grid):.2f}") # DEBUG
                
                # Get probability predictions using predict_y
                prob_mean_tf, _ = self.gp_model.predict_y(self.X_grid_norm_tf)
                prob_mean_raw = prob_mean_tf.numpy().flatten()
                
                # Fix any NaN or Inf values
                if not np.all(np.isfinite(prob_mean_raw)):
                    print(f"Warning: {np.sum(~np.isfinite(prob_mean_raw))} non-finite probability values detected. Fixing.")
                    prob_mean_raw = np.nan_to_num(prob_mean_raw, nan=0.5, posinf=1.0, neginf=0.0)
                
                print(f"Raw predicted prob range: {np.min(prob_mean_raw):.2f} to {np.max(prob_mean_raw):.2f}") # DEBUG
                
                # Store the raw probability without gamma/lambda scaling
                self.prob_mean_on_grid = prob_mean_raw
                
                # Create a 2D version for spatial processing
                prob_2d = prob_mean_raw.reshape(self.grid_res, self.grid_res)
                
                # Apply spatial smoothing for visualization only in scientific style
                n_samples = len(self.y_samples)
                
                # Fixed smoothing until ≥ 10 samples to prevent early flicker
                if n_samples < 10:
                    smooth_sigma = 1.0  # Fixed smoothing for early samples
                else:
                    smooth_sigma = max(0.5, 1.0 - 0.02 * n_samples)  # Reduce smoothing as we get more data
                
                # Only apply smoothing in scientific mode
                if hasattr(self, 'scientific_style') and self.scientific_style:
                    try:
                        smoothed_2d = gaussian_filter(prob_2d, sigma=smooth_sigma)
                        self.display_mean_on_grid = smoothed_2d.flatten(order="F")
                        print(f"Applied smoothing with sigma={smooth_sigma:.2f} (Scientific mode)")
                    except Exception as e:
                        print(f"Smoothing failed: {e}, using unsmoothed probabilities")
                        self.display_mean_on_grid = self.prob_mean_on_grid.copy()
                else:
                    # In default or colorblind mode, use unsmoothed version
                    self.display_mean_on_grid = self.prob_mean_on_grid.copy()
                    print("Using unsmoothed probabilities (non-Scientific mode)")
                
                print("Predictions calculated.")
                
            except Exception as e:
                print(f"Prediction failed: {e}")
                # Fallback to distance heuristic
                self._distance_heuristic()
                return False  # Indicate failure

            # --- Calculate Acquisition Function (Pure BALD) ---
            print("Calculating acquisition function (BALD)...") # DEBUG PRINT
            bald_acq_tf = self._calculate_bald_acquisition(
                latent_mean_tf, latent_var_tf, C=BALD_C # Removed gamma, lamda args
            )
            bald_acq = bald_acq_tf.numpy().flatten()
            print(f"BALD acq range: {np.min(bald_acq):.3f} to {np.max(bald_acq):.3f}") # DEBUG

            # --- Use Pure BALD for Acquisition ---
            self.acq_on_grid = bald_acq
            print("Acquisition calculated (BALD only).") # DEBUG PRINT

            # --- Find next point based on BALD, refined by distance ---
            # Find the next point to sample using BALD and distance-based refinement
            if len(self.X_samples) >= 12:
                # Find indices of the top 2.5% of acquisition values (pure BALD)
                num_top_acq = max(1, int(len(self.acq_on_grid) * 0.025)) # Ensure at least 1
                # Handle potential NaN/Inf values in BALD before sorting
                finite_acq = np.nan_to_num(self.acq_on_grid, nan=-np.inf)
                if np.all(~np.isfinite(finite_acq)) or len(finite_acq) == 0:
                    # Fallback if all acquisition values are bad
                    print("Warning: All acquisition values are non-finite. Using distance heuristic.")
                    self._distance_heuristic() # Use fallback
                    # Need to ensure next_pt is set after heuristic
                    if not hasattr(self, 'next_pt') or self.next_pt is None:
                         if hasattr(self, 'next_pt_norm') and self.next_pt_norm is not None:
                             self.next_pt = self._denormalize_point(self.next_pt_norm)
                         else: self.next_pt = self._denormalize_point(np.array([0.5, 0.5]))
                    # Update plot and status after heuristic
                    self._trigger_update()
                    return False # Indicate fit/acquisition process had issues

                top_acq_indices = np.argsort(finite_acq)[-num_top_acq:]

                # Calculate distances to known points (excluding initial seeds)
                top_acq_points = self.X_grid_norm[top_acq_indices]
                if len(self.X_samples_norm) > 8:
                     X_arr_norm = np.vstack(self.X_samples_norm)
                     known_points_for_dist = X_arr_norm[8:] # Exclude Halton seeds for distance check
                     max_min_dist_sq = -1.0
                     best_idx = top_acq_indices[-1] # Default to highest BALD if no known points or issue

                     if len(known_points_for_dist) > 0:
                          for i, point_idx in enumerate(top_acq_indices):
                               point = self.X_grid_norm[point_idx]
                               point_dists_sq = np.sum((known_points_for_dist - point)**2, axis=1)
                               min_dist_sq = np.min(point_dists_sq)
                               if min_dist_sq > max_min_dist_sq:
                                   max_min_dist_sq = min_dist_sq
                                   best_idx = point_idx # Select point furthest from existing non-seed samples

                     next_pt_idx = best_idx
                else:
                     # Not enough non-seed points yet, just pick the max acquisition value
                     next_pt_idx = top_acq_indices[-1] # Index with highest finite BALD

            else:
                # If fewer than 12 samples, just pick the max acquisition value
                finite_acq = np.nan_to_num(self.acq_on_grid, nan=-np.inf)
                if np.all(~np.isfinite(finite_acq)) or len(finite_acq) == 0:
                    # Fallback if all acquisition values are bad
                    print("Warning: All acquisition values are non-finite. Using distance heuristic.")
                    self._distance_heuristic() # Use fallback
                    if not hasattr(self, 'next_pt') or self.next_pt is None:
                         if hasattr(self, 'next_pt_norm') and self.next_pt_norm is not None:
                             self.next_pt = self._denormalize_point(self.next_pt_norm)
                         else: self.next_pt = self._denormalize_point(np.array([0.5, 0.5]))
                    self._trigger_update()
                    return False # Indicate fit/acquisition process had issues

                next_pt_idx = np.argmax(finite_acq)

            self.next_pt_norm = self.X_grid_norm[next_pt_idx]
            self.next_pt = self._denormalize_point(self.next_pt_norm)
            print(f"Next point selected (norm): {self.next_pt_norm}, (real): {self.next_pt} (y is log10)") # DEBUG PRINT

            print("--- GP Fitting/Acquisition Successful ---") # DEBUG PRINT
            # # --- sanity check ---------------------------------------------------
            # p_train, _ = self.gp_model.predict_y(X_tf)          # P(detect) for training points
            # err = np.abs(p_train.numpy().ravel() - y_arr.ravel())
            # print(f"mean|pred‑label| on training set = {err.mean():.3f}")
            # print(np.column_stack([self.y_samples, p_train.numpy().ravel()]))
            # # --------------------------------------------------------------------
            return True # Success
        
        except Exception as e:
            # --- DEBUG PRINT ---
            print(f"\n!!! EXCEPTION DURING GP FIT/PREDICT !!!")
            print(f"Error type: {type(e)}")
            print(f"Error message: {e}")
            import traceback
            print("Traceback:")
            traceback.print_exc() # Print full traceback
            print("!!! Falling back to distance heuristic !!!\n")
            # --- END DEBUG PRINT ---

            # Fallback: Use distance heuristic
            self._distance_heuristic()
            if not hasattr(self, 'next_pt') or self.next_pt is None:
                 if hasattr(self, 'next_pt_norm') and self.next_pt_norm is not None:
                     self.next_pt = self._denormalize_point(self.next_pt_norm)
                 else: self.next_pt = self._denormalize_point(np.array([0.5, 0.5]))
            # Clear GP results
            self.prob_mean_on_grid = None
            self.latent_mean_on_grid = None
            self.latent_var_on_grid = None
            # acq_on_grid is set by heuristic
            return False # Indicate failure


    def _distance_heuristic(self):
        """Simple distance-based heuristic when GP model isn't available."""
        # Create a dummy probability map (e.g., constant 0.5)
        self.prob_mean_on_grid = np.full(self.grid_res * self.grid_res, 0.5)
        self.display_mean_on_grid = self.prob_mean_on_grid.copy()
        self.latent_mean_on_grid = np.zeros(self.grid_res * self.grid_res) # Assume 0 latent mean
        self.latent_var_on_grid = np.ones(self.grid_res * self.grid_res) # Assume high variance

        # Acquisition is distance from known points (in normalized space)
        if len(self.X_samples_norm) > 0:
            X_known_norm = np.vstack(self.X_samples_norm)
            diffs = self.X_grid_norm[:, np.newaxis, :] - X_known_norm[np.newaxis, :, :]
            min_sq_dist = np.min(np.sum(diffs**2, axis=2), axis=1)
            self.acq_on_grid = min_sq_dist # Maximize distance
        else:
            # If zero samples, pick the center point
            self.acq_on_grid = -np.sum((self.X_grid_norm - 0.5)**2, axis=1) # Minimize dist to center

        # Select point furthest from existing samples (or closest to center if none)
        next_pt_idx = np.argmax(self.acq_on_grid)
        self.next_pt_norm = self.X_grid_norm[next_pt_idx]
        # self.next_pt will be set outside after denormalization

    # ─── GUI update Trigger ─── #
    def _trigger_update(self):
         """Starts the timer to defer the plot update."""
         self.update_timer.start()

    def _update_plot_and_status(self):
        """Updates the plot and status bar (called by timer)."""
        if not self.initialized:
            self.statusBar().showMessage("Not initialized.")
            return

        # Update plot
        self.plot.refresh(
            self.X_grid, # Pass grid with log contrast
            self.prob_mean_on_grid, # For contour
            self.next_pt,
            self.var1_lbl,
            self.var2_lbl, # This already includes log10 indicator
            self.X_samples,
            self.y_samples,
            display_mean_prob=self.display_mean_on_grid # For heatmap
        )

        # Update status bar
        max_acq_curr = 0.0
        if self.acq_on_grid is not None and len(self.acq_on_grid) > 0:
            max_acq_curr = float(np.max(self.acq_on_grid))

        if self.next_pt is not None:
             # Format next point with units
             x_unit = self.unit1.text()
             y_unit = self.unit2.text()
             next_pt_str = f"{self.next_pt[0]:.3f} {x_unit}, {self.next_pt[1]:.3f} {y_unit}"
             self.nextLab.setText(f"Sample: {next_pt_str}")
             status_msg = (
                f"Next: {next_pt_str} | Max Acq: {max_acq_curr:.3f} bits | "
                f"Samples: {len(self.X_samples)} | LML: {self.last_log_marginal_likelihood:.2f}"
             )
        else: # Converged or error
             next_pt_str = "N/A (Converged or Error)"
             self.nextLab.setText(f"Sample: {next_pt_str}")
             status_msg = (
                 f"Converged or Error | Max Acq: {max_acq_curr:.3f} bits | "
                 f"Samples: {len(self.X_samples)} | LML: {self.last_log_marginal_likelihood:.2f}"
            )

        self.statusBar().showMessage(status_msg)

    # ─── Fit, Update, and Convergence Check ─── #
    def _fit_and_update(self):
         """Runs the fitting process and triggers the deferred update."""
         start_time = time.time()
         self.statusBar().showMessage("Fitting GP model...")
         QApplication.processEvents() # Keep UI responsive

         # Store current max acquisition value *before* refitting
         max_acq_prev = 0.0
         if self.acq_on_grid is not None and len(self.acq_on_grid) > 0:
             max_acq_prev = float(np.max(self.acq_on_grid))

         # Fit the model and calculate next point
         fit_success = self._fit()
         fit_time = time.time() - start_time

         print(f"Fit time: {fit_time:.2f}s") # Log fit time

         # Trigger the deferred update for plot and status
         self._trigger_update()

         # --- Convergence Check ---
         # Stop if the *utility* of adding the point we just added was low,
         # OR if max trials reached, OR if fit failed repeatedly? (handle later)
         converged = False
         if not fit_success:
              print("Fit failed, considering convergence...")
              # Simple rule: if fit fails maybe stop? Or allow continuation?
              # Let's allow continuation for now, relying on max trials.
              pass
         elif ((max_acq_prev < self.BALD_STOP_BITS and len(self.y_samples) > 10) or # Min samples condition
             (len(self.X_samples) >= self.MAX_TRIALS)):
              converged = True
              self.next_pt = None # Signal convergence by clearing next point
              self.next_pt_norm = None
              self.btnDet.setEnabled(False); self.btnMiss.setEnabled(False)
              final_msg = (f"Converged after {len(self.X_samples)} trials. "
                           f"Max Acq before last step: {max_acq_prev:.3f} bits "
                           f"(Stop Threshold: {self.BALD_STOP_BITS}).")
              self.statusBar().showMessage(final_msg)
              print(final_msg)
              QMessageBox.information(self, "Converged", final_msg)
              # Trigger one final update to show state without next point marker
              self._trigger_update()


    # ─── record user response ─── #
    def _record(self, outcome: int):
        if not self.initialized or self.next_pt is None or self.next_pt_norm is None:
            self.statusBar().showMessage("Cannot record response: Not initialized or no next point.")
            return

        # Add data (real and normalized)
        self.X_samples.append(self.next_pt.copy())
        self.X_samples_norm.append(self.next_pt_norm.copy())
        self.y_samples.append(outcome)

        # Refit and find next point (this also handles convergence check)
        self._fit_and_update()


    # ─── if bounds edited mid‑run ─── #
    def _bounds_changed(self):
        if self.initialized:
            reply = QMessageBox.warning(self, "Bounds Changed",
                "Changing bounds mid-run requires re-normalizing points and "
                "refitting the model. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                # TODO: Restore previous bounds if possible (need to store them)
                return

            # Recalculate grid and normalized samples
            self._make_grid()
            if len(self.X_samples) > 0:
                self.X_samples_norm = [self._normalize_point(x) for x in self.X_samples]
            else:
                self.X_samples_norm = []

            # Force model rebuild and refit
            self.gp_model = None
            self._fit_and_update()
            self.statusBar().showMessage("Bounds changed. Model reset and refit.")


    # ─── save CSV + PNG + CSF contour ─── #
    def _save(self):
        if not self.initialized:
            QMessageBox.warning(self, "Save Error", "Cannot save before initializing.")
            return

        name, ok = QInputDialog.getText(self, "Save", "Folder name:")
        if not ok or not name: return
        try:
            out = Path("mlcsfOutputs_GPflow") / name; out.mkdir(parents=True, exist_ok=True) # Separate folder

            # CSV of raw trials (real-space)
            with (out / "data.csv").open("w", newline="") as f:
                wr = csv.writer(f); wr.writerow([self.var1_lbl, self.var2_lbl, "outcome"])
                min_len = min(len(self.X_samples), len(self.y_samples))
                for i in range(min_len):
                    wr.writerow([self.X_samples[i][0], self.X_samples[i][1], self.y_samples[i]])

            # Posterior surface image
            self.plot.export_png(out / "posterior.png", len(self.y_samples))

            # 50 % CSF curve (from the non-sharpened posterior mean probability)
            self._save_csf_curve(out / "csf_curve.csv")

            # Optional: Save GP model parameters
            if self.gp_model:
                 with (out / "gp_model_summary.txt").open("w") as f:
                      f.write(print_summary(self.gp_model, fmt='notebook'))
                 # Could also save parameters using gpflow.utilities.parameter_dict

            QMessageBox.information(self, "Saved", f"Results written to {out}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save results: {e}")
            print(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()


    def _save_csf_curve(self, path):
        # Use the calculated mean probability (incorporating gamma/lambda approx)
        if self.prob_mean_on_grid is None:
            print("Cannot save CSF curve, posterior mean probability not available.")
            return

        prob = self.prob_mean_on_grid.reshape(self.grid_res, self.grid_res, order="F")
        curve = []

        # Iterate through real-space frequency axis (self.xs - axis 1 in reshaped prob)
        for j, x_freq in enumerate(self.xs): # Index j corresponds to x-axis (frequency)
            # Get the probability column for this frequency (axis 0 is contrast/y)
            log_contrast_probs = prob[:, j]

            # Find log contrast where probability crosses 0.5
            indices_above_threshold = np.where(log_contrast_probs >= 0.5)[0]

            if len(indices_above_threshold) > 0:
                first_idx = indices_above_threshold[0]
                # Linear interpolation for smoother threshold
                if first_idx == 0:
                    threshold_log_contrast = self.ys[0] # Threshold at or below lowest log contrast
                else:
                    p1 = log_contrast_probs[first_idx - 1]
                    p2 = log_contrast_probs[first_idx]
                    y1 = self.ys[first_idx - 1]
                    y2 = self.ys[first_idx]
                    # Avoid division by zero / handle flat regions
                    if p2 - p1 > 1e-6:
                        threshold_log_contrast = y1 + (y2 - y1) * (0.5 - p1) / (p2 - p1)
                    else: # If probabilities are too close, just take the boundary log contrast
                        threshold_log_contrast = self.ys[first_idx]
                # Ensure threshold is within bounds due to interpolation overshoot
                threshold_log_contrast = np.clip(threshold_log_contrast, self.ys[0], self.ys[-1])
                curve.append((x_freq, threshold_log_contrast))
            else:
                # If no contrast reaches 0.5 threshold for this frequency
                curve.append((x_freq, float('nan'))) # Use NaN

        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([self.var1_lbl, f"Threshold {self.var2.text()} (50%)"])
            wr.writerows(curve)
            
        # Also save a version with converted linear contrast values
        # Fix the Path.replace error - convert Path to string first
        linear_path = str(path).replace(".csv", "_linear.csv")
        with open(linear_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([self.var1_lbl, f"Threshold Contrast % (50%)"])
            # Convert log contrast to linear contrast percentage
            linear_curve = [(freq, 100 * 10**log_contrast if not np.isnan(log_contrast) else float('nan')) 
                            for freq, log_contrast in curve]
            wr.writerows(linear_curve)

    def _update_style(self, style: str):
        """Update plot style based on dropdown selection"""
        self.scientific_style = (style == "Scientific")
        self.colorblind_style = (style == "Colorblind")
        self.plot.set_scientific_style(self.scientific_style)
        self.plot.set_colorblind_style(self.colorblind_style)
        if self.initialized:
            self._trigger_update() # Use trigger


# ──────────────────────────── entrypoint ──────────────────────────────── #
def main():
    app = QApplication(sys.argv)
    # Set default font (optional, improves cross-platform consistency)
    # font = fm.FontProperties(family='sans-serif') # Or specific like 'Arial', 'DejaVu Sans'
    # plt.rcParams['font.family'] = font.get_name()

    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # Set GPU memory growth if using GPU (prevents TensorFlow from grabbing all memory)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled GPU memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"Could not set GPU memory growth: {e}")

    main()