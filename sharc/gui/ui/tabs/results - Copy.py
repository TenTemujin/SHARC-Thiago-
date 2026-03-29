import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import numpy as np
import threading
import io
import webbrowser
import traceback
from pathlib import Path

# Image handling for Tkinter
from PIL import Image, ImageTk

# Plotly Imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Attempt to import config, fallback if missing
try:
    from config import RESULT_FIELDNAME_TO_PLOT_INFO
except ImportError:
    RESULT_FIELDNAME_TO_PLOT_INFO = {}


class ResultsTab:
    """
    Results Tab (Plotly Version - Optimized):
    - Detailed per-subplot configuration (Titles, Axes, Criteria).
    - Reactive Auto-Refresh with Debounce.
    - Intuitive File Management.
    - Robust Error Handling for Rendering.
    """

    def __init__(self, app, parent_frame):
        self.app = app
        self.frame = parent_frame

        # --- State Management ---
        if not hasattr(self.app, "res_dirs"):
            self.app.res_dirs = []  # List of full directory paths

        # Caches
        self._data_cache = {}    # (folder_path, field) -> (mtime, numpy_array)
        self._plot_preview_job = None
        self._photo_image = None  # Keep reference to prevent GC
        self._max_axes = 9       # Max grid size (3x3)
        self._disable_traces = False  # Flag to prevent loops during UI updates

        # Initialize Subplot Configurations (List of dicts)
        self.result_fields = sorted(list(RESULT_FIELDNAME_TO_PLOT_INFO.keys()))
        if not self.result_fields:
            self.result_fields = ["ExampleField"]

        default_field = self.result_fields[0]

        # We store config for 9 potential subplots
        self._axes_cfg = []
        for i in range(self._max_axes):
            self._axes_cfg.append({
                "field": default_field,
                "mode": "CDF",      # CDF or CCDF
                "title": "",        # Custom Title override
                "x_label": "",      # Custom X Label override
                "y_label": "",      # Custom Y Label override
                "x_log": False,     # Log Scale X
                "x_shift": 0.0,     # Shift X data
                "legend_suffix": "",
                # List of dicts: {val, type='v'/'h', label, color}
                "criteria": []
            })

        # --- Initialize Tkinter Variables ---
        self._init_vars()

        # --- Build UI ---
        self._build_ui()

        # Initial Plot Trigger
        self._schedule_update()

    def _init_vars(self):
        """Initialize Tkinter variables linked to UI widgets."""
        # Grid Layout
        self.var_rows = tk.IntVar(value=1)
        self.var_cols = tk.IntVar(value=1)

        # Selected Subplot Index (0 to 8)
        self.var_current_subplot_idx = tk.IntVar(value=0)

        # Editing Variables (Linked to the currently selected subplot)
        self.var_edit_field = tk.StringVar()
        self.var_edit_mode = tk.StringVar()
        self.var_edit_title = tk.StringVar()
        self.var_edit_xlabel = tk.StringVar()
        self.var_edit_ylabel = tk.StringVar()
        self.var_edit_xlog = tk.BooleanVar()
        self.var_edit_ylog = tk.BooleanVar()
        self.var_edit_leg_suffix = tk.StringVar()
        self.var_edit_xshift = tk.DoubleVar(value=0.0)

        # Trace variables to trigger auto-refresh on change
        self._trace_vars = [
            self.var_edit_field, self.var_edit_mode, self.var_edit_title,
            self.var_edit_xlabel, self.var_edit_ylabel, self.var_edit_xlog,
            self.var_edit_ylog, self.var_edit_leg_suffix, self.var_rows, self.var_cols
        ]
        for v in self._trace_vars:
            v.trace_add("write", self._on_config_change)

        # Trace shift separately to handle float errors gracefully
        self.var_edit_xshift.trace_add("write", self._on_config_change)

    def _build_ui(self):
        """Constructs the split-pane UI."""
        # Main Split Container
        paned = ttk.PanedWindow(self.frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Slightly wider for better layout
        left_frame = ttk.Frame(paned, width=420)
        right_frame = ttk.Frame(paned)

        paned.add(left_frame, weight=0)  # Controls (Fixed width)
        paned.add(right_frame, weight=1)  # Plot (Expandable)

        # ================= LEFT: CONTROLS =================
        self._build_file_manager(left_frame)
        self._build_layout_controls(left_frame)
        self._build_subplot_config(left_frame)

        # ================= RIGHT: PREVIEW =================
        self._build_plot_preview(right_frame)

        # Load initial values
        self._load_subplot_config_to_ui()

    def _build_file_manager(self, parent):
        """Builds the file list and buttons."""
        frm = ttk.LabelFrame(parent, text="Result Folders")
        frm.pack(fill="x", padx=5, pady=5)

        # Listbox with Scrollbar
        list_frame = ttk.Frame(frm)
        list_frame.pack(fill="x", padx=5, pady=5)

        self.lb_dirs = tk.Listbox(list_frame, height=5, selectmode="extended")
        self.lb_dirs.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, command=self.lb_dirs.yview)
        sb.pack(side="right", fill="y")
        self.lb_dirs.config(yscrollcommand=sb.set)

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.pack(fill="x", padx=5, pady=2)

        ttk.Button(btn_frame, text="Add Folder...",
                   command=self._add_dir).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Remove", command=self._remove_dir).pack(
            side="left", padx=2)
        ttk.Button(btn_frame, text="Clear All",
                   command=self._clear_all_dirs).pack(side="right", padx=2)

    def _build_layout_controls(self, parent):
        """Global Grid Layout Config."""
        frm = ttk.LabelFrame(parent, text="Grid Layout")
        frm.pack(fill="x", padx=5, pady=5)

        ttk.Label(frm, text="Rows:").pack(side="left", padx=5)
        ttk.Spinbox(frm, from_=1, to=3, textvariable=self.var_rows,
                    width=3).pack(side="left")
        ttk.Label(frm, text="Cols:").pack(side="left", padx=5)
        ttk.Spinbox(frm, from_=1, to=3, textvariable=self.var_cols,
                    width=3).pack(side="left")

    def _build_subplot_config(self, parent):
        """Detailed configuration for the active subplot."""
        frm = ttk.LabelFrame(parent, text="Active Subplot Settings")
        frm.pack(fill="both", expand=True, padx=5, pady=5)

        # Selector
        sel_frame = ttk.Frame(frm)
        sel_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(sel_frame, text="Editing Subplot:").pack(side="left")

        self.cb_subplot_sel = ttk.Combobox(
            sel_frame, state="readonly", width=5)
        self.cb_subplot_sel['values'] = [
            str(i+1) for i in range(self._max_axes)]
        self.cb_subplot_sel.current(0)
        self.cb_subplot_sel.pack(side="left", padx=5)
        self.cb_subplot_sel.bind(
            "<<ComboboxSelected>>", self._on_subplot_selection_change)

        # Tabs for Config
        nb = ttk.Notebook(frm)
        nb.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Tab 1: Data & Axis ---
        tab_axis = ttk.Frame(nb)
        nb.add(tab_axis, text="Data & Axis")

        grid_opts = {'padx': 5, 'pady': 3, 'sticky': 'w'}

        # Field & Mode
        ttk.Label(tab_axis, text="CSV Field:").grid(
            row=0, column=0, **grid_opts)
        ttk.Combobox(tab_axis, textvariable=self.var_edit_field,
                     values=self.result_fields).grid(row=0, column=1, sticky="ew")

        ttk.Label(tab_axis, text="Mode:").grid(row=1, column=0, **grid_opts)
        ttk.Combobox(tab_axis, textvariable=self.var_edit_mode, values=[
                     "CDF", "CCDF"], state="readonly").grid(row=1, column=1, sticky="ew")

        # Titles & Labels
        ttk.Separator(tab_axis, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(tab_axis, text="Chart Title:").grid(
            row=3, column=0, **grid_opts)
        ttk.Entry(tab_axis, textvariable=self.var_edit_title).grid(
            row=3, column=1, sticky="ew")

        ttk.Label(tab_axis, text="X Label:").grid(row=4, column=0, **grid_opts)
        ttk.Entry(tab_axis, textvariable=self.var_edit_xlabel).grid(
            row=4, column=1, sticky="ew")

        ttk.Label(tab_axis, text="Y Label:").grid(row=5, column=0, **grid_opts)
        ttk.Entry(tab_axis, textvariable=self.var_edit_ylabel).grid(
            row=5, column=1, sticky="ew")

        # Scales & Shift
        ttk.Separator(tab_axis, orient="horizontal").grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=5)

        chk_frame = ttk.Frame(tab_axis)
        chk_frame.grid(row=7, column=0, columnspan=2, sticky="w", padx=5)
        ttk.Checkbutton(chk_frame, text="Log Scale X", variable=self.var_edit_xlog).pack(
            side="left", padx=(0, 10))
        ttk.Checkbutton(chk_frame, text="Log Scale Y",
                        variable=self.var_edit_ylog).pack(side="left")

        shift_frame = ttk.Frame(tab_axis)
        shift_frame.grid(row=8, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(shift_frame, text="X Shift:").pack(side="left", padx=5)
        ttk.Entry(shift_frame, textvariable=self.var_edit_xshift,
                  width=6).pack(side="left")
        ttk.Label(shift_frame, text="Legend Suffix:").pack(side="left", padx=5)
        ttk.Entry(shift_frame, textvariable=self.var_edit_leg_suffix).pack(
            side="left", fill="x", expand=True, padx=5)

        tab_axis.columnconfigure(1, weight=1)

        # --- Tab 2: Protection Criteria ---
        tab_crit = ttk.Frame(nb)
        nb.add(tab_crit, text="Protection Criteria")

        # Treeview
        cols = ("val", "type", "label")
        self.tv_crit = ttk.Treeview(
            tab_crit, columns=cols, show="headings", height=6)
        self.tv_crit.heading("val", text="Value")
        self.tv_crit.column("val", width=60)
        self.tv_crit.heading("type", text="Type")
        self.tv_crit.column("type", width=80)
        self.tv_crit.heading("label", text="Label")
        self.tv_crit.column("label", width=100)
        self.tv_crit.pack(fill="both", expand=True, padx=5, pady=5)

        btn_crit = ttk.Frame(tab_crit)
        btn_crit.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_crit, text="Remove Sel",
                   command=self._remove_criteria).pack(side="right")
        ttk.Button(btn_crit, text="Add Criteria...",
                   command=self._add_criteria_dialog).pack(side="right", padx=5)

    def _build_plot_preview(self, parent):
        """Right side: Toolbar and Image Preview."""
        toolbar = ttk.Frame(parent)
        toolbar.pack(side="top", fill="x", padx=5, pady=5)

        ttk.Button(toolbar, text="Refresh ",
                   command=self._update_plot_preview).pack(side="left")
        ttk.Button(toolbar, text="Open Interactive (Browser)",
                   command=self._open_browser).pack(side="right", padx=5)

        # Preview Container
        self.preview_frame = ttk.Frame(parent, relief="sunken", borderwidth=1)
        self.preview_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.lbl_preview = ttk.Label(
            self.preview_frame, text="Initializing Preview...", anchor="center")
        self.lbl_preview.pack(fill="both", expand=True)

        # Bind resize to update preview size
        self.preview_frame.bind("<Configure>", self._on_resize)

    # ---------------- UI Logic & Events ----------------

    def _on_subplot_selection_change(self, event=None):
        """Updates UI vars when user changes the active subplot combo."""
        try:
            idx = int(self.cb_subplot_sel.get()) - 1
        except ValueError:
            idx = 0
        self.var_current_subplot_idx.set(idx)

        # Load config into UI
        self._load_subplot_config_to_ui()

    def _load_subplot_config_to_ui(self):
        """Loads the dict config into the Tkinter variables."""
        idx = self.var_current_subplot_idx.get()
        cfg = self._axes_cfg[idx]

        self._disable_traces = True  # Pause monitoring

        self.var_edit_field.set(cfg["field"])
        self.var_edit_mode.set(cfg["mode"])
        self.var_edit_title.set(cfg.get("title", ""))
        self.var_edit_xlabel.set(cfg.get("x_label", ""))
        self.var_edit_ylabel.set(cfg.get("y_label", ""))
        self.var_edit_xlog.set(cfg.get("x_log", False))
        self.var_edit_ylog.set(cfg.get("y_log", False))
        self.var_edit_leg_suffix.set(cfg.get("legend_suffix", ""))
        self.var_edit_xshift.set(cfg.get("x_shift", 0.0))

        self._refresh_criteria_list(cfg.get("criteria", []))

        self._disable_traces = False  # Resume monitoring

    def _on_config_change(self, *args):
        """Called when any edit variable changes."""
        if self._disable_traces:
            return

        # Save UI state back to Config Dict
        idx = self.var_current_subplot_idx.get()
        cfg = self._axes_cfg[idx]

        cfg["field"] = self.var_edit_field.get()
        cfg["mode"] = self.var_edit_mode.get()
        cfg["title"] = self.var_edit_title.get()
        cfg["x_label"] = self.var_edit_xlabel.get()
        cfg["y_label"] = self.var_edit_ylabel.get()
        cfg["x_log"] = self.var_edit_xlog.get()
        cfg["y_log"] = self.var_edit_ylog.get()
        cfg["legend_suffix"] = self.var_edit_leg_suffix.get()

        try:
            cfg["x_shift"] = self.var_edit_xshift.get()
        except tk.TclError:
            pass  # Ignore temporary float errors during typing

        self._schedule_update()

    def _schedule_update(self):
        """Debounce timer: updates plot 600ms after last change."""
        if hasattr(self, "_update_timer") and self._update_timer:
            self.app.after_cancel(self._update_timer)
        self._update_timer = self.app.after(250, self._update_plot_preview)

    def _on_resize(self, event):
        """Handle window resize with debounce."""
        if hasattr(self, "_resize_timer") and self._resize_timer:
            self.app.after_cancel(self._resize_timer)
        self._resize_timer = self.app.after(700, self._update_plot_preview)

    # ---------------- File Management ----------------

    def _refresh_dir_listbox(self):
        self.lb_dirs.delete(0, "end")
        for p in self.app.res_dirs:
            # Show only folder name for cleanliness, tooltip could show full path
            name = os.path.basename(os.path.normpath(p))
            if not name:
                name = p
            self.lb_dirs.insert("end", name)

    def _add_dir(self):
        start = getattr(self, "_last_dir", os.getcwd())
        path = filedialog.askdirectory(
            initialdir=start, title="Select Result Folder")
        if path:
            self._last_dir = path
            if path not in self.app.res_dirs:
                self.app.res_dirs.append(path)
                self._refresh_dir_listbox()
                self._schedule_update()

    def _remove_dir(self):
        sel = list(self.lb_dirs.curselection())
        for i in reversed(sel):
            del self.app.res_dirs[i]
        self._refresh_dir_listbox()
        self._schedule_update()

    def _clear_all_dirs(self):
        if messagebox.askyesno("Confirm", "Clear all folders?"):
            self.app.res_dirs.clear()
            self._refresh_dir_listbox()
            self._schedule_update()

    # ---------------- Criteria Dialog ----------------

    def _refresh_criteria_list(self, criteria_list):
        self.tv_crit.delete(*self.tv_crit.get_children())
        for i, c in enumerate(criteria_list):
            self.tv_crit.insert("", "end", iid=str(i), values=(
                c.get("val"), c.get("type"), c.get("label")))

    def _add_criteria_dialog(self):
        idx = self.var_current_subplot_idx.get()
        cfg = self._axes_cfg[idx]

        win = tk.Toplevel(self.frame)
        win.title("Add Criteria")
        win.geometry("320x250")

        v_val = tk.DoubleVar()
        v_type = tk.StringVar(value="Vertical (X)")
        v_label = tk.StringVar()
        v_color = tk.StringVar(value="red")

        ttk.Label(win, text="Value (X or Probability):").pack(pady=(10, 0))
        ttk.Entry(win, textvariable=v_val).pack()

        ttk.Label(win, text="Type:").pack(pady=(5, 0))
        ttk.Combobox(win, textvariable=v_type, values=[
                     "Vertical (X)", "Horizontal (Prob)"], state="readonly").pack()

        ttk.Label(win, text="Label (Legend):").pack(pady=(5, 0))
        ttk.Entry(win, textvariable=v_label).pack()

        ttk.Label(win, text="Color:").pack(pady=(5, 0))
        ttk.Combobox(win, textvariable=v_color, values=[
                     "red", "green", "blue", "black", "orange", "purple"]).pack()

        def _save():
            try:
                new_c = {
                    "val": v_val.get(),
                    "type": v_type.get(),
                    "label": v_label.get(),
                    "color": v_color.get()
                }
                cfg["criteria"].append(new_c)
                self._refresh_criteria_list(cfg["criteria"])
                self._schedule_update()
                win.destroy()
            except tk.TclError:
                messagebox.showerror("Error", "Invalid Number")

        ttk.Button(win, text="Add", command=_save).pack(pady=15)

    def _remove_criteria(self):
        sel = self.tv_crit.selection()
        if not sel:
            return
        idx = self.var_current_subplot_idx.get()
        cfg = self._axes_cfg[idx]

        i = int(sel[0])
        if 0 <= i < len(cfg["criteria"]):
            cfg["criteria"].pop(i)
            self._refresh_criteria_list(cfg["criteria"])
            self._schedule_update()

    # ---------------- Data & Plot Logic ----------------

    def _get_data(self, folder, field):
        """Reads CSV with caching and error handling."""
        fpath = os.path.join(folder, f"{field}.csv")
        if not os.path.exists(fpath):
            return None

        try:
            mtime = os.path.getmtime(fpath)
            key = (folder, field)
            if key in self._data_cache:
                cm, data = self._data_cache[key]
                if cm == mtime:
                    return data

            # Read CSV
            df = pd.read_csv(fpath)
            if df.empty:
                return None

            # Find column (flexible matching)
            target_cols = [c for c in df.columns if field.lower(
            ) in c.lower() or "value" in c.lower()]

            data = None
            if target_cols:
                data = df[target_cols[0]].dropna().values
            elif df.shape[1] > 0:
                data = df.iloc[:, 0].dropna().values

            if data is not None and len(data) > 0:
                self._data_cache[key] = (mtime, data)
                return data

        except Exception as e:
            print(f"Error reading {fpath}: {e}")
        return None

    def _compute_ecdf(self, x, ccdf=False):
        x = np.sort(x)
        n = x.size
        if n == 0:
            return [], []
        y = np.arange(1, n+1) / n
        if ccdf:
            y = 1.0 - y
        return x, y

    def _create_plotly_fig(self):
        """Generates the Plotly Figure object with optimized titles and layout."""
        rows = max(1, self.var_rows.get())
        cols = max(1, self.var_cols.get())
        n_plots = min(rows * cols, self._max_axes)

        # 1. Pre-calculate titles for subplots to avoid manual annotation hacking
        subplot_titles = []
        for i in range(n_plots):
            cfg = self._axes_cfg[i]
            field = cfg["field"]
            # Default title if custom is empty
            t = cfg.get("title")
            if not t:
                t = RESULT_FIELDNAME_TO_PLOT_INFO.get(
                    field, {}).get("title", field)
            subplot_titles.append(t)

        # 2. Create Subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            subplot_titles=subplot_titles
        )

        folders = self.app.res_dirs

        for i in range(n_plots):
            cfg = self._axes_cfg[i]
            r = (i // cols) + 1
            c = (i % cols) + 1

            field = cfg["field"]

            # --- Plot Data Traces ---
            for folder in folders:
                data = self._get_data(folder, field)
                if data is None or len(data) == 0:
                    continue

                x, y = self._compute_ecdf(data, ccdf=(cfg["mode"] == "CCDF"))
                x = x + cfg.get("x_shift", 0.0)

                # Robust Log Handling
                if cfg["y_log"]:
                    mask = y > 0
                    x, y = x[mask], y[mask]

                if cfg["x_log"]:
                    mask = x > 0
                    x, y = x[mask], y[mask]

                if len(x) == 0:
                    continue

                fname = os.path.basename(os.path.normpath(folder))
                name = f"{fname}{cfg.get('legend_suffix', '')}"

                # Use Scattergl for performance if many points
                trace_type = go.Scattergl if len(x) > 5000 else go.Scatter

                fig.add_trace(trace_type(
                    x=x, y=y, mode='lines', name=name,
                    legendgroup=folder,
                    # Only show legend for first plot to avoid clutter
                    showlegend=(i == 0)
                ), row=r, col=c)

            # --- Plot Criteria Lines ---
            for crit in cfg.get("criteria", []):
                try:
                    val = float(crit["val"])
                    ctype = crit["type"]
                    label = crit["label"]
                    color = crit.get("color", "red")

                    if "Vertical" in ctype:
                        fig.add_vline(x=val, line_width=1.5, line_dash="dash", line_color=color,
                                      annotation_text=label, annotation_position="top right",
                                      row=r, col=c)
                    else:
                        fig.add_hline(y=val, line_width=1.5, line_dash="dash", line_color=color,
                                      annotation_text=label, annotation_position="bottom right",
                                      row=r, col=c)
                except (ValueError, TypeError):
                    pass

            # --- Configure Axes ---
            xlab = cfg.get("x_label") or field
            x_type = "log" if cfg["x_log"] else "linear"

            ylab = cfg.get("y_label") or f"Prob ({cfg['mode']})"
            y_type = "log" if cfg["y_log"] else "linear"

            fig.update_xaxes(title_text=xlab, type=x_type, row=r, col=c,
                             showgrid=True, gridwidth=1, gridcolor='LightGray')

            fig.update_yaxes(title_text=ylab, type=y_type, row=r, col=c,
                             showgrid=True, gridwidth=1, gridcolor='LightGray')

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=50, r=20, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1),
            font=dict(size=11)
        )
        return fig

    # ---------------- Rendering ----------------

    def _update_plot_preview(self):
        """Threaded rendering of the static preview."""
        if self._plot_preview_job and self._plot_preview_job.is_alive():
            return

        w = self.lbl_preview.winfo_width()
        h = self.lbl_preview.winfo_height()
        # Cap dimensions to avoid excessively slow renders on resize
        w = max(400, min(w, 1920))
        h = max(300, min(h, 1080))

        self.lbl_preview.configure(text="Rendering...")

        self._plot_preview_job = threading.Thread(
            target=self._render_worker, args=(w, h))
        self._plot_preview_job.daemon = True
        self._plot_preview_job.start()

    def _render_worker(self, w, h):
        """Background worker to generate PNG."""
        try:
            fig = self._create_plotly_fig()

            # Static image generation requires 'kaleido'
            # We wrap this in a specific try/except block
            img_bytes = fig.to_image(format="png", width=w, height=h, scale=1)
            self.app.after(0, self._display_image, img_bytes)

        except ImportError:
            self.app.after(0, self._display_error,
                           "Static preview requires 'kaleido' package.\nClick 'Open Interactive' to view.")
        except Exception as e:
            # Handle kaleido crashes or other render errors
            err_msg = str(e)
            if "executable" in err_msg.lower() or "pipe" in err_msg.lower():
                err_msg = "Kaleido engine failed (OS limitation?).\nUse 'Open Interactive'."

            self.app.after(0, self._display_error, f"Render Error:\n{err_msg}")

    def _display_image(self, img_bytes):
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            self._photo_image = ImageTk.PhotoImage(pil_img)
            self.lbl_preview.configure(image=self._photo_image, text="")
        except Exception as e:
            self._display_error(str(e))

    def _display_error(self, msg):
        self.lbl_preview.configure(image="", text=msg)

    def _open_browser(self):
        """Opens the interactive Plotly chart in default browser."""
        try:
            fig = self._create_plotly_fig()
            import tempfile
            fd, path = tempfile.mkstemp(suffix=".html")
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(fig.to_html(include_plotlyjs='cdn'))
            webbrowser.open(f"file://{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open browser: {str(e)}")
