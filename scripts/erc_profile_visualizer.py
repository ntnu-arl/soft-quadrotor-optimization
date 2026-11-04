from __future__ import annotations

import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from erc_design import ERCDesignConfig, ERCDesigner, PiecewiseLinearTorqueProfile, SpringCatalog

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


DEFAULT_POINTS_DEG_NM = [
    (-0.25, -0.25),
    (0.10, 0.10),
    (30.0, 0.0),
    (76.0, 0.0),
    (78.0, 0.2),
]


class EditableTree(ttk.Treeview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._edit_entry: ttk.Entry | None = None
        self.bind("<Double-1>", self._begin_edit)

    def _begin_edit(self, event) -> None:
        region = self.identify("region", event.x, event.y)
        if region != "cell":
            return
        item = self.identify_row(event.y)
        column = self.identify_column(event.x)
        if not item or column == "#0":
            return

        bbox = self.bbox(item, column)
        if not bbox:
            return
        x, y, width, height = bbox
        col_index = int(column[1:]) - 1
        values = list(self.item(item, "values"))

        self._edit_entry = ttk.Entry(self)
        self._edit_entry.insert(0, values[col_index])
        self._edit_entry.select_range(0, tk.END)
        self._edit_entry.focus_set()
        self._edit_entry.place(x=x, y=y, width=width, height=height)

        def commit(_event=None) -> None:
            if self._edit_entry is None:
                return
            values[col_index] = self._edit_entry.get()
            self.item(item, values=values)
            self._edit_entry.destroy()
            self._edit_entry = None
            self.event_generate("<<CellEdited>>")

        self._edit_entry.bind("<Return>", commit)
        self._edit_entry.bind("<FocusOut>", commit)
        self._edit_entry.bind("<Escape>", lambda _event: self._cancel_edit())

    def _cancel_edit(self) -> None:
        if self._edit_entry is not None:
            self._edit_entry.destroy()
            self._edit_entry = None


class ERCPlotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ERC Cam Profile Designer")
        self.root.geometry("1320x850")

        self.repo_root = PROJECT_ROOT
        self.catalog = SpringCatalog.from_yaml(self.repo_root / "configs" / "springs.yaml")
        self.spring_ids = [spring.spring_id for spring in self.catalog]
        self.result = None

        self.spring_var = tk.StringVar(value=self.spring_ids[0])
        self.sf_var = tk.StringVar(value="1.1")
        self.grid_var = tk.StringVar(value="1000")
        self.ref_angle_var = tk.StringVar(value="0.0")
        self.min_angle_var = tk.StringVar(value="-90")
        self.max_angle_var = tk.StringVar(value="90")
        self.status_var = tk.StringVar(value="")

        self._drag_idx: int | None = None
        self._drag_xy: tuple[float, float] | None = None
        self._knot_artists: tuple | None = None  # (line, scatter)

        self._build_layout()
        self._populate_points(DEFAULT_POINTS_DEG_NM)
        self.recompute()

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main, width=360)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        controls.pack_propagate(False)

        plot_area = ttk.Frame(main)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(controls, text="Torque Profile").pack(anchor=tk.W)
        self.points_table = EditableTree(
            controls,
            columns=("angle_deg", "torque_nm"),
            show="headings",
            height=9,
            selectmode="browse",
        )
        self.points_table.heading("angle_deg", text="Angle [deg]")
        self.points_table.heading("torque_nm", text="Torque [Nm]")
        self.points_table.column("angle_deg", width=120, anchor=tk.E)
        self.points_table.column("torque_nm", width=120, anchor=tk.E)
        self.points_table.pack(fill=tk.X, pady=(4, 4))
        self.points_table.bind("<<CellEdited>>", lambda _event: self.recompute())

        row_buttons = ttk.Frame(controls)
        row_buttons.pack(fill=tk.X, pady=(0, 12))
        ttk.Button(row_buttons, text="Add Row", command=self.add_row).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(row_buttons, text="Delete Row", command=self.delete_row).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0)
        )

        ttk.Label(controls, text="Spring").pack(anchor=tk.W)
        spring_box = ttk.Combobox(
            controls, textvariable=self.spring_var, values=self.spring_ids, state="readonly"
        )
        spring_box.pack(fill=tk.X, pady=(4, 8))
        spring_box.bind("<<ComboboxSelected>>", lambda _event: self.recompute())

        self.spring_table = ttk.Treeview(
            controls, columns=("property", "value"), show="headings", height=7
        )
        self.spring_table.heading("property", text="Property")
        self.spring_table.heading("value", text="Value")
        self.spring_table.column("property", width=155)
        self.spring_table.column("value", width=150, anchor=tk.E)
        self.spring_table.pack(fill=tk.X, pady=(0, 12))

        numeric = ttk.LabelFrame(controls, text="Design Settings", padding=8)
        numeric.pack(fill=tk.X, pady=(0, 12))
        self._entry_row(numeric, "Min angle [deg]", self.min_angle_var)
        self._entry_row(numeric, "Max angle [deg]", self.max_angle_var)
        self._entry_row(numeric, "Safety factor", self.sf_var)
        self._entry_row(numeric, "Grid points", self.grid_var)
        self._entry_row(numeric, "Energy ref. [deg]", self.ref_angle_var)

        action_row = ttk.Frame(controls)
        action_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(action_row, text="Recompute", command=self.recompute).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(action_row, text="Export Curve", command=self.export_curve).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0)
        )

        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=330,
            foreground="#333333",
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(4, 0))

        self.fig = Figure(figsize=(9.5, 7.5), dpi=100, constrained_layout=True)
        self.ax_torque = self.fig.add_subplot(2, 2, 1)
        self.ax_cam = self.fig.add_subplot(2, 2, 2)
        self.ax_radius = self.fig.add_subplot(2, 2, 3)
        self.ax_curvature = self.fig.add_subplot(2, 2, 4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_area, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _entry_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable, justify=tk.RIGHT)
        entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda _event: self.recompute())
        entry.bind("<FocusOut>", lambda _event: self.recompute())

    def _populate_points(self, points: list[tuple[float, float]]) -> None:
        for item in self.points_table.get_children():
            self.points_table.delete(item)
        for angle_deg, torque_nm in points:
            self.points_table.insert("", tk.END, values=(f"{angle_deg:.6g}", f"{torque_nm:.6g}"))

    def add_row(self) -> None:
        points = self._read_points(raise_on_error=False)
        if len(points) >= 2:
            angle = points[-1][0] + (points[-1][0] - points[-2][0])
            torque = points[-1][1]
        elif len(points) == 1:
            angle = points[0][0] + 1.0
            torque = points[0][1]
        else:
            angle = 0.0
            torque = 0.0
        self.points_table.insert("", tk.END, values=(f"{angle:.6g}", f"{torque:.6g}"))
        self.recompute()

    def delete_row(self) -> None:
        selected = self.points_table.selection()
        if selected:
            self.points_table.delete(selected[0])
            self.recompute()

    def _read_points(self, *, raise_on_error: bool = True) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        try:
            for item in self.points_table.get_children():
                angle_deg, torque_nm = self.points_table.item(item, "values")
                points.append((float(angle_deg), float(torque_nm)))
        except ValueError:
            if raise_on_error:
                raise ValueError("Torque table entries must be numeric") from None
        points.sort(key=lambda point: point[0])
        return points

    def _read_config(self) -> ERCDesignConfig:
        safety_factor = float(self.sf_var.get())
        n_grid = int(float(self.grid_var.get()))
        ref_angle = math.radians(float(self.ref_angle_var.get()))
        return ERCDesignConfig(
            n_grid=n_grid,
            safety_factor=safety_factor,
            angle_convention="joint",
            joint_angle_limits_rad=(
                math.radians(float(self.min_angle_var.get())),
                math.radians(float(self.max_angle_var.get())),
            ),
            energy_reference_angle_rad=ref_angle,
            invert_output_torque=True,
        )

    def _read_angle_limits(self) -> tuple[float, float]:
        min_angle = float(self.min_angle_var.get())
        max_angle = float(self.max_angle_var.get())
        if min_angle >= max_angle:
            raise ValueError("Min angle must be smaller than max angle")
        return min_angle, max_angle

    def recompute(self) -> None:
        try:
            min_angle, max_angle = self._read_angle_limits()
            points = self._read_points()
            if len(points) < 2:
                raise ValueError("At least two torque-profile rows are required")
            lower = min(point[0] for point in points)
            upper = max(point[0] for point in points)
            if lower < min_angle or upper > max_angle:
                raise ValueError(
                    "Torque-profile angles must stay inside "
                    f"[{min_angle:g}, {max_angle:g}] deg"
                )
            self._populate_points(points)
            angles_deg = torch.tensor([point[0] for point in points], dtype=torch.float64)
            torques = torch.tensor([point[1] for point in points], dtype=torch.float64)
            profile = PiecewiseLinearTorqueProfile.from_xy(angles_deg * math.pi / 180.0, torques)

            spring = self.catalog[self.spring_var.get()]
            config = self._read_config()
            designer = ERCDesigner(spring, config)
            self.result = designer.design(profile)
            self._update_spring_table(spring)
            self._update_plots((min_angle, max_angle))
            self._draw_knots(points)
            self.canvas.draw_idle()
            self.status_var.set(
                "Admissible profile ready. "
                f"alpha={self.result.alpha:.6g}, "
                f"repaired={self.result.was_repaired}, "
                f"RMSE={self.result.torque_rmse_nm:.6g} Nm"
            )
        except Exception as exc:
            self.result = None
            self.status_var.set(f"Input error: {exc}")

    def _update_spring_table(self, spring) -> None:
        for item in self.spring_table.get_children():
            self.spring_table.delete(item)
        rows = [
            ("material", spring.material),
            ("wire_diameter_m", spring.wire_diameter_m),
            ("external_diameter_m", spring.external_diameter_m),
            ("free_length_m", spring.free_length_m),
            ("max_length_m", spring.max_length_m),
            ("spring_constant_n_per_m", spring.spring_constant_n_per_m),
            ("max_extension_m", spring.max_extension_m),
            ("tension_at_rest_n", spring.tension_at_rest_n),
            ("maximum_load_n", spring.maximum_load_n),
            ("count", spring.count),
            ("total_k_n_per_m", spring.total_stiffness_n_per_m),
            ("total_tmax_n", spring.total_max_tension_n),
        ]
        for key, value in rows:
            if value is None or value == "":
                continue
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            self.spring_table.insert("", tk.END, values=(key, text))

    def _update_plots(self, angle_limits_deg: tuple[float, float]) -> None:
        result = self.result
        if result is None:
            return

        angles_deg = (result.output_torque_table[:, 0].detach().cpu() * 180.0 / math.pi).numpy()
        target = result.target_torque_nm.detach().cpu().numpy()
        scaled = result.scaled_torque_nm.detach().cpu().numpy()
        admissible = result.repaired_torque_nm.detach().cpu().numpy()

        cam = result.cam_xy_m.detach().cpu().numpy() * 1000.0
        repaired_cam = result.repaired_cam_xy_m.detach().cpu().numpy() * 1000.0
        radius = result.support_radius_m.detach().cpu().numpy() * 1000.0
        repaired_radius = result.repaired_support_radius_m.detach().cpu().numpy() * 1000.0
        curvature = result.curvature_radius_m.detach().cpu().numpy() * 1000.0
        repaired_curvature = result.repaired_curvature_radius_m.detach().cpu().numpy() * 1000.0

        self._knot_artists = None
        self.ax_torque.clear()
        self.ax_torque.plot(angles_deg, target, color="#303030", label="requested")
        if result.was_scaled:
            self.ax_torque.plot(angles_deg, scaled, "--", color="#b7791f", label="energy scaled")
        self.ax_torque.plot(angles_deg, admissible, color="#0072b2", label="admissible")
        self.ax_torque.set_title("Torque Profile")
        self.ax_torque.set_xlabel("joint angle [deg]")
        self.ax_torque.set_ylabel("torque [Nm]")
        self.ax_torque.set_xlim(*angle_limits_deg)
        self.ax_torque.grid(True, alpha=0.3)
        self.ax_torque.legend()

        self.ax_cam.clear()
        self.ax_cam.plot(cam[:, 0], cam[:, 1], color="#777777", label="pre-repair")
        self.ax_cam.plot(repaired_cam[:, 0], repaired_cam[:, 1], color="#0072b2", label="admissible")
        self.ax_cam.scatter([0.0], [0.0], color="#303030", s=18)
        self.ax_cam.set_title("Cam Profile")
        self.ax_cam.set_xlabel("x [mm]")
        self.ax_cam.set_ylabel("y [mm]")
        self.ax_cam.axis("equal")
        self.ax_cam.grid(True, alpha=0.3)
        self.ax_cam.legend()

        self.ax_radius.clear()
        self.ax_radius.plot(angles_deg, radius, color="#777777", label="pre-repair")
        self.ax_radius.plot(angles_deg, repaired_radius, color="#0072b2", label="admissible")
        self.ax_radius.set_title("Support Radius")
        self.ax_radius.set_xlabel("joint angle [deg]")
        self.ax_radius.set_ylabel("r [mm]")
        self.ax_radius.set_xlim(*angle_limits_deg)
        self.ax_radius.grid(True, alpha=0.3)
        self.ax_radius.legend()

        self.ax_curvature.clear()
        self.ax_curvature.axhline(0.0, color="#303030", linewidth=1)
        self.ax_curvature.plot(angles_deg, curvature, color="#777777", label="pre-repair")
        self.ax_curvature.plot(angles_deg, repaired_curvature, color="#0072b2", label="admissible")
        self.ax_curvature.set_title("Convexity Radius")
        self.ax_curvature.set_xlabel("joint angle [deg]")
        self.ax_curvature.set_ylabel("r + r'' [mm]")
        self.ax_curvature.set_xlim(*angle_limits_deg)
        self.ax_curvature.grid(True, alpha=0.3)
        self.ax_curvature.legend()

    def _draw_knots(self, points: list[tuple[float, float]]) -> None:
        import numpy as np
        sorted_pts = sorted(points, key=lambda p: p[0])
        angles = [p[0] for p in sorted_pts]
        torques = [p[1] for p in sorted_pts]
        line, = self.ax_torque.plot(angles, torques, "--", color="#303030", lw=1, zorder=5)
        scat = self.ax_torque.scatter(angles, torques, color="#303030", s=40, zorder=6)
        self._knot_artists = (line, scat)

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax_torque or event.button != 1:
            return
        if getattr(self.canvas.toolbar, "mode", "") != "":
            return
        points = self._read_points(raise_on_error=False)
        if not points:
            return
        min_dist = float("inf")
        min_idx = None
        for i, (a, t) in enumerate(points):
            kx, ky = self.ax_torque.transData.transform((a, t))
            dist = math.hypot(event.x - kx, event.y - ky)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_dist < 14:
            self._drag_idx = min_idx

    def _on_motion(self, event) -> None:
        import numpy as np
        if self._drag_idx is None or event.inaxes is not self.ax_torque:
            return
        if event.xdata is None or event.ydata is None:
            return
        points = self._read_points(raise_on_error=False)
        if self._drag_idx >= len(points):
            return
        new_angle = round(event.xdata, 4)
        new_torque = round(event.ydata, 4)
        points[self._drag_idx] = (new_angle, new_torque)
        children = self.points_table.get_children()
        if self._drag_idx < len(children):
            self.points_table.item(
                children[self._drag_idx],
                values=(f"{new_angle:.6g}", f"{new_torque:.6g}"),
            )
        if self._knot_artists is not None:
            sorted_pts = sorted(points, key=lambda p: p[0])
            xs = [p[0] for p in sorted_pts]
            ys = [p[1] for p in sorted_pts]
            line, scat = self._knot_artists
            line.set_xdata(xs)
            line.set_ydata(ys)
            scat.set_offsets(np.column_stack([xs, ys]))
            self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._drag_idx is None or event.button != 1:
            return
        self._drag_idx = None
        self.recompute()

    def export_curve(self) -> None:
        if self.result is None:
            messagebox.showerror("ERC export", "No valid result to export.")
            return
        self.result.export_sldcrv(self.repo_root / "results" / "cam_profile.sldcrv")
        self.result.export_xyz(self.repo_root / "results" / "cam_profile.xyz")
        messagebox.showinfo(
            "ERC export",
            "Wrote results/cam_profile.sldcrv and results/cam_profile.xyz",
        )


def main() -> None:
    root = tk.Tk()
    app = ERCPlotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
