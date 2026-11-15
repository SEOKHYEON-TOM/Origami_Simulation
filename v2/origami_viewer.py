"""Matplotlib presentation for an OrigamiDegree4Simulator session.

The model owns continuation, branch identity and contact decisions.  This module
only forwards user actions and renders accepted states.  Keep the returned
viewer alive while the window is open; it owns the widgets and their callbacks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _representative(values):
    """Representative angles for a graph, including a consistent pi seam."""
    return (np.asarray(values, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _plot_pieces(states):
    """Split a graph at nonfinite rows and representative-angle seams."""
    states = np.asarray(states, dtype=float)
    if states.size == 0:
        return []
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("Plot data must have shape (n, 4).")
    data = _representative(states)
    good = np.isfinite(data).all(axis=1)
    cuts = (
        np.flatnonzero(
            (~good[:-1])
            | (~good[1:])
            | np.any(np.abs(np.diff(data, axis=0)) > np.pi, axis=1)
        )
        + 1
    )
    return [
        part[np.isfinite(part).all(axis=1)]
        for part in np.split(data, cuts)
        if np.isfinite(part).any()
    ]


class OrigamiViewer:
    """Inspectable 3D viewer with a guarded, periodic driving-angle slider.

    ``slider.set_val(...)`` follows the same callback as a mouse drag.  In
    no-contact mode the slider has a small extension at both ends so Matplotlib
    does not clamp an attempted seam crossing before the model receives it.
    Rejected movements restore the old accepted position and display the model
    reason.  No alternate branch is selected implicitly.
    """

    def __init__(self, sim, branch=None, contact=None, wrap_margin=0.1):
        if not np.isfinite(wrap_margin) or not 0 < wrap_margin < np.pi:
            raise ValueError("wrap_margin must be finite and between 0 and pi.")
        if not getattr(sim, "branches", None):
            raise ValueError("No branches available; run run_simulation() first.")
        self.sim = sim
        self.wrap_margin = float(wrap_margin)
        self.session = sim.create_session(branch=branch, contact=contact)
        self._guard = False
        self.last_error = None
        self.last_movement_success = True
        self.update_count = 0
        self._branch_ids = list(sim.branches)
        self._graph_lines = []

        self.fig = plt.figure(figsize=(15, 8))
        self.ax_3d = self.fig.add_axes([0.03, 0.28, 0.44, 0.65], projection="3d")
        self.ax_2d = self.fig.add_axes([0.55, 0.28, 0.42, 0.65])
        self.ax_3d.set(xlabel="X", ylabel="Y", zlabel="Z")
        self.ax_3d.set_box_aspect([1, 1, 1])
        self.ax_2d.set(
            xlim=(-np.pi, np.pi),
            ylim=(-np.pi, np.pi),
            xlabel=f"Driving angle e{sim.driving_crease} [rad]",
            ylabel="Folding angles [rad]",
        )
        self.ax_2d.axhline(0, color="black", lw=0.6)
        self.ax_2d.axvline(0, color="black", lw=0.6)
        self.ax_2d.grid(alpha=0.3)
        self._other = [i for i in range(4) if i != sim.driving_crease - 1]
        self._colors = ["royalblue", "seagreen", "firebrick"]
        self.poly = Poly3DCollection([], edgecolors="#333", linewidths=1, alpha=0.82)
        self.poly.set_facecolor(["lightgray", "lightblue", "lightgreen", "lightcoral"])
        self.ax_3d.add_collection3d(self.poly)
        self.vertices = self.ax_3d.scatter([], [], [], c="black", s=25)
        self.tracker_line = self.ax_2d.axvline(0, c="gray", ls="--", alpha=0.7)
        self.tracker_points = [
            self.ax_2d.scatter([], [], c=c, s=48, zorder=5) for c in self._colors
        ]
        self.status = self.fig.text(0.22, 0.025, "", fontsize=9, va="bottom", wrap=True)

        # This extended numeric interval is essential: a native [-pi,pi]
        # Slider clips mouse events before its callback receives an overshoot.
        lo, hi = self._limits()
        slider_ax = self.fig.add_axes([0.24, 0.13, 0.51, 0.035])
        self.slider = Slider(
            slider_ax,
            f"Fold e{sim.driving_crease}",
            lo,
            hi,
            valinit=self.session.state.drive_wrapped,
            valfmt="%1.5f rad",
        )
        family_ax = self.fig.add_axes([0.24, 0.195, 0.51, 0.035])
        self.free_slider = Slider(
            family_ax,
            "Free fold",
            -np.pi,
            np.pi,
            valinit=float(getattr(self.session, "free_angle", 0.0)),
            valfmt="%1.5f rad",
        )
        radio_ax = self.fig.add_axes([0.025, 0.02, 0.17, 0.20])
        self.radio = RadioButtons(
            radio_ax,
            self._branch_ids,
            active=self._branch_ids.index(self.session.branch_id),
        )
        for label in self.radio.labels:
            label.set_fontsize(7)
        contact_ax = self.fig.add_axes([0.84, 0.09, 0.14, 0.09])
        self.check = CheckButtons(
            contact_ax, ["Contact bounds"], [self.session.contact]
        )
        self._callback_ids = (
            self.slider.on_changed(self.update),
            self.radio.on_clicked(self.set_branch),
            self.check.on_clicked(self._on_contact),
            self.free_slider.on_changed(self.move_family),
        )
        # Keep a discoverable strong reference even when callers keep only fig.
        self.fig.origami_viewer = self
        self._sync_slider()
        self.redraw_background()
        self._render_state()

    def _limits(self):
        extra = 0.0 if self.session.contact else self.wrap_margin
        return -np.pi - extra, np.pi + extra

    def _sync_slider(self):
        lo, hi = self._limits()
        self.slider.valmin, self.slider.valmax = lo, hi
        self.slider.ax.set_xlim(lo, hi)
        self.slider.set_active(not getattr(self.session, "is_family", False))
        eventson = self.slider.eventson
        self.slider.eventson = False
        try:
            self.slider.set_val(self.session.state.drive_wrapped)
        finally:
            self.slider.eventson = eventson

    def _sync_family(self):
        is_family = bool(getattr(self.session, "is_family", False))
        self.free_slider.ax.set_visible(is_family)
        self.free_slider.set_active(is_family)
        if is_family:
            free_crease = getattr(
                self.sim.branches[self.session.branch_id], "family", {}
            ).get("free_crease")
            self.free_slider.label.set_text(
                "Free fold" if free_crease is None else f"Free e{free_crease}"
            )
            eventson = self.free_slider.eventson
            self.free_slider.eventson = False
            try:
                self.free_slider.set_val(float(self.session.free_angle))
            finally:
                self.free_slider.eventson = eventson

    def _sync_contact(self):
        checked = bool(self.check.get_status()[0])
        if checked != self.session.contact:
            eventson = self.check.eventson
            self.check.eventson = False
            try:
                self.check.set_active(0)
            finally:
                self.check.eventson = eventson

    def _error(self, error):
        self.last_error = str(error)
        self.last_movement_success = False
        self.status.set_text(self.last_error)
        self.status.set_color("firebrick")
        self._sync_slider()
        self.fig.canvas.draw_idle()

    def update(self, value=None):
        """Process one slider action; programmatic relocation emits no action."""
        if self._guard:
            return self.session.state
        self._guard = True
        try:
            self.update_count += 1
            self.session.move_slider(float(self.slider.val if value is None else value))
            self.last_error = None
            self.last_movement_success = True
            self._sync_slider()
            self._render_state()
        except ValueError as error:
            self._error(error)
        finally:
            self._guard = False
        return self.session.state

    def set_branch(self, branch_id):
        """Explicit user branch choice; failed selection retains old session."""
        if self._guard or branch_id == self.session.branch_id:
            return self.session.state
        self._guard = True
        try:
            new_session = self.sim.create_session(
                branch=branch_id, contact=self.session.contact
            )
            self.session = new_session
            self.last_error = None
            self.last_movement_success = True
            eventson = self.radio.eventson
            self.radio.eventson = False
            try:
                self.radio.set_active(self._branch_ids.index(self.session.branch_id))
            finally:
                self.radio.eventson = eventson
            self._sync_slider()
            self.redraw_background()
            self._render_state()
        except ValueError as error:
            previous = self._branch_ids.index(self.session.branch_id)
            eventson = self.radio.eventson
            self.radio.eventson = False
            try:
                self.radio.set_active(previous)
            finally:
                self.radio.eventson = eventson
            self._error(error)
        finally:
            self._guard = False
        return self.session.state

    def move_family(self, value):
        """Move a fixed-drive continuous family through its independent fold."""
        if self._guard or not getattr(self.session, "is_family", False):
            return self.session.state
        self._guard = True
        try:
            self.update_count += 1
            self.session.move_family(float(value))
            self.last_error = None
            self.last_movement_success = True
            self._sync_slider()
            self._render_state()
        except ValueError as error:
            self._sync_family()
            self._error(error)
        finally:
            self._guard = False
        return self.session.state

    def set_contact(self, contact):
        """Try contact transition; rejected transitions leave widgets coherent."""
        if self._guard:
            return False
        self._guard = True
        try:
            accepted = self.session.set_contact(bool(contact))
            if accepted is False:
                self._error("CONTACT_LIMIT: current path cannot enable contact bounds.")
            else:
                self.last_error = None
                self.last_movement_success = True
                self.redraw_background()
                self._render_state()
            self._sync_contact()
            self._sync_slider()
            return accepted is not False
        except ValueError as error:
            self._sync_contact()
            self._error(error)
            return False
        finally:
            self._guard = False

    def _on_contact(self, _label):
        if not self._guard:
            self.set_contact(bool(self.check.get_status()[0]))

    def redraw_background(self):
        """Render separate admissible pieces; never join gaps or pi seams."""
        for line in self._graph_lines:
            line.remove()
        self._graph_lines.clear()
        pieces = self.sim.plot_data(self.session.branch_id, self.session.contact)
        for part_idx, part in enumerate(pieces):
            for piece in _plot_pieces(part):
                if not len(piece):
                    continue
                for j, col in enumerate(self._other):
                    label = f"e{col+1}" if part_idx == 0 else "_nolegend_"
                    (line,) = self.ax_2d.plot(
                        piece[:, self.sim.driving_crease - 1],
                        piece[:, col],
                        c=self._colors[j],
                        lw=1.8,
                        label=label,
                    )
                    self._graph_lines.append(line)
        handles, labels = self.ax_2d.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            self.ax_2d.legend(unique.values(), unique.keys(), loc="upper right")
        elif self.ax_2d.legend_ is not None:
            self.ax_2d.legend_.remove()
        policy = "Contact bounds" if self.session.contact else "Periodic angles"
        self.ax_2d.set_title(f"{self.session.branch_id} | {policy}", fontsize=10)

    def _render_state(self):
        self._sync_family()
        state = self.session.state
        rhos = np.asarray(state.rhos, dtype=float)
        faces, vertices = self.sim.get_3d_geometry(rhos)
        vertices = np.asarray(vertices, dtype=float)
        self.poly.set_verts(faces)
        self.vertices._offsets3d = tuple(vertices[:, k] for k in range(3))
        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        extent = max(float(np.max(np.ptp(vertices, axis=0))) * 0.6, 0.5)
        self.ax_3d.set_xlim(center[0] - extent, center[0] + extent)
        self.ax_3d.set_ylim(center[1] - extent, center[1] + extent)
        self.ax_3d.set_zlim(center[2] - extent, center[2] + extent)
        x = float(state.drive_wrapped)
        y = _representative(rhos)
        self.tracker_line.set_xdata([x, x])
        for marker, col in zip(self.tracker_points, self._other):
            marker.set_offsets([[x, y[col]]])
        self.ax_3d.set_title(
            f"{self.session.branch_id}\n"
            f"e{self.sim.driving_crease} = {np.rad2deg(x):.3f} deg | "
            f"continuous = {np.rad2deg(state.drive_unwrapped):.3f} deg | "
            f"turns = {state.winding_count}",
            fontsize=9,
        )
        events = getattr(state, "events", ())
        if isinstance(events, str):
            message = events
        else:
            message = ", ".join(map(str, events)) if events else ""
        if getattr(self.session, "is_family", False):
            message = (
                message + " | " if message else ""
            ) + "Driving fold is fixed; use the free-fold slider."
        elif not self.session.contact:
            message = (
                message + " | " if message else ""
            ) + "Drag beyond +/-pi to continue across an allowed seam."
        self.status.set_text(message)
        self.status.set_color("#333")
        self.fig.canvas.draw_idle()

    def close(self):
        """Release callbacks and close this viewer."""
        self.slider.disconnect(self._callback_ids[0])
        self.radio.disconnect(self._callback_ids[1])
        self.check.disconnect(self._callback_ids[2])
        self.free_slider.disconnect(self._callback_ids[3])
        plt.close(self.fig)


def create_viewer(sim, branch=None, contact=None, wrap_margin=0.1, show=True):
    """Create the viewer; ``show=False`` supports headless callback testing."""
    viewer = OrigamiViewer(sim, branch=branch, contact=contact, wrap_margin=wrap_margin)
    if show:
        plt.show()
    return viewer
