"""Matplotlib presentation for an OrigamiDegree4Simulator session.

The model owns solving, continuation and contact decisions. The viewer caches
immutable models for explicit drive selections and matches their current paths.
Keep the returned viewer alive while the window is open; it owns the widgets
and their callbacks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d


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

    def __init__(
        self, sim, branch=None, contact=None, wrap_margin=0.1, random_seed=None
    ):
        if not np.isfinite(wrap_margin) or not 0 < wrap_margin < np.pi:
            raise ValueError("wrap_margin must be finite and between 0 and pi.")
        if not getattr(sim, "branches", None):
            raise ValueError("No branches available; run run_simulation() first.")
        self.sim = sim
        self.wrap_margin = float(wrap_margin)
        self._rng = np.random.default_rng(random_seed)
        self.random_initializations = []
        self.session = sim.create_session(
            branch=branch if branch in sim.branches else None, contact=contact
        )
        self._build_branch_options()
        if branch is not None:
            choice = self._resolve_choice(branch)
            self.session = sim.create_session(
                self._branch_options[choice][0], contact=self.session.contact
            )
        if not self.session.contact:
            self.session = self._random_session(
                self._branch_options[self._selected_option()]
            )
        self._guard = False
        self.last_error = None
        self.last_movement_success = True
        self.update_count = 0
        self._graph_lines = []
        self._models = {sim.driving_crease: sim}
        self._resolution = getattr(sim, "resolution", 1000)
        self._press_xy = None
        self.driver_history = []
        self._crease_colors = ["royalblue", "seagreen", "firebrick", "mediumpurple"]

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
        self._colors = [self._crease_colors[j] for j in self._other]
        self.poly = Poly3DCollection([], edgecolors="#333", linewidths=1, alpha=0.82)
        self.poly.set_facecolor(["lightgray", "lightblue", "lightgreen", "lightcoral"])
        self.ax_3d.add_collection3d(self.poly)
        self.vertices = self.ax_3d.scatter([], [], [], c="black", s=25)
        self.crease_lines = [self.ax_3d.plot([], [], [], zorder=8)[0] for _ in range(4)]
        self.crease_labels = [
            self.ax_3d.text(0, 0, 0, f"e{i}", fontsize=9) for i in range(1, 5)
        ]
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
            active=self._branch_ids.index(self._selected_option()),
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
        self._mouse_callback_ids = (
            self.fig.canvas.mpl_connect("button_press_event", self._on_crease_press),
            self.fig.canvas.mpl_connect(
                "button_release_event", self._on_crease_release
            ),
        )
        # Keep a discoverable strong reference even when callers keep only fig.
        self.fig.origami_viewer = self
        self._sync_slider()
        self.redraw_background()
        self._render_state()

    @property
    def branch_options(self):
        """Displayed choice -> underlying actuation segments (read-only copy)."""
        return dict(self._branch_options)

    def _build_branch_options(self):
        # mode_id is assigned by the model's verified seam adjacency graph.
        # Fixed-drive families remain separate; never force every geometry to 2.
        groups = {}
        for bid, branch in self.sim.branches.items():
            key = (
                branch.mode_id
                if not self.session.contact and branch.family is None
                else bid
            )
            groups.setdefault(key, []).append(bid)
        self._branch_options = {}
        for key, members in groups.items():
            label = key
            prefix, _, suffix = key.rpartition("_c")
            if len(members) > 1 and suffix.isdigit():
                label = prefix
            if label in self._branch_options:
                label = key
            self._branch_options[label] = tuple(members)
        self._branch_ids = list(self._branch_options)

    def _selected_option(self):
        return next(
            label
            for label, members in self._branch_options.items()
            if self.session.branch_id in members
        )

    def _resolve_choice(self, choice):
        if choice in self._branch_options:
            return choice
        for label, members in self._branch_options.items():
            if choice in members:
                return label
        raise ValueError("Unknown branch " + str(choice))

    def _random_session(self, members):
        """Sample the interior of this path, then validate exact continuation.

        Randomness belongs only to viewer initialization / explicit path choice.
        The solver's seed states and deterministic create_session API are intact.
        """
        intervals = []
        for bid in members:
            branch = self.sim.branches[bid]
            bounds = (
                branch.family["valid_intervals"]
                if branch.family is not None
                else branch.reachable_intervals
            )
            for lo, hi in bounds:
                margin = max(1e-9, 0.01 * (hi - lo))
                if hi - lo > 2 * margin:
                    intervals.append((bid, lo + margin, hi - margin))
        if not intervals:
            raise ValueError("No regular interior available for random initialization.")
        weights = np.array([hi - lo for _, lo, hi in intervals])
        weights /= weights.sum()
        last_error = None
        for _ in range(24):
            bid, lo, hi = intervals[int(self._rng.choice(len(intervals), p=weights))]
            target = float(self._rng.uniform(lo, hi))
            new = self.sim.create_session(bid, contact=False)
            try:
                if new.is_family:
                    candidate = new.branch.family["evaluate"](target)
                    if candidate is None:
                        raise ValueError("No family state at sampled free angle.")
                    q = candidate.rhos.copy()
                    q[self.sim.drive_index] = new.branch.family["fixed_drive"]
                    drive = float(q[self.sim.drive_index])
                    new.free_angle = target
                else:
                    rows = new.branch.states
                    ids = np.argsort(abs(rows[:, self.sim.drive_index] - target))[:2]
                    if len(ids) < 2:
                        raise ValueError("Insufficient neighboring states.")
                    a, z = rows[ids]
                    dt = float(z[self.sim.drive_index] - a[self.sim.drive_index])
                    if abs(dt) < 1e-12:
                        raise ValueError("Degenerate drive coordinate.")
                    q, new.tangent, _ = self.sim._advance(
                        a, float(a[self.sim.drive_index]), target, (z - a) / dt
                    )
                    drive = target
                new.state = new._state(q, drive, ("RANDOM_START",))
                if (
                    not np.isfinite(q).all()
                    or new.state.residual > self.sim._core.residual_tol
                ):
                    raise ValueError("Sampled state failed closure validation.")
            except ValueError as error:
                last_error = error
                continue
            self.random_initializations.append(
                {"branch_id": bid, "drive": drive, "free_angle": new.free_angle}
            )
            return new
        raise ValueError(f"Cannot validate a random interior state: {last_error}")

    def _sync_radio(self):
        eventson = self.radio.eventson
        self.radio.eventson = False
        try:
            self.radio.set_active(self._branch_ids.index(self._selected_option()))
        finally:
            self.radio.eventson = eventson

    def _rebuild_branch_radio(self):
        """Refresh options after contact/drive changes without moving the state."""
        self.radio.disconnect(self._callback_ids[1])
        self.radio.disconnect_events()
        radio_ax = self.radio.ax
        radio_ax.clear()
        self._build_branch_options()
        self.radio = RadioButtons(
            radio_ax,
            self._branch_ids,
            active=self._branch_ids.index(self._selected_option()),
        )
        for label in self.radio.labels:
            label.set_fontsize(7)
        callbacks = list(self._callback_ids)
        callbacks[1] = self.radio.on_clicked(self.set_branch)
        self._callback_ids = tuple(callbacks)

    def _limits(self):
        extra = 0.0 if self.session.contact else self.wrap_margin
        return -np.pi - extra, np.pi + extra

    def _on_crease_press(self, event):
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        self._press_xy = (
            (event.x, event.y)
            if event.button == 1
            and event.inaxes is self.ax_3d
            and not self._guard
            and not str(getattr(toolbar, "mode", ""))
            else None
        )

    def _on_crease_release(self, event):
        start, self._press_xy = self._press_xy, None
        if (
            start is None
            or event.button != 1
            or event.inaxes is not self.ax_3d
            or np.linalg.norm(np.array([event.x, event.y]) - start) > 5
        ):
            return  # A drag rotates the camera; it must not change the driver.
        chosen = self._crease_at(event.x, event.y)
        if chosen is not None:
            self.set_driving_crease(chosen)

    def _crease_at(self, x, y):
        """Pick projected rays or labels, not triangle edges or depth guesses."""
        renderer = self.fig.canvas.get_renderer()
        labels = []
        for i, label in enumerate(self.crease_labels, start=1):
            # Text3D.get_window_extent uses its unprojected coordinates outside
            # draw(); project the label anchor explicitly for correct picking.
            position = label.get_position_3d()
            xp, yp, _ = proj3d.proj_transform(*position, self.ax_3d.get_proj())
            px, py = self.ax_3d.transData.transform((xp, yp))
            width, height, descent = renderer.get_text_width_height_descent(
                label.get_text(), label.get_fontproperties(), ismath=False
            )
            if (
                px - 3 <= x <= px + width + 3
                and py - descent - 3 <= y <= py + height - descent + 3
            ):
                labels.append(
                    (
                        np.linalg.norm(
                            np.array([x, y])
                            - [px + width / 2, py + height / 2 - descent]
                        ),
                        i,
                    )
                )
        if labels:
            return min(labels)[1]
        vertices = self.sim.get_3d_geometry(self.session.state.rhos)[1]
        xyz = proj3d.proj_transform(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], self.ax_3d.get_proj()
        )
        points = self.ax_3d.transData.transform(np.column_stack(xyz[:2]))
        matches = []
        for i in range(1, 5):
            ray = points[i] - points[0]
            length = np.dot(ray, ray)
            if length < 4:
                continue
            fraction = np.dot(np.array([x, y]) - points[0], ray) / length
            if not 0.15 <= fraction <= 1.05:
                continue  # The shared vertex cannot identify a crease.
            closest = points[0] + np.clip(fraction, 0, 1) * ray
            distance = np.linalg.norm(np.array([x, y]) - closest)
            if distance <= 8:
                matches.append((distance, i))
        matches.sort()
        if len(matches) > 1 and matches[1][0] - matches[0][0] < 1.5:
            self._error(
                "Creases overlap in this view. Click an e1-e4 label or rotate the view."
            )
            return None
        return matches[0][1] if matches else None

    def _motion_tangent(self):
        """Incoming path direction, used only to distinguish coincident modes."""
        old = self.session
        if old.is_family:
            index = old.branch.family["free_crease"] - 1
            q = old.branch.family["evaluate"](old.free_angle + 1e-5)
            if q is not None:
                return _representative(q.rhos - old.state.rhos) / 1e-5
        if old.tangent is not None:
            return old.tangent.copy()
        data = old.branch.states
        if len(data) < 2:
            return None
        ids = np.argsort(abs(data[:, self.sim.drive_index] - old.state.drive_wrapped))[
            :2
        ]
        a, b = data[ids]
        dt = b[self.sim.drive_index] - a[self.sim.drive_index]
        return (b - a) / dt if abs(dt) > 1e-12 else None

    @staticmethod
    def _direction_distance(a, b):
        if a is None or b is None or min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-10:
            return 1.0
        return 1 - abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))

    def _session_at_shape(self, model, motion):
        """Match all four folds and local path before committing a new axis."""
        from origami_simulator import PathError

        old = self.session
        q = old.state.rhos
        index = model.drive_index
        target = float(_representative(q[index]))
        if q[index] > 0 and abs(target + np.pi) < 2e-14:
            target = float(np.pi)
        movable = motion is not None and abs(motion[index]) > 1e-7
        matches = []
        for bid, b in model.branches.items():
            if b.family is not None:
                if abs(float(_representative(target - b.family["fixed_drive"]))) > 1e-7:
                    continue
                free = b.family["free_crease"] - 1
                c = b.family["evaluate"](float(q[free]))
                if c is None or np.linalg.norm(_representative(c.rhos - q)) > 2e-6:
                    continue
                neighbor = b.family["evaluate"](float(q[free]) + 1e-5)
                tangent = (
                    None
                    if neighbor is None
                    else _representative(neighbor.rhos - c.rhos) / 1e-5
                )
                score = self._direction_distance(motion, tangent) + (
                    2 if movable else 0
                )
                # Keep the sign of the current +/-pi representative when tied.
                score += 0.01 * abs(target - b.family["fixed_drive"])
                matches.append((score, bid, tangent))
                continue
            if not any(
                lo - 1e-7 <= target <= hi + 1e-7 for lo, hi in b.reachable_intervals
            ):
                continue
            data = b.states
            ids = np.argsort(abs(data[:, index] - target))[:2]
            if len(ids) < 2:
                continue
            a, z = data[ids]
            dt = z[index] - a[index]
            if abs(dt) < 1e-12:
                continue
            tangent = (z - a) / dt
            try:
                reached, tangent, _ = model._advance(
                    a, float(a[index]), target, tangent
                )
            except PathError:
                continue
            mismatch = np.linalg.norm(_representative(reached - q))
            if mismatch > 2e-6:
                continue
            score = self._direction_distance(motion, tangent) + mismatch
            matches.append((score, bid, tangent))
        matches.sort(key=lambda row: (row[0], row[1]))
        if not matches:
            raise ValueError(
                "Cannot continue the current shape with that crease. Move slightly away from the turning point and try again."
            )
        if len(matches) > 1 and abs(matches[1][0] - matches[0][0]) < 1e-4:
            raise ValueError(
                "The current shape lies on multiple paths. Move the slider slightly, then select a crease again."
            )
        _, bid, tangent = matches[0]
        new = model.create_session(bid, contact=old.contact)
        new.state = new._state(q.copy(), float(q[index]), ("DRIVING_CREASE_CHANGED",))
        new.tangent = motion / motion[index] if movable else tangent
        new.contact_reference_history = list(old.contact_reference_history)
        if new.is_family:
            new.free_angle = float(q[new.branch.family["free_crease"] - 1])
        if new.contact:
            new.contact_reference = new._copy_state()
            new._contact_reference_tangent = new.tangent
        return new

    def set_driving_crease(self, crease):
        """Explicit viewer action; immutable input models remain unchanged."""
        if self._guard:
            return False
        if isinstance(crease, bool) or crease not in (1, 2, 3, 4):
            raise ValueError("driving crease must be e1, e2, e3 or e4")
        crease = int(crease)
        if crease == self.sim.driving_crease:
            return True
        self._guard = True
        try:
            self.status.set_text(f"Updating graph for driving crease e{crease}...")
            self.fig.canvas.draw_idle()
            motion = self._motion_tangent()
            if crease not in self._models:
                self._models[crease] = type(self.sim)(
                    self.sim.original_alphas_deg,
                    driving_crease=crease,
                    contact=self.session.contact,
                    verbose=False,
                    seed_count=self.sim.seed_count,
                    max_seeds=self.sim.max_seeds,
                    wrap_margin=self.wrap_margin,
                    residual_tol=self.sim._core.residual_tol,
                ).run_simulation(self._resolution)
            model = self._models[crease]
            new_session = self._session_at_shape(model, motion)
            self.driver_history.append(
                (self.sim.driving_crease, self.session._copy_state())
            )
            self.sim, self.session = model, new_session
            self._other = [i for i in range(4) if i != model.drive_index]
            self._colors = [self._crease_colors[j] for j in self._other]
            self.ax_2d.set_xlabel(f"Driving angle e{crease} [rad]")
            self.slider.label.set_text(f"Fold e{crease}")
            self._rebuild_branch_radio()
            self.last_error = None
            self.last_movement_success = True
            self._sync_slider()
            self.redraw_background()
            self._render_state()
            return True
        except ValueError as error:
            self._error(error)
            return False
        finally:
            self._guard = False

    def _sync_slider(self):
        lo, hi = self._limits()
        self.slider.valmin, self.slider.valmax = lo, hi
        self.slider.ax.set_xlim(lo, hi)
        self.slider.set_active(
            not self.session.is_family and not self.session.drive_locked
        )
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
        if self._guard:
            return self.session.state
        self._guard = True
        try:
            choice = self._resolve_choice(branch_id)
            if choice == self._selected_option():
                return self.session.state
            members = self._branch_options[choice]
            new_session = (
                self.sim.create_session(branch=members[0], contact=True)
                if self.session.contact
                else self._random_session(members)
            )
            self.session = new_session
            self.last_error = None
            self.last_movement_success = True
            self._sync_radio()
            self._sync_slider()
            self.redraw_background()
            self._render_state()
        except ValueError as error:
            self._sync_radio()
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
            accepted = self.session.set_contact(bool(contact), reanchor=True)
            if accepted is False:
                self._error(
                    self.session.last_error
                    or "CONTACT_LIMIT: cannot enable contact bounds."
                )
            else:
                self.last_error = None
                self.last_movement_success = True
                self._rebuild_branch_radio()
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
        pieces = self.session.plot_data()
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
                        marker="o" if len(piece) == 1 else None,
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
        self.ax_2d.set_title(f"{self._selected_option()} | {policy}", fontsize=10)

    def _render_state(self):
        self._sync_family()
        state = self.session.state
        rhos = np.asarray(state.rhos, dtype=float)
        faces, vertices = self.sim.get_3d_geometry(rhos)
        vertices = np.asarray(vertices, dtype=float)
        self.poly.set_verts(faces)
        self.vertices._offsets3d = tuple(vertices[:, k] for k in range(3))
        for i, label in enumerate(self.crease_labels, start=1):
            point = vertices[i] * (1.07 + 0.03 * i)
            label.set_position((point[0], point[1]))
            label.set_3d_properties(point[2], zdir=None)
            active = i == self.sim.driving_crease
            color = "#d97706" if active else self._crease_colors[i - 1]
            line = self.crease_lines[i - 1]
            line.set_data(
                [vertices[0, 0], vertices[i, 0]], [vertices[0, 1], vertices[i, 1]]
            )
            line.set_3d_properties([vertices[0, 2], vertices[i, 2]])
            line.set_color(color)
            line.set_linewidth(4.5 if active else 1.8)
            label.set_color(color)
            label.set_fontweight("bold" if active else "normal")
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
            marker.set_color(self._crease_colors[col])
        self.ax_3d.set_title(
            f"{self._selected_option()}\n"
            f"e{self.sim.driving_crease} = {np.rad2deg(x):.3f} deg | "
            f"continuous = {np.rad2deg(state.drive_unwrapped):.3f} deg | "
            f"turns = {state.winding_count}",
            fontsize=9,
        )
        events = getattr(state, "events", ())
        if isinstance(events, str):
            message = events
        else:
            messages = {
                "RANDOM_START": "Random valid interior start selected.",
                "CONTACT_REFERENCE_RESET": "Contact enabled at current shape; angle reference reset.",
                "DRIVING_CREASE_CHANGED": f"Driving crease changed to e{self.sim.driving_crease}; shape preserved.",
            }
            message = (
                ", ".join(messages.get(e, str(e)) for e in events) if events else ""
            )
        if getattr(self.session, "is_family", False):
            message = (
                message + " | " if message else ""
            ) + "Driving fold is fixed; use the free-fold slider."
        elif self.session.drive_locked:
            message = (
                message + " | " if message else ""
            ) + "CONTACT_LIMIT: isolated contact state; driving fold is fixed."
        elif not self.session.contact:
            message = (
                message + " | " if message else ""
            ) + "Drag beyond +/-pi to continue across an allowed seam."
        message += " | Click a crease or its e1-e4 label to change the driving fold."
        self.status.set_text(message)
        self.status.set_color("#333")
        self.fig.canvas.draw_idle()

    def close(self):
        """Release callbacks and close this viewer."""
        self.slider.disconnect(self._callback_ids[0])
        self.radio.disconnect(self._callback_ids[1])
        self.check.disconnect(self._callback_ids[2])
        self.free_slider.disconnect(self._callback_ids[3])
        for cid in self._mouse_callback_ids:
            self.fig.canvas.mpl_disconnect(cid)
        self.radio.disconnect_events()
        plt.close(self.fig)


def create_viewer(
    sim, branch=None, contact=None, wrap_margin=0.1, show=True, random_seed=None
):
    """Create the viewer; ``show=False`` supports headless callback testing."""
    viewer = OrigamiViewer(
        sim,
        branch=branch,
        contact=contact,
        wrap_margin=wrap_margin,
        random_seed=random_seed,
    )
    if show:
        plt.show()
    return viewer
