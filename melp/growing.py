import numpy as np

from manimlib.animation.growing import GrowArrow as _ManimGrowArrow
from manimlib.mobject.geometry import Arrow, Line, StrokeArrow
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.space_ops import get_norm, normalize, rotation_between_vectors

__all__ = ["GrowArrow"]


class GrowArrow(_ManimGrowArrow):
    """Grow an arrowhead along its shaft instead of along the end-to-end chord."""

    def begin(self):
        self._final_arrow = self.mobject.copy()
        self._mode = self._get_growth_mode(self._final_arrow)

        if self._mode == "detached_tip":
            self._prepare_detached_tip_growth()
        elif self._mode == "curved_integrated_tip":
            self._prepare_integrated_tip_growth()

        super().begin()

    def _get_growth_mode(self, arrow):
        has_tip = getattr(arrow, "has_tip", lambda: False)()
        if isinstance(arrow, VMobject) and arrow.has_points() and has_tip:
            return "detached_tip"

        path_arc = getattr(arrow, "path_arc", 0.0)
        if isinstance(arrow, (Arrow, StrokeArrow)) and not np.isclose(path_arc, 0):
            return "curved_integrated_tip"

        return "scale"

    def _prepare_detached_tip_growth(self):
        arrow = self._final_arrow
        self._guide = arrow.copy()
        guide_start = arrow.get_start().copy()
        guide_end = arrow.get_end().copy()

        for attr in ("tip", "start_tip"):
            tip = getattr(self._guide, attr, None)
            if tip in self._guide.submobjects:
                self._guide.remove(tip)

        self._guide.put_start_and_end_on(guide_start, guide_end)
        self._prepare_path_lookup(self._guide)

    def _prepare_integrated_tip_growth(self):
        arrow = self._final_arrow
        self._path_arc = arrow.path_arc
        self._guide_start = arrow.get_start().copy()
        self._guide = Line(
            self._guide_start,
            arrow.get_end(),
            path_arc=self._path_arc,
        )
        self._prepare_path_lookup(self._guide)

    def _prepare_path_lookup(self, path):
        sample_count = max(65, 16 * path.get_num_curves() + 1)
        self._path_parameters = np.linspace(0.0, 1.0, sample_count)
        points = np.array(
            [path.quick_point_from_proportion(t) for t in self._path_parameters]
        )
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self._path_distances = np.append(0.0, np.cumsum(lengths))

        total_length = self._path_distances[-1]
        if total_length > 0:
            self._path_distances /= total_length

    def _path_parameter(self, alpha):
        if self._path_distances[-1] == 0:
            return alpha
        return np.interp(alpha, self._path_distances, self._path_parameters)

    def _path_tangent(self, path_parameter):
        step = 1e-4 / max(1, self._guide.get_num_curves())
        lower = max(0.0, path_parameter - step)
        upper = min(1.0, path_parameter + step)
        tangent = self._guide.quick_point_from_proportion(
            upper
        ) - self._guide.quick_point_from_proportion(lower)

        if get_norm(tangent) == 0:
            tangent = self._guide.get_end() - self._guide.get_start()

        return normalize(tangent)

    def _place_tip(self, tip, final_tip, point, tangent, alpha):
        tip.become(final_tip)
        tip_point = tip.get_tip_point()
        tip_vector = tip.get_vector()
        tip.scale(alpha, about_point=tip_point)

        if get_norm(tip_vector) > 0 and get_norm(tangent) > 0:
            tip.apply_matrix(
                rotation_between_vectors(tip_vector, tangent),
                about_point=tip_point,
            )

        tip.shift(point - tip_point)

    def interpolate_mobject(self, alpha):
        if self._mode == "scale":
            super().interpolate_mobject(alpha)
            return

        progress = self.get_sub_alpha(self.time_spanned_alpha(alpha), 0, 1)
        if progress >= 1:
            self.mobject.become(self._final_arrow)
            return

        path_parameter = self._path_parameter(progress)
        if self._mode == "detached_tip":
            self._interpolate_detached_tip(path_parameter, progress)
        else:
            self._interpolate_integrated_tip(path_parameter, progress)

    def _interpolate_detached_tip(self, path_parameter, progress):
        self.mobject.become(self._final_arrow)
        self.mobject.pointwise_become_partial(
            self._final_arrow,
            0.0,
            path_parameter,
        )

        point = self._guide.quick_point_from_proportion(path_parameter)
        tangent = self._path_tangent(path_parameter)
        self._place_tip(
            self.mobject.tip,
            self._final_arrow.tip,
            point,
            tangent,
            progress,
        )

        if getattr(self._final_arrow, "has_start_tip", lambda: False)():
            start_tip = self.mobject.start_tip
            final_start_tip = self._final_arrow.start_tip
            start_tip.become(final_start_tip)
            start_tip.scale(progress, about_point=final_start_tip.get_tip_point())

    def _interpolate_integrated_tip(self, path_parameter, progress):
        if progress <= 0:
            self.mobject.become(self.starting_mobject)
            return

        point = self._guide.quick_point_from_proportion(path_parameter)
        path_arc = self._path_arc * path_parameter
        grown_arrow = self._final_arrow.copy()

        if isinstance(grown_arrow, StrokeArrow):
            stroke_width = float(self._final_arrow.original_stroke_width) * progress
            VMobject.set_stroke(grown_arrow, width=stroke_width)
            grown_arrow.original_stroke_width = stroke_width
        else:
            grown_arrow.thickness = self._final_arrow.thickness * progress

        grown_arrow.set_points_by_ends(
            self._guide_start,
            point,
            path_arc=path_arc,
        )
        self.mobject.become(grown_arrow)
