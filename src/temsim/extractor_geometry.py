"""One axisymmetric extractor contour for metal, field and beam clearance.

Lengths are millimetres in the installed gun frame; +z points downstream.
The counterbore faces the tip. Its floor, rather than the outer front face or
body centre, defines the tip-to-extraction-aperture distance. Unspecified
counterbores remain the historical straight bore for saved assemblies.
"""
from dataclasses import dataclass
import math
from numbers import Real


COUNTERBORE_FIELDS = ("counterbore_depth_mm", "counterbore_diameter_mm")
FEI_EXTRACTOR_SOURCE = "https://patents.google.com/patent/US8736170B1/en"


@dataclass(frozen=True)
class ExtractorContour:
    start_mm: float
    end_mm: float
    bore_diameter_mm: float
    outer_diameter_mm: float
    counterbore_depth_mm: float = 0.
    counterbore_diameter_mm: float = 0.

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
            raise ValueError("Extractor dimensions must be finite real numbers")
        try:
            finite = all(math.isfinite(v) for v in values)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Extractor dimensions must be finite real numbers")
        if not self.start_mm < self.end_mm or not 0 < self.bore_diameter_mm < self.outer_diameter_mm:
            raise ValueError("Extractor requires positive thickness and 0 < bore < outer diameter")
        if not 0 <= self.counterbore_depth_mm < self.end_mm-self.start_mm:
            raise ValueError("Extractor counterbore depth must leave a positive aperture web thickness")
        if self.counterbore_diameter_mm < 0:
            raise ValueError("Extractor counterbore diameter must be nonnegative")
        if self.counterbore_depth_mm and not self.bore_diameter_mm < self.counterbore_diameter_mm < self.outer_diameter_mm:
            raise ValueError("Extractor counterbore requires bore < counterbore < outer diameter")

    @classmethod
    def from_component(cls, component):
        center, half = component.mechanical_center_from_tip_mm, component.mechanical_length_mm/2
        return cls(center-half, center+half, component.mechanical_clear_bore_diameter_mm,
                   component.mechanical_outer_diameter_mm,
                   *(getattr(component, name, 0.) for name in COUNTERBORE_FIELDS))

    @classmethod
    def from_part(cls, part, *, axial_shift_mm=0.):
        return cls(float(part["local_start_z_mm"])+axial_shift_mm,
                   float(part["local_end_z_mm"])+axial_shift_mm,
                   float(part["bore_diameter_mm"]), float(part["outer_diameter_mm"]),
                   *(part.get(name, 0.) for name in COUNTERBORE_FIELDS))

    @property
    def aperture_plane_mm(self):
        return self.start_mm+self.counterbore_depth_mm

    @property
    def aperture_web_thickness_mm(self):
        return self.end_mm-self.aperture_plane_mm

    def straight_ray_acceptance(self, origin_z_mm=0.):
        """On-axis point-source clearance only, with no electric or magnetic field.

        The limiting half-angle is a supremum: equality touches metal. Off-axis
        surface emission and accelerated trajectories require actual transport.
        """
        if not math.isfinite(origin_z_mm) or origin_z_mm >= self.start_mm:
            raise ValueError("Acceptance requires an on-axis origin upstream of the extractor")
        limits = [(math.atan2(inner, stop-origin_z_mm), stop, inner)
                  for _start, stop, inner, _outer in self.annuli_mm]
        angle, plane, radius = min(limits)
        return {"model": "on_axis_straight_rays_without_fields", "origin_z_mm": origin_z_mm,
                "half_angle_deg": math.degrees(angle), "full_cone_angle_deg": 2*math.degrees(angle),
                "limiting_z_mm": plane, "limiting_radius_mm": radius,
                "entrance_half_angle_deg": math.degrees(math.atan2(self.bore_diameter_mm/2, self.aperture_plane_mm-origin_z_mm)),
                "boundary": "grazing_metal_is_intercepted", "field_acceptance": "requires_launch_position_energy_and_field_transport"}

    @property
    def annuli_mm(self):
        """Contiguous (start, end, inner radius, outer radius) material pieces."""
        outer = self.outer_diameter_mm/2
        throat = (self.aperture_plane_mm, self.end_mm, self.bore_diameter_mm/2, outer)
        if not self.counterbore_depth_mm:
            return (throat,)
        return ((self.start_mm, self.aperture_plane_mm, self.counterbore_diameter_mm/2, outer), throat)

    @property
    def section_zr_mm(self):
        """Closed-boundary polygon, without duplicated vertices at a seam."""
        inner, outer = self.bore_diameter_mm/2, self.outer_diameter_mm/2
        if not self.counterbore_depth_mm:
            return ((self.start_mm, inner), (self.start_mm, outer),
                    (self.end_mm, outer), (self.end_mm, inner))
        recess = self.counterbore_diameter_mm/2
        return ((self.start_mm, recess), (self.start_mm, outer),
                (self.end_mm, outer), (self.end_mm, inner),
                (self.aperture_plane_mm, inner), (self.aperture_plane_mm, recess))


def resolved_extractor_contour(part):
    """Read an installed extractor, retaining the assembly's axial translation."""
    if part.data.get("geometry_template_key", part.key) != "feg_extractor":
        return None
    shift = float(part.start_z_mm)-float(part.data["local_start_z_mm"])
    return ExtractorContour.from_part(part.data, axial_shift_mm=shift)
