from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from py_linq import Enumerable

from pylinac.core.mtf import MTF
from pylinac.core.nps import (
    average_power,
    max_frequency,
    noise_power_spectrum_1d,
    noise_power_spectrum_2d,
)
from pylinac.core.profile import CollapsedCircleProfile
from pylinac.core.roi import LowContrastDiskROI, RectangleROI
from pylinac.core.warnings import capture_warnings
from pylinac.ct import (
    CatPhanBase, 
    CatPhanModule,
    GeometricLine,
    Point,
    Slice,
    ThicknessROI,
    add_title,
    get_regions,
    go,
)
from pylinac.phantoms import CP504

# The ramp angle ratio is from the Catphan manual ("Scan slice geometry" section)
# and represents the fact that the wire is at an oblique angle (23°), making it appear
# longer than it is if it were normal or perpendicular to the z (imaging) axis. This ratio
# fixes the length to represent it as if it were perpendicular to the imaging axis.
RAMP_ANGLE_RATIO = 0.42

AIR = -1000
PMP = -196
LDPE = -104
POLY = -47
ACRYLIC = 115
DELRIN = 365
TEFLON = 1000
BONE_20 = 237
BONE_50 = 725
WATER = 0


# Catphan Modules 
# CTP730 - (aka CTP515) Low Contrast Module
# CTP732 - (aka CTP404) Geometry and HU Module
# CTP729 - (aka CTP486) Uniformity Module
# CTP528 - High Resolution Module - module unnamed in manual

class CTP732(CP504.CTP404):

    """
    CTP732 is the manual definition for the updated version of CPT404 found on the CatPhan 504.
    """
    roi_dist_mm = 58.7
    roi_radius_mm = 5
    roi_settings = {
        "Air": {
            "value": AIR,
            "angle": -90,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "PMP": {
            "value": PMP,
            "angle": -120,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "50% Bone": {
            "value": BONE_50,
            "angle": -150,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "LDPE": {
            "value": LDPE,
            "angle": 180,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "Poly": {
            "value": POLY,
            "angle": 120,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "Acrylic": {
            "value": ACRYLIC,
            "angle": 60,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "20% Bone": {
            "value": BONE_20,
            "angle": 30,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "Delrin": {
            "value": DELRIN,
            "angle": 0,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "Teflon": {
            "value": TEFLON,
            "angle": -60,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
    }
    background_roi_settings = {
        "1": {"angle": -30, "distance": roi_dist_mm, "radius": roi_radius_mm},
        "2": {"angle": -210, "distance": roi_dist_mm, "radius": roi_radius_mm},
    }    


class CTP729(CP504.CTP486):
    """ Uniformity module analysis unchanged"""
    pass

class CTP730(CP504.CTP515):
    """ Low contrast module unchanged """
    pass

class CTP486(CTP729):
    pass

class CTP528(CP504.CTP528):
    """Alias for namespace continuity.
    CP604 High resolution module unchanged
    """

class CTP404(CTP732):
    """
    Alias, CTP732 is the manual definition for the updated version of CPT404 found on the CatPhan 504.
    """
    pass

class CTP515(CTP730):
    """
    Alias, Low contrast module unchanged
    """
    pass

@capture_warnings
class CatPhan604(CatPhanBase):
    """A class for loading and analyzing CT DICOM files of a CatPhan 604. Can be from a CBCT or CT scanner
    Analyzes: Uniformity (CTP486), High-Contrast Spatial Resolution (CTP528),
    Image Scaling & HU Linearity (CTP404), and Low contrast (CTP515).
    """

    _demo_url = "CatPhan604.zip"
    _model = "604"
    catphan_radius_mm = 101
    modules = {
        CTP732: {"offset": 0},
        CTP729: {"offset": -80},
        CTP528: {"offset": 40},
        CTP730: {"offset": -40},
    }

    @staticmethod
    def run_demo(show: bool = True):
        """Run the CBCT demo using high-quality head protocol images."""
        cbct = CatPhan604.from_demo_images()
        cbct.analyze()
        print(cbct.results())
        cbct.plot_analyzed_image(show)

    def refine_origin_slice(self, initial_slice_num: int) -> int:
        """The HU plugs are longer than the 'wire section'. This applies a refinement to find the
        slice that has the least angle between the centers of the left and right wires.

        Under normal conditions, we would simply apply an offset to the
        initial slice. The rods extend 4-5mm past the wire section.
        Unfortunately, we also sometimes need to account for users with the
        RM R1-4 jig which will cause localization issues due to the base of the
        jig touching the phantom.
        This solution is robust to the jig being present, but can suffer
        from images where the angle of the wire, due to noise,
        appears small but doesn't actually represent the wire ramp.

        Starting with the initial slice, go +/- 5 slices to find the slice with the least angle
        between the left and right wires.

        This suffers from a weakness that the roll is not yet determined.
        This will thus return the slice that has the least ABSOLUTE
        roll. If the phantom has an inherent roll, this will not be detected and may be off by a slice or so.
        Given the angle of the wire, the error would be small and likely only 1-2 slices max.
        """
        angles = []
        # make a CTP module for the purpose of easily extracting the thickness ROIs
        ctp404, offset = self._get_module(CTP732, raise_empty=True)
        # we don't want to set up our other ROIs (they sometimes fail) so we temporarily override the method
        original_setup = ctp404._setup_rois
        ctp404._setup_rois = lambda x: x
        ctp = ctp404(
            self,
            offset=offset,
            clear_borders=self.clear_borders,
            hu_tolerance=0,
            scaling_tolerance=0,
            thickness_tolerance=0,
        )
        # we have to reset the method after we're done for future calls
        ctp404._setup_rois = original_setup
        for slice_num in range(initial_slice_num - 5, initial_slice_num + 5):
            slice = Slice(self, slice_num, clear_borders=self.clear_borders)
            # make a slice and add the wire ROIs to it.
            troi = {}
            for name, setting in ctp.thickness_roi_settings.items():
                troi[name] = ThicknessROI.from_phantom_center(
                    slice.image,
                    setting["width_pixels"],
                    setting["height_pixels"],
                    setting["angle_corrected"],
                    setting["distance_pixels"],
                    slice.phan_center,
                )
            # now find the angle between the left and right and top and bottom wires via the long profile
            left_wire = troi["Left"].long_profile.center_idx
            right_wire = troi["Right"].long_profile.center_idx
            h_angle = abs(left_wire - right_wire)
            top_wire = troi["Top"].long_profile.center_idx
            bottom_wire = troi["Bottom"].long_profile.center_idx
            v_angle = abs(top_wire - bottom_wire)
            angle = (h_angle + v_angle) / 2

            angles.append(
                {
                    "slice": slice_num,
                    "angle": angle,
                    "left width": troi["Left"].long_profile.field_width_px,
                    "right width": troi["Right"].long_profile.field_width_px,
                    # the values AT the FWXM
                    "left center": troi["Left"].long_profile.y_at_x(left_wire),
                    "right center": troi["Right"].long_profile.y_at_x(right_wire),
                    "left profile": troi["Left"].long_profile.values,
                    "right profile": troi["Right"].long_profile.values,
                }
            )

        # some slices might not include the wire
        # we need to drop those; we do so by dropping pairs that have a field width well below the median
        # or by fields who don't appear to have the wire in them (max value is near median)
        median_width_l = np.median([angle["left width"] for angle in angles])
        median_width_r = np.median([angle["right width"] for angle in angles])
        median_width = (median_width_l + median_width_r) / 2
        # get median and max pixel values of all the profiles
        median_left_pixel_val = np.median(
            np.concatenate([a["left profile"] for a in angles])
        )
        median_right_pixel_val = np.median(
            np.concatenate([a["right profile"] for a in angles])
        )
        median_pixel_val = (median_left_pixel_val + median_right_pixel_val) / 2
        max_left_pixel_val = np.max(np.concatenate([a["left profile"] for a in angles]))
        max_right_pixel_val = np.max(
            np.concatenate([a["right profile"] for a in angles])
        )
        max_pixel_val = (max_left_pixel_val + max_right_pixel_val) / 2

        for angle_set in angles.copy():
            # field width is well below the median; probably not in the slice; drop it
            if (
                angle_set["left width"] < median_width * 0.7
                or angle_set["right width"] < median_width * 0.7
            ):
                angles.remove(angle_set)
                continue
            # if the max pixel value of the angle set is closer to the overall median than the max
            # it means the wire isn't in the slice; drop it
            fwxm_pixel = np.mean((angle_set["left center"], angle_set["right center"]))
            delta_median = abs(median_pixel_val - fwxm_pixel)
            delta_max = abs(max_pixel_val - fwxm_pixel)
            if delta_median < delta_max:
                angles.remove(angle_set)

        # now find the slice with the least angle, accounting for the phantom roll
        m_slice_num = np.argsort([a["angle"] - self.catphan_roll for a in angles])
        refined_slice = angles[m_slice_num[0]]["slice"]
        return refined_slice
