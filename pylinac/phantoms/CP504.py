
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
# CTP515 - Low Contrast Module
# CTP404 - Geometry and HU Module
# CTP486 - Uniformity Module
# CTP528 - High Resolution Module




class CTP404(CatPhanModule):
    """Class for analysis of the HU linearity, geometry, and slice thickness regions of the CTP404."""

    attr_name = "ctp404"
    common_name = "HU Linearity"
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
        "2": {"angle": -150, "distance": roi_dist_mm, "radius": roi_radius_mm},
        "3": {"angle": -210, "distance": roi_dist_mm, "radius": roi_radius_mm},
        "4": {"angle": 30, "distance": roi_dist_mm, "radius": roi_radius_mm},
    }
    # thickness
    thickness_roi_height = 40
    thickness_roi_width = 10
    thickness_roi_distance_mm = 38
    thickness_roi_settings = {
        "Left": {
            "angle": 180,
            "width": thickness_roi_width,
            "height": thickness_roi_height,
            "distance": thickness_roi_distance_mm,
        },
        "Bottom": {
            "angle": 90,
            "width": thickness_roi_height,
            "height": thickness_roi_width,
            "distance": thickness_roi_distance_mm,
        },
        "Right": {
            "angle": 0,
            "width": thickness_roi_width,
            "height": thickness_roi_height,
            "distance": thickness_roi_distance_mm,
        },
        "Top": {
            "angle": -90,
            "width": thickness_roi_height,
            "height": thickness_roi_width,
            "distance": thickness_roi_distance_mm,
        },
    }
    # geometry
    geometry_roi_size_mm = 35
    geometry_roi_settings = {
        "Top-Horizontal": (0, 1),
        "Bottom-Horizontal": (2, 3),
        "Left-Vertical": (0, 2),
        "Right-Vertical": (1, 3),
    }
    pad: str | int
    thickness_image: Slice

    def __init__(
        self,
        catphan,
        offset: int,
        hu_tolerance: float,
        thickness_tolerance: float,
        scaling_tolerance: float,
        clear_borders: bool = True,
        thickness_slice_straddle: str | int = "auto",
        expected_hu_values: dict[str, float | int] | None = None,
    ):
        """
        Parameters
        ----------
        catphan : `~pylinac.cbct.CatPhanBase` instance.
        offset : int
        hu_tolerance : float
        thickness_tolerance : float
        scaling_tolerance : float
        clear_borders : bool
        """
        self.mm_per_pixel = catphan.mm_per_pixel
        self.hu_tolerance = hu_tolerance
        self.thickness_tolerance = thickness_tolerance
        self.scaling_tolerance = scaling_tolerance
        self.thickness_rois = {}
        self.lines = {}
        self.thickness_slice_straddle = thickness_slice_straddle
        self.expected_hu_values = expected_hu_values
        super().__init__(
            catphan, tolerance=hu_tolerance, offset=offset, clear_borders=clear_borders
        )

    def preprocess(self, catphan) -> None:
        # for the thickness analysis image, combine thin slices or just use one slice if slices are thick
        if (
            isinstance(self.thickness_slice_straddle, str)
            and self.thickness_slice_straddle.lower() == "auto"
        ):
            if float(catphan.dicom_stack.metadata.SliceThickness) < 3.5:
                self.pad = 1
            else:
                self.pad = 0
        else:
            self.pad = self.thickness_slice_straddle
        self.thickness_image = Slice(
            catphan,
            combine_method="mean",
            num_slices=self.num_slices + self.pad,
            slice_num=self.slice_num,
            clear_borders=self.clear_borders,
        ).image

    def _replace_hu_values(self):
        """Possibly replace the HU values in the ROI settings with the expected values if the key is present."""
        if self.expected_hu_values is not None:
            for name, value in self.expected_hu_values.items():
                if name in self.roi_settings:
                    self.roi_settings[name]["value"] = value

    def _setup_rois(self) -> None:
        self._replace_hu_values()
        super()._setup_rois()
        self._setup_thickness_rois()
        self._setup_geometry_rois()

    def _setup_thickness_rois(self) -> None:
        for name, setting in self.thickness_roi_settings.items():
            self.thickness_rois[name] = ThicknessROI.from_phantom_center(
                self.thickness_image,
                setting["width_pixels"],
                setting["height_pixels"],
                setting["angle_corrected"],
                setting["distance_pixels"],
                self.phan_center,
            )

    def _setup_geometry_rois(self) -> None:
        boxsize = self.geometry_roi_size_mm / self.mm_per_pixel
        xbounds = (int(self.phan_center.x - boxsize), int(self.phan_center.x + boxsize))
        ybounds = (int(self.phan_center.y - boxsize), int(self.phan_center.y + boxsize))
        geo_img = self.image[ybounds[0] : ybounds[1], xbounds[0] : xbounds[1]]
        # clip to the nearest of the two extremes
        # this can arise from direct density scans. In that case the
        # 1 teflon node will not get detected as the edge intensity is much less than the other nodes (unlike normal)
        # So, we clip the sub-image to the nearest extreme to the median.
        # This does very little to normal scans. RAM-4056
        median = np.median(geo_img)
        nearest_extreme = min(abs(median - geo_img.max()), abs(median - geo_img.min()))
        geo_clipped = np.clip(
            geo_img, a_min=median - nearest_extreme, a_max=median + nearest_extreme
        )
        larr, regionprops, num_roi = get_regions(
            geo_clipped, fill_holes=True, clear_borders=False
        )
        # check that there is at least 1 ROI
        if num_roi < 4:
            raise ValueError("Unable to locate the Geometric nodes")
        elif num_roi > 4:
            regionprops = sorted(
                regionprops, key=lambda x: x.filled_area, reverse=True
            )[:4]
        sorted_regions = sorted(
            regionprops, key=lambda x: (2 * x.centroid[0] + x.centroid[1])
        )
        centers = [
            Point(
                r.weighted_centroid[1] + xbounds[0], r.weighted_centroid[0] + ybounds[0]
            )
            for r in sorted_regions
        ]
        #  setup the geometric lines
        for name, order in self.geometry_roi_settings.items():
            self.lines[name] = GeometricLine(
                centers[order[0]],
                centers[order[1]],
                self.mm_per_pixel,
                self.scaling_tolerance,
            )

    @property
    def lcv(self) -> float:
        """The low-contrast visibility"""
        return (
            2
            * abs(self.rois["LDPE"].pixel_value - self.rois["Poly"].pixel_value)
            / (self.rois["LDPE"].std + self.rois["Poly"].std)
        )

    def plotly_linearity(
        self, plot_delta: bool = True, show_legend: bool = True
    ) -> go.Figure:
        fig = go.Figure()
        nominal_x_values = [roi.nominal_val for roi in self.rois.values()]
        if plot_delta:
            values = [roi.value_diff for roi in self.rois.values()]
            nominal_measurements = [0] * len(values)
            ylabel = "HU Delta"
        else:
            values = [roi.pixel_value for roi in self.rois.values()]
            nominal_measurements = nominal_x_values
            ylabel = "Measured Values"
        fig.add_scatter(
            x=nominal_x_values,
            y=values,
            name="Measured values",
            mode="markers",
            marker=dict(color="green", size=10, symbol="cross", line=dict(width=1)),
        )
        fig.add_scatter(
            x=nominal_x_values,
            y=nominal_measurements,
            mode="lines",
            name="Nominal Values",
            marker_color="green",
        )
        fig.add_scatter(
            x=nominal_x_values,
            y=np.array(nominal_measurements) + self.hu_tolerance,
            mode="lines",
            name="Upper Tolerance",
            line=dict(dash="dash", color="red"),
        )
        fig.add_scatter(
            x=nominal_x_values,
            y=np.array(nominal_measurements) - self.hu_tolerance,
            mode="lines",
            name="Lower Tolerance",
            line=dict(dash="dash", color="red"),
        )
        fig.update_layout(
            xaxis_title="Nominal Values", yaxis_title=ylabel, showlegend=show_legend
        )
        add_title(fig, "HU Linearity")
        return fig

    def plot_linearity(
        self, axis: plt.Axes | None = None, plot_delta: bool = True
    ) -> tuple:
        """Plot the HU linearity values to an axis.

        Parameters
        ----------
        axis : None, matplotlib.Axes
            The axis to plot the values on. If None, will create a new figure.
        plot_delta : bool
            Whether to plot the actual measured HU values (False), or the difference from nominal (True).
        """
        nominal_x_values = [roi.nominal_val for roi in self.rois.values()]
        if axis is None:
            fig, axis = plt.subplots()
        if plot_delta:
            values = [roi.value_diff for roi in self.rois.values()]
            nominal_measurements = [0] * len(values)
            ylabel = "HU Delta"
        else:
            values = [roi.pixel_value for roi in self.rois.values()]
            nominal_measurements = nominal_x_values
            ylabel = "Measured Values"
        points = axis.plot(nominal_x_values, values, "g+", markersize=15, mew=2)
        axis.plot(nominal_x_values, nominal_measurements)
        axis.plot(
            nominal_x_values, np.array(nominal_measurements) + self.hu_tolerance, "r--"
        )
        axis.plot(
            nominal_x_values, np.array(nominal_measurements) - self.hu_tolerance, "r--"
        )
        axis.margins(0.05)
        axis.grid(True)
        axis.set_xlabel("Nominal Values")
        axis.set_ylabel(ylabel)
        axis.set_title("HU linearity")
        return points

    @property
    def passed_hu(self) -> bool:
        """Boolean specifying whether all the ROIs passed within tolerance."""
        return all(roi.passed for roi in self.rois.values())

    def plotly_rois(self, fig: go.Figure) -> None:
        super().plotly_rois(fig)
        # plot slice thickness / ramp ROIs
        for name, roi in self.thickness_rois.items():
            roi.plotly(fig, line_color="blue", name=f"Ramp {name}")
        # plot geometry lines
        for name, line in self.lines.items():
            line.plotly(fig, color=line.pass_fail_color, name=f"Geometry {name}")

    def plot_rois(self, axis: plt.Axes) -> None:
        """Plot the ROIs onto the image, as well as the background ROIs"""
        # plot HU linearity ROIs
        super().plot_rois(axis)
        # plot thickness ROIs
        for roi in self.thickness_rois.values():
            roi.plot2axes(axis, edgecolor="blue")
        # plot geometry lines
        for line in self.lines.values():
            line.plot2axes(axis, color=line.pass_fail_color)

    @property
    def passed_thickness(self) -> bool:
        """Whether the slice thickness was within tolerance from nominal."""
        return (
            self.slice_thickness - self.thickness_tolerance
            < self.meas_slice_thickness
            < self.slice_thickness + self.thickness_tolerance
        )

    @property
    def meas_slice_thickness(self) -> float:
        """The average slice thickness for the 4 wire measurements in mm."""
        return np.mean(
            sorted(
                roi.wire_fwhm * self.mm_per_pixel * RAMP_ANGLE_RATIO
                for roi in self.thickness_rois.values()
            )
        ) / (1 + 2 * self.pad)

    @property
    def avg_line_length(self) -> float:
        return float(np.mean([line.length_mm for line in self.lines.values()]))

    @property
    def passed_geometry(self) -> bool:
        """Returns whether all the line lengths were within tolerance."""
        return all(line.passed for line in self.lines.values())

class CTP486(CatPhanModule):
    """Class for analysis of the Uniformity slice of the CTP module. Measures 5 ROIs around the slice that
    should all be close to the same value.
    """

    attr_name = "ctp486"
    common_name = "HU Uniformity"
    roi_dist_mm = 53
    roi_radius_mm = 10
    nominal_value = 0
    nps_rois: dict[str, RectangleROI]
    roi_settings = {
        "Top": {
            "value": nominal_value,
            "angle": -90,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "Right": {
            "value": nominal_value,
            "angle": 0,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "Bottom": {
            "value": nominal_value,
            "angle": 90,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "Left": {
            "value": nominal_value,
            "angle": 180,
            "distance": roi_dist_mm,
            "radius": roi_radius_mm,
        },
        "Center": {
            "value": nominal_value,
            "angle": 0,
            "distance": 0,
            "radius": roi_radius_mm,
        },
    }

    def plot_profiles(self, axis: plt.Axes | None = None) -> None:
        """Plot the horizontal and vertical profiles of the Uniformity slice.

        Parameters
        ----------
        axis : None, matplotlib.Axes
            The axis to plot on; if None, will create a new figure.
        """
        if axis is None:
            fig, axis = plt.subplots()
        horiz_data = self.image[int(self.phan_center.y), :]
        vert_data = self.image[:, int(self.phan_center.x)]
        axis.plot(horiz_data, "g", label="Horizontal")
        axis.plot(vert_data, "b", label="Vertical")
        axis.autoscale(tight=True)
        axis.axhline(self.nominal_value + self.tolerance, color="r", linewidth=3)
        axis.axhline(self.nominal_value - self.tolerance, color="r", linewidth=3)
        axis.grid(True)
        axis.set_ylabel("HU")
        axis.legend(loc=8, fontsize="small", title="")
        axis.set_title("Uniformity Profiles")

    def _setup_rois(self) -> None:
        """Generate our NPS ROIs. They are just square versions of the existing ROIs."""
        super()._setup_rois()
        self.nps_rois = {}
        for name, setting in self.roi_settings.items():
            self.nps_rois[name] = RectangleROI.from_phantom_center(
                array=self.image,
                width=setting["radius_pixels"] * 2,
                height=setting["radius_pixels"] * 2,
                angle=setting["angle_corrected"],
                dist_from_center=setting["distance_pixels"],
                phantom_center=self.phan_center,
            )

    def plot(self, axis: plt.Axes):
        """Plot the ROIs but also the noise power spectrum ROIs"""
        for nps_roi in self.nps_rois.values():
            nps_roi.plot2axes(axis, edgecolor="green", linestyle="-.")
        super().plot(axis)

    def plotly(self, **kwargs) -> go.Figure:
        fig = super().plotly(**kwargs)
        for name, nps_roi in self.nps_rois.items():
            nps_roi.plotly(
                fig, line_color="green", line_dash="dash", name=f"NPS {name}"
            )
        return fig

    @property
    def overall_passed(self) -> bool:
        """Boolean specifying whether all the ROIs passed within tolerance."""
        return all(roi.passed for roi in self.rois.values())

    @property
    def uniformity_index(self) -> float:
        """The Uniformity Index. Elstrom et al equation 2. https://www.tandfonline.com/doi/pdf/10.3109/0284186X.2011.590525"""
        center = self.rois["Center"]
        uis = [
            100 * ((roi.pixel_value - center.pixel_value) / (center.pixel_value + 1000))
            for roi in self.rois.values()
        ]
        abs_uis = np.abs(uis)
        return uis[np.argmax(abs_uis)]

    @property
    def integral_non_uniformity(self) -> float:
        """The Integral Non-Uniformity. Elstrom et al equation 1. https://www.tandfonline.com/doi/pdf/10.3109/0284186X.2011.590525"""
        maxhu = max(roi.pixel_value for roi in self.rois.values())
        minhu = min(roi.pixel_value for roi in self.rois.values())
        return (maxhu - minhu) / (maxhu + minhu + 2000)

    @cached_property
    def power_spectrum_2d(self) -> np.ndarray:
        """The power spectrum of the uniformity ROI."""
        return noise_power_spectrum_2d(
            pixel_size=self.mm_per_pixel,
            rois=[r.pixel_array for r in self.nps_rois.values()],
        )

    @cached_property
    def power_spectrum_1d(self) -> np.ndarray:
        """The 1D power spectrum of the uniformity ROI."""
        return noise_power_spectrum_1d(self.power_spectrum_2d)

    @property
    def avg_noise_power(self) -> float:
        """The average noise power of the uniformity ROI."""
        return average_power(self.power_spectrum_1d)

    @property
    def max_noise_power_frequency(self) -> float:
        """The frequency of the maximum noise power. 0 means no pattern."""
        return max_frequency(self.power_spectrum_1d)

class CTP528(CatPhanModule):

    """Class for analysis of the Spatial Resolution slice of the CBCT dicom data set.

    A collapsed circle profile is taken of the line-pair region. This profile is search for
    peaks and valleys. The MTF is calculated from those peaks & valleys.

    Attributes
    ----------

    radius2linepairs_mm : float
        The radius in mm to the line pairs.

    """

    attr_name: str = "ctp528"
    common_name: str = "Spatial Resolution"
    radius2linepairs_mm = 47
    combine_method: str = "max"
    num_slices: int = 3
    boundaries: tuple[float, ...] = (
        0,
        0.107,
        0.173,
        0.236,
        0.286,
        0.335,
        0.387,
        0.434,
        0.479,
    )
    start_angle: float = np.pi
    ccw: bool = True
    roi_settings = {
        "region 1": {
            "start": boundaries[0],
            "end": boundaries[1],
            "num peaks": 2,
            "num valleys": 1,
            "peak spacing": 0.021,
            "gap size (cm)": 0.5,
            "lp/mm": 0.1,
        },
        "region 2": {
            "start": boundaries[1],
            "end": boundaries[2],
            "num peaks": 3,
            "num valleys": 2,
            "peak spacing": 0.01,
            "gap size (cm)": 0.25,
            "lp/mm": 0.2,
        },
        "region 3": {
            "start": boundaries[2],
            "end": boundaries[3],
            "num peaks": 4,
            "num valleys": 3,
            "peak spacing": 0.006,
            "gap size (cm)": 0.167,
            "lp/mm": 0.3,
        },
        "region 4": {
            "start": boundaries[3],
            "end": boundaries[4],
            "num peaks": 4,
            "num valleys": 3,
            "peak spacing": 0.00557,
            "gap size (cm)": 0.125,
            "lp/mm": 0.4,
        },
        "region 5": {
            "start": boundaries[4],
            "end": boundaries[5],
            "num peaks": 4,
            "num valleys": 3,
            "peak spacing": 0.004777,
            "gap size (cm)": 0.1,
            "lp/mm": 0.5,
        },
        "region 6": {
            "start": boundaries[5],
            "end": boundaries[6],
            "num peaks": 5,
            "num valleys": 4,
            "peak spacing": 0.00398,
            "gap size (cm)": 0.083,
            "lp/mm": 0.6,
        },
        "region 7": {
            "start": boundaries[6],
            "end": boundaries[7],
            "num peaks": 5,
            "num valleys": 4,
            "peak spacing": 0.00358,
            "gap size (cm)": 0.071,
            "lp/mm": 0.7,
        },
        "region 8": {
            "start": boundaries[7],
            "end": boundaries[8],
            "num peaks": 5,
            "num valleys": 4,
            "peak spacing": 0.0027866,
            "gap size (cm)": 0.063,
            "lp/mm": 0.8,
        },
    }

    def _setup_rois(self):
        pass

    def _convert_units_in_settings(self):
        pass

    @cached_property
    def mtf(self) -> MTF:
        """The Relative MTF of the line pairs, normalized to the first region.

        Returns
        -------
        dict
        """
        maxs = list()
        mins = list()
        for key, value in self.roi_settings.items():
            max_indices, max_values = self.circle_profile.find_peaks(
                min_distance=value["peak spacing"],
                max_number=value["num peaks"],
                search_region=(value["start"], value["end"]),
            )
            # check that the right number of peaks were found before continuing, otherwise stop searching for regions
            if len(max_values) != value["num peaks"]:
                break
            maxs.append(max_values.mean())
            _, min_values = self.circle_profile.find_valleys(
                min_distance=value["peak spacing"],
                max_number=value["num valleys"],
                search_region=(min(max_indices), max(max_indices)),
            )
            mins.append(min_values.mean())
        if not maxs:
            raise ValueError(
                "Did not find any spatial resolution pairs to analyze. File an issue on github (https://github.com/jrkerns/pylinac/issues) if this is a valid dataset."
            )

        spacings = [roi["lp/mm"] for roi in self.roi_settings.values()]
        mtf = MTF(lp_spacings=spacings, lp_maximums=maxs, lp_minimums=mins)
        return mtf

    @property
    def radius2linepairs(self) -> float:
        """Radius from the phantom center to the line-pair region, corrected for pixel spacing."""
        return self.radius2linepairs_mm * self.scaling_factor / self.mm_per_pixel

    def plotly_rois(self, fig: go.Figure) -> None:
        self.circle_profile.plotly(fig, color="blue", plot_peaks=False)
        fig.update_layout(
            showlegend=False,
        )

    def plot_rois(self, axis: plt.Axes) -> None:
        """Plot the circles where the profile was taken within."""
        self.circle_profile.plot2axes(axis, edgecolor="blue", plot_peaks=False)

    @cached_property
    def circle_profile(self) -> CollapsedCircleProfile:
        """Calculate the median profile of the Line Pair region.

        Returns
        -------
        :class:`pylinac.core.profile.CollapsedCircleProfile` : A 1D profile of the Line Pair region.
        """
        circle_profile = CollapsedCircleProfile(
            self.phan_center,
            self.radius2linepairs,
            image_array=self.image,
            start_angle=self.start_angle + np.deg2rad(self.catphan_roll),
            width_ratio=0.04 * self.roi_size_factor,
            sampling_ratio=2,
            ccw=self.ccw,
        )
        circle_profile.filter(0.001, kind="gaussian")
        circle_profile.ground()
        return circle_profile

    
    """Class for analysis of the low contrast slice of the CTP module. Low contrast is measured by obtaining
    the average pixel value of the contrast ROIs and comparing that value to the average background value. To obtain
    a more "human" detection level, the contrast (which is largely the same across different-sized ROIs) is multiplied
    by the diameter. This value is compared to the contrast threshold to decide if it can be "seen".
    """

    attr_name = "ctp515"
    common_name = "Low Contrast"
    num_slices = 1
    roi_dist_mm = 50
    roi_radius_mm = [6, 3.5, 3, 2.5, 2, 1.5]
    roi_angles = [-87.4, -69.1, -52.7, -38.5, -25.1, -12.9]
    roi_settings = {
        "15": {
            "angle": roi_angles[0],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[0],
        },
        "9": {
            "angle": roi_angles[1],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[1],
        },
        "8": {
            "angle": roi_angles[2],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[2],
        },
        "7": {
            "angle": roi_angles[3],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[3],
        },
        "6": {
            "angle": roi_angles[4],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[4],
        },
        "5": {
            "angle": roi_angles[5],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[5],
        },
    }
    background_roi_dist_ratio = 0.75
    background_roi_radius_mm = 4
    WINDOW_SIZE = 50

    def __init__(
        self,
        catphan,
        tolerance: float,
        cnr_threshold: float,
        offset: int,
        contrast_method: str,
        visibility_threshold: float,
        clear_borders: bool = True,
    ):
        self.cnr_threshold = cnr_threshold
        self.contrast_method = contrast_method
        self.visibility_threshold = visibility_threshold
        super().__init__(
            catphan, tolerance=tolerance, offset=offset, clear_borders=clear_borders
        )

    def _setup_rois(self):
        # create both background rois dynamically, then create the actual sample ROI as normal
        for name, setting in self.roi_settings.items():
            self.background_rois[name + "-outer"] = (
                LowContrastDiskROI.from_phantom_center(
                    self.image,
                    setting["angle_corrected"],
                    self.background_roi_radius_mm / self.mm_per_pixel,
                    setting["distance_pixels"] * (2 - self.background_roi_dist_ratio),
                    self.phan_center,
                )
            )
            self.background_rois[name + "-inner"] = (
                LowContrastDiskROI.from_phantom_center(
                    self.image,
                    setting["angle_corrected"],
                    self.background_roi_radius_mm / self.mm_per_pixel,
                    setting["distance_pixels"] * self.background_roi_dist_ratio,
                    self.phan_center,
                )
            )
            background_val = float(
                np.mean(
                    [
                        self.background_rois[name + "-outer"].pixel_value,
                        self.background_rois[name + "-inner"].pixel_value,
                    ]
                )
            )

            self.rois[name] = LowContrastDiskROI.from_phantom_center(
                self.image,
                setting["angle_corrected"],
                setting["radius_pixels"],
                setting["distance_pixels"],
                self.phan_center,
                contrast_reference=background_val,
                cnr_threshold=self.cnr_threshold,
                contrast_method=self.contrast_method,
                visibility_threshold=self.visibility_threshold,
            )

    @property
    def rois_visible(self) -> int:
        """The number of ROIs "visible"."""
        return sum(roi.passed_visibility for roi in self.rois.values())

    @property
    def window_min(self) -> float:
        """Lower bound of CT window/leveling to show on the plotted image. Improves apparent contrast."""
        return (
            Enumerable(self.background_rois.values()).min(lambda r: r.pixel_value)
            - self.WINDOW_SIZE
        )

    @property
    def window_max(self) -> float:
        """Upper bound of CT window/leveling to show on the plotted image. Improves apparent contrast"""
        return (
            Enumerable(self.rois.values()).max(lambda r: r.pixel_value)
            + self.WINDOW_SIZE
        )

class CTP515(CatPhanModule):
    """Class for analysis of the low contrast slice of the CTP module. Low contrast is measured by obtaining
    the average pixel value of the contrast ROIs and comparing that value to the average background value. To obtain
    a more "human" detection level, the contrast (which is largely the same across different-sized ROIs) is multiplied
    by the diameter. This value is compared to the contrast threshold to decide if it can be "seen".
    """

    attr_name = "ctp515"
    common_name = "Low Contrast"
    num_slices = 1
    roi_dist_mm = 50
    roi_radius_mm = [6, 3.5, 3, 2.5, 2, 1.5]
    roi_angles = [-87.4, -69.1, -52.7, -38.5, -25.1, -12.9]
    roi_settings = {
        "15": {
            "angle": roi_angles[0],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[0],
        },
        "9": {
            "angle": roi_angles[1],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[1],
        },
        "8": {
            "angle": roi_angles[2],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[2],
        },
        "7": {
            "angle": roi_angles[3],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[3],
        },
        "6": {
            "angle": roi_angles[4],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[4],
        },
        "5": {
            "angle": roi_angles[5],
            "distance": roi_dist_mm,
            "radius": roi_radius_mm[5],
        },
    }
    background_roi_dist_ratio = 0.75
    background_roi_radius_mm = 4
    WINDOW_SIZE = 50

    def __init__(
        self,
        catphan,
        tolerance: float,
        cnr_threshold: float,
        offset: int,
        contrast_method: str,
        visibility_threshold: float,
        clear_borders: bool = True,
    ):
        self.cnr_threshold = cnr_threshold
        self.contrast_method = contrast_method
        self.visibility_threshold = visibility_threshold
        super().__init__(
            catphan, tolerance=tolerance, offset=offset, clear_borders=clear_borders
        )

    def _setup_rois(self):
        # create both background rois dynamically, then create the actual sample ROI as normal
        for name, setting in self.roi_settings.items():
            self.background_rois[name + "-outer"] = (
                LowContrastDiskROI.from_phantom_center(
                    self.image,
                    setting["angle_corrected"],
                    self.background_roi_radius_mm / self.mm_per_pixel,
                    setting["distance_pixels"] * (2 - self.background_roi_dist_ratio),
                    self.phan_center,
                )
            )
            self.background_rois[name + "-inner"] = (
                LowContrastDiskROI.from_phantom_center(
                    self.image,
                    setting["angle_corrected"],
                    self.background_roi_radius_mm / self.mm_per_pixel,
                    setting["distance_pixels"] * self.background_roi_dist_ratio,
                    self.phan_center,
                )
            )
            background_val = float(
                np.mean(
                    [
                        self.background_rois[name + "-outer"].pixel_value,
                        self.background_rois[name + "-inner"].pixel_value,
                    ]
                )
            )

            self.rois[name] = LowContrastDiskROI.from_phantom_center(
                self.image,
                setting["angle_corrected"],
                setting["radius_pixels"],
                setting["distance_pixels"],
                self.phan_center,
                contrast_reference=background_val,
                cnr_threshold=self.cnr_threshold,
                contrast_method=self.contrast_method,
                visibility_threshold=self.visibility_threshold,
            )

    @property
    def rois_visible(self) -> int:
        """The number of ROIs "visible"."""
        return sum(roi.passed_visibility for roi in self.rois.values())

    @property
    def window_min(self) -> float:
        """Lower bound of CT window/leveling to show on the plotted image. Improves apparent contrast."""
        return (
            Enumerable(self.background_rois.values()).min(lambda r: r.pixel_value)
            - self.WINDOW_SIZE
        )

    @property
    def window_max(self) -> float:
        """Upper bound of CT window/leveling to show on the plotted image. Improves apparent contrast"""
        return (
            Enumerable(self.rois.values()).max(lambda r: r.pixel_value)
            + self.WINDOW_SIZE
        )




@capture_warnings
class CatPhan504(CatPhanBase):
    """A class for loading and analyzing CT DICOM files of a CatPhan 504. Can be from a CBCT or CT scanner
    Analyzes: Uniformity (CTP486), High-Contrast Spatial Resolution (CTP528),
    Image Scaling & HU Linearity (CTP404), and Low contrast (CTP515).
    """

    _demo_url = "CatPhan504.zip"
    _model = "504"
    catphan_radius_mm = 101
    modules = {
        CTP404: {"offset": 0},
        CTP486: {"offset": -65},
        CTP528: {"offset": 30},
        CTP515: {"offset": -30},
    }

    @staticmethod
    def run_demo(show: bool = True):
        """Run the CBCT demo using high-quality head protocol images."""
        cbct = CatPhan504.from_demo_images()
        cbct.analyze()
        print(cbct.results())
        cbct.plot_analyzed_image(show)
