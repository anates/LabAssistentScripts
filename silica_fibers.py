#!/usr/bin/env python3
#
# Created on May 25 2022
#
# Created by Roland Axel Richter -- roland.richter@empa.ch
#
# Copyright (c) 2022
"""_summary_

Returns:
    _type_: _description_
"""
from __future__ import annotations
import numpy as np
import scipy.constants as scco
from scipy.interpolate import interp1d
import matplotlib

matplotlib.use("Qt5Cairo")


class FiberTemplate:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        self.dispersion_data_file = ""
        self.name = "Fiber-Template"
        self.core_diameter = 1.0
        self.cladding_diameter = 1.0
        self.coating_diameter = 1.0


class SMF28(FiberTemplate):
    """_summary_

    Args:
        FiberTemplate (_type_): _description_
    """

    def __init__(self) -> None:
        """_summary_"""
        super().__init__()
        self.dispersion_data_file = "Dispersion_SMF28.csv"
        self.name = "SMF-28"
        self.core_diameter = 8.2e-6
        self.cladding_diameter = 125e-6
        self.coating_diameter = 250e-6


class LMA25(FiberTemplate):
    """_summary_

    Args:
        FiberTemplate (_type_): _description_
    """

    def __init__(self) -> None:
        """_summary_"""
        super().__init__()
        self.dispersion_data_file = "Dispersion_SMF28.csv"
        self.name = "LMA-25"
        self.core_diameter = 25e-6
        self.cladding_diameter = 250e-6
        self.coating_diameter = 400e-6


class UHNA4(FiberTemplate):
    """_summary_

    Args:
        FiberTemplate (_type_): _description_
    """

    def __init__(self) -> None:
        """_summary_"""
        super().__init__()
        self.dispersion_data_file = "Dispersion_UHNA4.csv"
        self.name = "UHNA-4"
        self.core_diameter = 2.2e-6
        self.cladding_diameter = 125e-6
        self.coating_diameter = 250e-6


class UHNA3(FiberTemplate):
    """_summary_

    Args:
        FiberTemplate (_type_): _description_
    """

    def __init__(self) -> None:
        """_summary_"""
        super().__init__()
        self.dispersion_data_file = "Dispersion_UHNA3.csv"
        self.name = "UHNA-3"
        self.core_diameter = 1.8e-6
        self.cladding_diameter = 125e-6
        self.coating_diameter = 250e-6


class Fiber:
    """_summary_"""

    def __init__(
        self,
        fiber_dispersion_data: str,
        core_size: float,
        cladding_sizes: list[float],
        coating_size: float,
        name: str,
    ) -> None:
        """_summary_

        Args:
            fiber_dispersion_data (str): _description_
            core_size (float): _description_
            cladding_sizes (list[float]): _description_
            coating_size (float): _description_
            name (str): _description_
        """
        self.dispersion_data = fiber_dispersion_data
        self.core_size = core_size
        self.cladding_sizes = cladding_sizes
        self.name = name
        self.coating_size = coating_size
        self.effective_area = (core_size / 2) ** 2 * np.pi
        self.beta_vals = []
        self.d_vals = []
        self.wl_vals = []
        self.create_fiber()

    @classmethod
    def from_fiber_class(cls, fiber_template: FiberTemplate) -> Fiber:
        """_summary_

        Args:
            fiber_template (FiberTemplate): _description_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(fiber_template.cladding_diameter, list):
            return Fiber(
                fiber_template.dispersion_data_file,
                fiber_template.core_diameter,
                fiber_template.cladding_diameter,
                fiber_template.coating_diameter,
                fiber_template.name,
            )
        if isinstance(fiber_template.cladding_diameter, float):
            return Fiber(
                fiber_template.dispersion_data_file,
                fiber_template.core_diameter,
                [fiber_template.cladding_diameter],
                fiber_template.coating_diameter,
                fiber_template.name,
            )
        exception_string = (
            "Invalid parameter given for fiber_template.cladding_diameter. "
        )
        exception_string += (
            f"Variable is of type {type(fiber_template.cladding_diameter)},"
        )
        exception_string += ", but only list[float] or float are allowed."
        raise Exception(exception_string)

    def create_fiber(self) -> None:
        """_summary_"""
        fiber_wl_data = []
        fiber_ref_data = []
        with open(self.dispersion_data, "r", encoding="utf-8") as file_reader:
            data_lines = file_reader.readlines()
            for line in data_lines:
                data = line.split(";")
                if len(data) == 2:
                    fiber_wl_data.append(float(data[0].replace(",", ".")) * 1e-9)
                    fiber_ref_data.append(float(data[1].replace(",", ".")))
        self.d_vals = np.asarray(fiber_ref_data) * 1e-12 / (1e-9 * 1e3)
        self.wl_vals = np.asarray(fiber_wl_data)
        self.beta_vals = self.d_vals * (
            -1 * np.power(self.wl_vals, 2) / (2 * np.pi * scco.speed_of_light)
        )
        self.beta_vals_interpolator = interp1d(self.wl_vals, self.beta_vals)
        self.d_vals_interpolator = interp1d(self.wl_vals, self.d_vals)

    def determine_wavelength_range(self, wavelength: float) -> float:
        """_summary_

        Args:
            wavelength (_type_): _description_

        Returns:
            _type_: _description_
        """
        if wavelength < 0:
            return wavelength
        if wavelength > 0.1 and wavelength < 10:
            return wavelength * 1e-6  # We use um as wavelength
        if wavelength > 100:
            return wavelength * 1e-9  # We use nm as wavelength
        if wavelength < 0.1:
            return wavelength  # We use m as wavelength
        raise Exception(f"Wavelength value {wavelength} out of bounds")

    def calc_beta_val(self, wavelength: float):
        """_summary_

        Args:
            wavelength (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.beta_vals_interpolator(self.determine_wavelength_range(wavelength))

    def calc_d_val(self, wavelength: float):
        """_summary_

        Args:
            wavelength (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.d_vals_interpolator(self.determine_wavelength_range(wavelength))
