#!/usr/bin/env python3
#
# Created on May 25 2022
#
# Created by Roland Axel Richter -- roland.richter@empa.ch
#
# Copyright (c) 2022
""" _summary_

Returns:
    _type_: _description_
"""
from scipy import interpolate
import numpy as np


class LaserDiode:
    """_summary_"""

    def __init__(self, current_points: np.ndarray, power_points: np.ndarray) -> None:
        """__init__ _summary_

        Args:
            current_points (np.ndarray): _description_
            power_points (np.ndarray): _description_
        """
        self.current_points = current_points
        self.power_points = power_points
        self.power_function = interpolate.interp1d(
            self.current_points,
            self.power_points,
            kind="linear",
            fill_value="extrapolate",
        )

    def get_power(self, input_current: float) -> float:
        """get_power _summary_

        Args:
            input_current (_type_): _description_

        Returns:
            float: _description_
        """
        power = self.power_function(input_current)
        if power < 0:
            return 0.0
        else:
            return self.power_function(input_current)

    def get_power_from_voltage(
        self, input_voltage: float
    ) -> float:  # if LD is based on voltage
        """get_power_from_voltage _summary_

        Args:
            input_voltage (_type_): _description_

        Returns:
            float: _description_
        """
        return self.get_power(input_voltage / 0.8)


class DILAS_I(LaserDiode):
    """DILAS_I _summary_

    Args:
        LaserDiode (_type_): _description_
    """

    def __init__(self) -> None:
        """__init__ _summary_"""
        measured_current_points = np.array([10, 12.5, 15, 17.5, 20], dtype=np.float64)
        measured_power_points = np.array(
            [1630, 3220, 4770, 6380, 7930], dtype=np.float64
        )
        super().__init__(measured_current_points, measured_power_points)
        self.name = "Dilas Diode I"


class DILAS_II(LaserDiode):
    """DILAS_II _summary_

    Args:
        LaserDiode (_type_): _description_
    """

    def __init__(self) -> None:
        """__init__ _summary_"""
        measured_current_points = np.array(
            [8, 10, 12, 14, 16, 18, 20], dtype=np.float64
        )
        measured_power_points = np.array(
            [258, 1978, 3698, 5418, 7138, 8858, 10578], dtype=np.float64
        )
        super().__init__(measured_current_points, measured_power_points)
        self.name = "Dilas Diode II"


class DILAS_III(LaserDiode):
    """DILAS_III _summary_

    Args:
        LaserDiode (_type_): _description_
    """

    def __init__(self) -> None:
        """__init__ _summary_"""
        measured_current_points = np.array(
            [8, 10, 12, 14, 16, 18, 20], dtype=np.float64
        )
        measured_power_points = np.array(
            [570, 2470, 4370, 6270, 8170, 10070, 11970], dtype=np.float64
        )
        super().__init__(measured_current_points, measured_power_points)
        self.name = "Dilas Diode III"


class nLight(LaserDiode):
    """nLight _summary_

    Args:
        LaserDiode (_type_): _description_
    """

    def __init__(self) -> None:
        """__init__ _summary_"""
        measured_current_points = np.array(
            [
                0.720,
                0.785,
                0.845,
                0.910,
            ],
            dtype=np.float64,
        )
        measured_power_points = np.array([20, 252, 615, 990], dtype=np.float64)
        super().__init__(measured_current_points, measured_power_points)
        self.name = "nLight-Diode"

    def get_power(self, input_voltage: float) -> float:
        """get_power _summary_

        Args:
            input_voltage (_type_): _description_

        Returns:
            float: _description_
        """
        return super().get_power(input_voltage / 0.8)
