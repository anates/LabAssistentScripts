#!/usr/bin/env python3
#
# Created on May 25 2022
#
# Created by Roland Axel Richter -- roland.richter@empa.ch
#
# Copyright (c) 2022
from typing import Union
import numpy as np
from scipy.interpolate import interp1d
import matplotlib

matplotlib.use("Qt5Cairo")


class MirrorFilterILMA:
    """_summary_"""

    def __init__(self) -> None:
        """__init__ _summary_"""
        self.mirror_data_file = "109612 ASCII.csv"
        self.scaling_factor = 1
        self.wl_shift = 7
        self.name = "First mirror"
        self.line_parameters = [5, 1]


class MirrorFilterIILMA:
    """_summary_"""

    def __init__(self) -> None:
        """__init__ _summary_"""
        self.mirror_data_file = "1800 mirror.csv"
        self.scaling_factor = 95 / 58
        self.wl_shift = -6
        self.name = "Second mirror"
        self.line_parameters = [3, 2, 2700]


class FilterMirror:
    """_summary_"""

    def __init__(
        self,
        mirror_data: str,
        scaling_factor: float,
        wl_shift: float,
        name: str,
        line_parameters: Union[tuple[float, float], tuple[float, float, float]],
    ) -> None:
        """__init__ _summary_

        Args:
            mirror_data (_type_): _description_
            scaling_factor (_type_): _description_
            wl_shift (_type_): _description_
            name (_type_): _description_
            line_parameters (list, optional): _description_. Defaults to [].
        """
        self.mirror_data_file = mirror_data
        self.scaling_factor = scaling_factor
        self.wl_shift = wl_shift
        self.name = name
        self.line_parameters = line_parameters
        self.create_mirror()

    @classmethod
    def from_mirror_class(cls, mirror_template) -> FilterMirror:
        """fromMirrorClass _summary_

        Args:
            filter_mirror (_type_): _description_
            mirror_template (_type_): _description_

        Returns:
            _type_: _description_
        """
        return FilterMirror(
            mirror_template.mirror_data_file,
            mirror_template.scaling_factor,
            mirror_template.wl_shift,
            mirror_template.name,
            mirror_template.line_parameters,
        )

    def apply_mirror(self, wl_data, meas_data) -> tuple[np.ndarray, np.ndarray]:
        """apply_mirror _summary_

        Args:
            wl_data (_type_): _description_
            meas_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ret_meas_data = []
        for i, elem in enumerate(wl_data):
            ret_meas_data.append(self.mirror_interpolator(elem) * meas_data[i])
        return (np.asarray(wl_data), np.asarray(ret_meas_data))

    def create_mirror(self) -> None:
        """create_mirror _summary_"""
        mirror_wl_data = []
        mirror_ref_data = []
        with open(self.mirror_data_file, "r", encoding="utf-8") as mirror_file_reader:
            data_lines = mirror_file_reader.readlines()
            for line_number, line in enumerate(data_lines):
                data = line.split(",")
                if len(self.line_parameters) == 2:
                    if (
                        len(data) == self.line_parameters[0]
                        and line_number > self.line_parameters[1]
                    ):
                        mirror_wl_data.append(float(data[0]) + self.wl_shift)
                        mirror_ref_data.append(float(data[1]))
                else:
                    if len(self.line_parameters) == 3:
                        if (
                            len(data) == self.line_parameters[0]
                            and line_number > self.line_parameters[1]
                            and line_number < self.line_parameters[2]
                        ):
                            mirror_wl_data.append(float(data[0]) + self.wl_shift)
                            mirror_ref_data.append(float(data[1]))
        self.mirror_wl_data = np.flip(np.asarray(mirror_wl_data))
        self.mirror_ref_data = 1 / (
            1 - self.scaling_factor * np.flip(np.asarray(mirror_ref_data)) / 100
        )
        self.mirror_interpolator = interp1d(self.mirror_wl_data, self.mirror_ref_data)
