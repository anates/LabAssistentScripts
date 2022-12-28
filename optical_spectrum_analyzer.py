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
from typing import Any, Optional
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import scipy.constants as scco
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import tikzplotlib
from labellines import labelLines

from itertools import islice

from filter_mirrors import FilterMirror


def nth_key(dct, n) -> Any:
    it = iter(dct)
    next(islice(it, n, n), None)
    return next(it)


class OSA_spectrum:
    """_summary_"""

    def __init__(
        self,
        wl_data: np.ndarray,
        linear_meas_data: np.ndarray,
        log_meas_data: np.ndarray,
        spectrum_name: str,
        linear_data_below_threshold: np.ndarray = np.zeros(0),
        normalized_data: np.ndarray = np.zeros(0),
        normalized_data_below_threshold: np.ndarray = np.zeros(0),
        normalized_log_data: np.ndarray = np.zeros(0),
        normalized_log_data_below_threshold: np.ndarray = np.zeros(0),
        threshold_data: float = -1,
        stack_spectrum: bool = False,
    ) -> None:
        """__init__ _summary_

        Args:
            wl_data (np.ndarray): _description_
            linear_meas_data (np.ndarray): _description_
            log_meas_data (np.ndarray): _description_
            spectrum_name (str): _description_
            linear_data_below_threshold (np.ndarray, optional): _description_. Defaults to np.zeros(0).
            normalized_data (np.ndarray, optional): _description_. Defaults to np.zeros(0).
            normalized_data_below_threshold (np.ndarray, optional): _description_. Defaults to np.zeros(0).
            normalized_log_data (np.ndarray, optional): _description_. Defaults to np.zeros(0).
            normalized_log_data_below_threshold (np.ndarray, optional): _description_. Defaults to np.zeros(0).
            threshold_data (float, optional): _description_. Defaults to -1.
            stack_spectrum (bool, optional): _description_. Defaults to False.
        """
        self.wl_data = wl_data
        self.linear_meas_data = linear_meas_data
        self.threshold_data = threshold_data
        if threshold_data < 0:
            self.cut_off_linear_meas_data = self.linear_meas_data.copy()
        else:
            self.cut_off_linear_meas_data = linear_data_below_threshold
        self.log_meas_data = log_meas_data
        self.normalized_data = normalized_data
        self.normalized_data_below_threshold = normalized_data_below_threshold
        self.normalized_log_data = normalized_log_data
        self.normalized_log_data_below_treshold = normalized_log_data_below_threshold
        self.spectrum_name = spectrum_name
        self.spectrum_is_stacked = stack_spectrum
        self.fit_function = "None"
        self.fit_meas_data = np.zeros(0)
        self.fit_data = np.zeros(0)
        self.FWHM = 0.0
        self.bandwidth = 0.0
        self.fit_can_be_used = False


class OSA:
    """_summary_"""

    def __init__(self) -> None:
        """__init__ _summary_

        Returns:
            _type_: _description_
        """
        self.spectra = {}

    def butter_lowpass_filter(
        self, data: np.ndarray, cutoff: float, sampling_frequency: float, order: int
    ) -> np.ndarray:
        """butter_lowpass_filter _summary_

        Args:
            data (np.ndarray): _description_
            cutoff (float): _description_
            sampling_frequency (float): _description_
            order (int): _description_

        Returns:
            np.ndarray: _description_
        """
        normal_cutoff = cutoff / (sampling_frequency * 0.5)
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    def gaussian_function_compact(self, x: float, data: list[float]) -> float:
        """gaussian_function_compact _summary_

        Args:
            x (float): _description_
            data (list[float]): _description_

        Returns:
            float: _description_
        """
        return data[3] + data[0] * np.sqrt(2 / np.pi) / data[1] * np.exp(
            -4 * np.log(2) * ((x - data[2]) / data[1]) ** 2
        )

    def gaussian_function(
        self, x: float, A: float, xc: float, w: float, y0: float
    ) -> float:
        """gaussian_function _summary_

        Args:
            x (float): _description_
            A (float): _description_
            xc (float): _description_
            w (float): _description_
            y0 (float): _description_

        Returns:
            float: _description_
        """
        return self.gaussian_function_compact(x, [A, xc, w, y0])

    def sech_function_compact(self, x: float, data: list[float]) -> float:
        """sech_function_compact _summary_

        Args:
            x (float): _description_
            data (list[float]): _description_

        Returns:
            float: _description_
        """
        y_val = (x - data[2]) / data[1]
        return data[3] + data[0] / np.cosh(y_val)
        # return data[3] + data[0]*np.exp(-2*((x-data[2])/data[1])**2)

    def sech_function(
        self, x: float, A: float, xc: float, w: float, y0: float
    ) -> float:
        """sech_function _summary_

        Args:
            x (float): _description_
            A (float): _description_
            xc (float): _description_
            w (float): _description_
            y0 (float): _description_

        Returns:
            float: _description_
        """
        return self.sech_function_compact(x, [A, xc, w, y0])

    def determine_wavelength_range(self, wavelength: float) -> float:
        """determine_wavelength_range _summary_

        Args:
            wavelength (float): _description_

        Returns:
            float: _description_
        """
        if wavelength < 0:
            # passthrough for -1
            return wavelength
        if wavelength > 100:
            # I assume it is given in nm
            return wavelength
        if wavelength > 0.1 and wavelength < 10:
            # I assume it is given in um
            return wavelength * 1e3
        # else: Wavelength is given in m
        return wavelength * 1e9

    def calc_single_spectrum_energy(
        self,
        spectrum: OSA_spectrum,
        minimum_wavelength: float,
        maximum_wavelength: float,
    ) -> float:
        """calc_single_spectrum_energy _summary_

        Args:
            spectrum (OSA_spectrum): _description_
            minimum_wavelength (float): _description_
            maximum_wavelength (float): _description_

        Returns:
            float: _description_
        """
        local_lin_meas_data = spectrum.linear_meas_data
        local_wl_data = spectrum.wl_data
        minimum_index = 0
        maximum_index = 0
        if minimum_wavelength < local_wl_data[0] or minimum_wavelength < 0:
            minimum_wavelength = local_wl_data[0]
            minimum_index = 0
        else:
            # Holy shit, that is a crappy algorithm!
            cur_index_distance = np.abs(local_wl_data[0] - minimum_wavelength)
            cur_index = 1
            while np.abs(
                local_wl_data[cur_index] - minimum_wavelength
            ) < cur_index_distance and cur_index < len(local_wl_data):
                cur_index_distance = np.abs(
                    local_wl_data[cur_index] - minimum_wavelength
                )
                cur_index += 1
            minimum_index = cur_index
        if maximum_wavelength > local_wl_data[-1] or maximum_wavelength < 0:
            maximum_wavelength = local_wl_data[-1]
            maximum_index = len(local_wl_data) - 1
        else:
            # Holy shit, that is a crappy algorithm!
            cur_index_distance = np.abs(local_wl_data[-1] - maximum_wavelength)
            cur_index = len(local_wl_data) - 2
            while np.abs(
                local_wl_data[cur_index] - maximum_wavelength
            ) < cur_index_distance and cur_index < len(local_wl_data):
                cur_index_distance = np.abs(
                    local_wl_data[cur_index] - maximum_wavelength
                )
                cur_index -= 1
            maximum_index = cur_index
        cur_energy = 0
        full_energy = 0
        """
        for j, elem in enumerate(local_lin_meas_data):
            cur_energy = elem
            if j == 0 or j == len(local_lin_meas_data) - 1:
                cur_energy *= 0.5
            full_energy += cur_energy
            """
        for j in range(minimum_index, maximum_index + 1):
            cur_energy = local_lin_meas_data[j]
            if j == minimum_index or j == maximum_index:
                cur_energy *= 0.5
            full_energy += cur_energy
        full_energy *= 1e-9
        return full_energy

    def calc_spectrum_energy(
        self,
        spectrum_number: int,
        minimum_wavelength: float = -1,
        maximum_wavelength: float = -1,
    ) -> dict[str, float]:
        """calc_spectrum_energy _summary_

        Args:
            spectrum_number (_type_): _description_
            minimum_wavelength (int, optional): _description_. Defaults to -1.
            maximum_wavelength (int, optional): _description_. Defaults to -1.

        Returns:
            dict[str, float]: _description_
        """
        spectrum_energies = {}
        if spectrum_number < 0 or spectrum_number > (len(self.spectra) - 1):
            for i in range(len(self.spectra)):
                spectrum_energies[
                    self.spectra[[*self.spectra][i]].spectrum_name
                ] = self.calc_single_spectrum_energy(
                    self.spectra[[*self.spectra][i]],
                    self.determine_wavelength_range(minimum_wavelength),
                    self.determine_wavelength_range(maximum_wavelength),
                )
        else:
            spectrum_energies[
                self.spectra[[*self.spectra][spectrum_number]].spectrum_name
            ] = self.calc_single_spectrum_energy(
                self.spectra[[*self.spectra][spectrum_number]],
                self.determine_wavelength_range(minimum_wavelength),
                self.determine_wavelength_range(maximum_wavelength),
            )
        return spectrum_energies

    def get_raw_spectrum_data(
        self,
        filename: str,
        spectrum_name: str,
        data_threshold: float,
        scaling_factor: float = 1,
    ) -> None:
        """get_raw_spectrum_data _summary_

        Args:
            filename (str): _description_
            spectrum_name (str): _description_
            data_threshold (float): _description_
            scaling_factor (float, optional): _description_. Defaults to 1.
        """
        wl_data = []
        meas_data = []
        with open(filename, "r", encoding="utf-8") as data_reader:
            lines = data_reader.readlines()
            if "AQ6375" in lines[1].split(" "):
                # Usual OSA
                for line_number, line in enumerate(lines):
                    if (
                        line_number > 30
                        and (len(line.split(",")) or len(line.split("\t"))) == 2
                    ):
                        line_data = line.split(",")
                        if len(line_data) != 2:
                            line_data = line.split("\t")
                        wl_data.append(float(line_data[0]))
                        meas_data.append(float(line_data[1]))
            else:
                # Small OSA
                for line_number, line in enumerate(lines):
                    if line_number > 1 and len(line.split("\t")) == 2:
                        line_data = line.split("\t")
                        if len(line_data) != 2:
                            line_data = line.split("\t")
                        wl_data.append(float(line_data[0]))
                        # adjust to something useful. I assume that 1 in the small OSA is equivalent to 1e-6 in the large OSA
                        meas_data.append(float(line_data[1]) * 1e-6)
                        if meas_data[-1] == 0:
                            meas_data[-1] = 1e-13
        meas_data = np.asarray(meas_data)
        wl_data = np.asarray(wl_data)
        if meas_data[0] < 0:
            linear_meas_data = np.power(10, meas_data / 10) * 1e-3
            log_meas_data = meas_data
        else:
            linear_meas_data = np.asarray(meas_data)
            log_meas_data = 10 * np.log10(meas_data)
        # Applying scaling/etc
        linear_meas_data *= scaling_factor
        normalized_linear_meas_data = linear_meas_data.copy()
        normalized_linear_meas_data -= np.min(normalized_linear_meas_data)
        normalized_linear_meas_data /= np.max(normalized_linear_meas_data)

        normalized_log_meas_data = log_meas_data.copy()
        normalized_log_meas_data -= np.min(normalized_log_meas_data)
        normalized_log_meas_data /= np.max(normalized_log_meas_data)

        if data_threshold > 0:
            linear_threshold_data_indices = linear_meas_data > data_threshold
            linear_threshold_data = linear_meas_data.copy()
            linear_threshold_data[linear_threshold_data_indices] = data_threshold
        wl_data = np.asarray(wl_data)
        if data_threshold > 0:
            self.spectra[spectrum_name] = OSA_spectrum(
                wl_data,
                linear_meas_data,
                log_meas_data,
                spectrum_name,
                linear_data_below_threshold=linear_threshold_data,
                normalized_data=normalized_linear_meas_data,
                normalized_log_data=normalized_log_meas_data,
                threshold_data=data_threshold,
            )
        else:
            self.spectra[spectrum_name] = OSA_spectrum(
                wl_data,
                linear_meas_data,
                log_meas_data,
                spectrum_name,
                normalized_data=normalized_linear_meas_data,
                normalized_log_data=normalized_log_meas_data,
                threshold_data=data_threshold,
            )

    def get_spectrum_data(
        self,
        filename: str,
        spectrum_name: str,
        data_threshold: float = -1,
        scaling_factor: float = 1,
    ) -> int:
        """get_spectrum_data _summary_

        Args:
            filename (str): _description_
            spectrum_name (str): _description_
            data_threshold (float, optional): _description_. Defaults to -1.
            scaling_factor (float, optional): _description_. Defaults to 1.

        Returns:
            int: _description_
        """
        self.get_raw_spectrum_data(
            filename,
            spectrum_name,
            data_threshold=data_threshold,
            scaling_factor=scaling_factor,
        )
        return len(self.spectra)

    def apply_filter_mirror(
        self, spectrum_number: int, filter_mirror: FilterMirror
    ) -> None:
        """apply_filter_mirror _summary_

        Args:
            spectrum_number (int): _description_
            filter_mirror (FilterMirror): _description_
        """
        if spectrum_number < 0 or spectrum_number > (len(self.spectra) - 1):
            for i in range(len(self.spectra)):
                local_lin_meas_data = self.spectra[[*self.spectra][i]].linear_meas_data
                local_wl_data = self.spectra[[*self.spectra][i]].wl_data
                interpolated_data = filter_mirror.apply_mirror(
                    local_wl_data, local_lin_meas_data
                )
                self.spectra[[*self.spectra][i]].linear_meas_data = np.asarray(
                    interpolated_data[1][:]
                )
                interpolated_threshold_indices = (
                    np.asarray(interpolated_data[1][:])
                    > self.spectra[[*self.spectra][i]].threshold_data
                )
                self.spectra[[*self.spectra][i]].cut_off_linear_meas_data = np.asarray(
                    interpolated_data[1][:]
                )
                # self.spectra[[*self.spectra][i]].cut_off_linear_meas_data[interpolated_threshold_indices] = self.spectra[[*self.spectra][i]].threshold_data
                self.spectra[[*self.spectra][i]].log_meas_data = 10 * np.log10(
                    interpolated_data[1][:] / 1e-3
                )

                normalized_linear_meas_data = self.spectra[
                    [*self.spectra][i]
                ].linear_meas_data.copy()
                normalized_linear_meas_data -= np.min(normalized_linear_meas_data)
                normalized_linear_meas_data /= np.max(normalized_linear_meas_data)
                self.spectra[
                    [*self.spectra][i]
                ].normalized_data = normalized_linear_meas_data.copy()

                normalized_log_meas_data = self.spectra[
                    [*self.spectra][i]
                ].log_meas_data.copy()
                normalized_log_meas_data -= np.min(normalized_log_meas_data)
                normalized_log_meas_data /= np.max(normalized_log_meas_data)
                self.spectra[
                    [*self.spectra][i]
                ].normalized_log_data = normalized_log_meas_data.copy()
        else:
            local_lin_meas_data = self.spectra[
                [*self.spectra][spectrum_number]
            ].linear_meas_data
            local_wl_data = self.spectra[[*self.spectra][spectrum_number]].wl_data
            interpolated_data = filter_mirror.apply_mirror(
                local_wl_data, local_lin_meas_data
            )
            self.spectra[
                [*self.spectra][spectrum_number]
            ].linear_meas_data = np.asarray(interpolated_data[1][:])
            self.spectra[
                [*self.spectra][spectrum_number]
            ].log_meas_data = 10 * np.log10(interpolated_data[1][:] / 1e-3)

            normalized_linear_meas_data = self.spectra[
                [*self.spectra][spectrum_number]
            ].linear_meas_data.copy()
            normalized_linear_meas_data -= np.min(normalized_linear_meas_data)
            normalized_linear_meas_data /= np.max(normalized_linear_meas_data)
            self.spectra[
                [*self.spectra][spectrum_number]
            ].normalized_data = normalized_linear_meas_data.copy()

            normalized_log_meas_data = self.spectra[
                [*self.spectra][spectrum_number]
            ].log_meas_data.copy()
            normalized_log_meas_data -= np.min(normalized_log_meas_data)
            normalized_log_meas_data /= np.max(normalized_log_meas_data)
            self.spectra[
                [*self.spectra][spectrum_number]
            ].normalized_log_data = normalized_log_meas_data.copy()

    def remove_filter_mirror(
        self, spectrum_number: int, filter_mirror: FilterMirror
    ) -> None:
        """remove_filter_mirror _summary_

        Args:
            spectrum_number (int): _description_
            filter_mirror (FilterMirror): _description_
        """
        if spectrum_number < 0 or spectrum_number > (len(self.spectra) - 1):
            for i in range(len(self.spectra)):
                local_lin_meas_data = self.spectra[[*self.spectra][i]].linear_meas_data
                local_wl_data = self.spectra[[*self.spectra][i]].wl_data
                interpolated_data = filter_mirror.remove_mirror(
                    local_wl_data, local_lin_meas_data
                )
                self.spectra[[*self.spectra][i]].linear_meas_data = np.asarray(
                    interpolated_data[1][:]
                )
                interpolated_threshold_indices = (
                    np.asarray(interpolated_data[1][:])
                    > self.spectra[[*self.spectra][i]].threshold_data
                )
                self.spectra[[*self.spectra][i]].cut_off_linear_meas_data = np.asarray(
                    interpolated_data[1][:]
                )
                # self.spectra[[*self.spectra][i]].cut_off_linear_meas_data[interpolated_threshold_indices] = self.spectra[[*self.spectra][i]].threshold_data
                self.spectra[[*self.spectra][i]].log_meas_data = 10 * np.log10(
                    interpolated_data[1][:] / 1e-3
                )

                normalized_linear_meas_data = self.spectra[
                    [*self.spectra][i]
                ].linear_meas_data.copy()
                normalized_linear_meas_data -= np.min(normalized_linear_meas_data)
                normalized_linear_meas_data /= np.max(normalized_linear_meas_data)
                self.spectra[
                    [*self.spectra][i]
                ].normalized_data = normalized_linear_meas_data.copy()

                normalized_log_meas_data = self.spectra[
                    [*self.spectra][i]
                ].log_meas_data.copy()
                normalized_log_meas_data -= np.min(normalized_log_meas_data)
                normalized_log_meas_data /= np.max(normalized_log_meas_data)
                self.spectra[
                    [*self.spectra][i]
                ].normalized_log_data = normalized_log_meas_data.copy()
        else:
            local_lin_meas_data = self.spectra[
                [*self.spectra][spectrum_number]
            ].linear_meas_data
            local_wl_data = self.spectra[[*self.spectra][spectrum_number]].wl_data
            interpolated_data = filter_mirror.remove_mirror(
                local_wl_data, local_lin_meas_data
            )
            self.spectra[
                [*self.spectra][spectrum_number]
            ].linear_meas_data = np.asarray(interpolated_data[1][:])
            self.spectra[
                [*self.spectra][spectrum_number]
            ].log_meas_data = 10 * np.log10(interpolated_data[1][:] / 1e-3)

            normalized_linear_meas_data = self.spectra[
                [*self.spectra][spectrum_number]
            ].linear_meas_data.copy()
            normalized_linear_meas_data -= np.min(normalized_linear_meas_data)
            normalized_linear_meas_data /= np.max(normalized_linear_meas_data)
            self.spectra[
                [*self.spectra][spectrum_number]
            ].normalized_data = normalized_linear_meas_data.copy()

            normalized_log_meas_data = self.spectra[
                [*self.spectra][spectrum_number]
            ].log_meas_data.copy()
            normalized_log_meas_data -= np.min(normalized_log_meas_data)
            normalized_log_meas_data /= np.max(normalized_log_meas_data)
            self.spectra[
                [*self.spectra][spectrum_number]
            ].normalized_log_data = normalized_log_meas_data.copy()

    def plot_spectrum(
        self,
        spectrum_number: int,
        use_stacked_spectra: bool = False,
        use_cube_spectra: bool = False,
        use_linear_scale: bool = False,
        use_normalized_data: bool = False,
        add_wavenumbers_on_top: bool = False,
        with_fit_FWHM: bool = False,
        x_min: float = -1,
        x_max: float = -1,
        y_min: float = -1,
        y_max: float = -1,
        use_grid: bool = False,
        number_precision: int = 3,
        put_legend_in_lines: bool = False,
        legend_in_line_xvals: list = [],
    ):

        fig = plt.figure()
        if use_cube_spectra:
            ax = plt.subplot(1, 1, 1, projection="3d")
            spectrum_keys = self.spectra.keys()
            for entry, key in enumerate(spectrum_keys):
                value = self.spectra[key]
                ax.plot(
                    value.wl_data,
                    np.ones(value.wl_data.size) * entry * 0.1,
                    value.normalized_data,
                )
                ax.add_collection3d(
                    plt.fill_between(
                        value.wl_data,
                        0.95 * value.normalized_data,
                        1.05 * value.normalized_data,
                        alpha=0.3,
                    ),
                    zs=entry,
                    zdir="y",
                )
        else:
            ax = plt.subplot(111)
            plt.xlabel("Wavelength [nm]")
            if y_max <= y_min:
                adjust_height_freely = True
            else:
                adjust_height_freely = False
            if not use_normalized_data:
                if use_linear_scale:
                    plt.ylabel("Intensity in [mW/nm]")
                else:
                    plt.ylabel("Intensity in [dBm/nm]")
            else:
                if use_linear_scale:
                    plt.ylabel("Intensity in [mW/nm], normalized")
                else:
                    plt.ylabel("Intensity in [dBm/nm], normalized")
            legend_entries = []
            plt.grid(use_grid)
            if (
                spectrum_number < 0 or spectrum_number > len(self.spectra) - 1
            ):  # plot all
                spectrum_entries = self.spectra.items()
                spectrum_keys = self.spectra.keys()
                for entry, key in enumerate(spectrum_keys):
                    value = self.spectra[key]
                    legend_entry = value.spectrum_name
                    linear_data_entry = []
                    log_data_entry = []
                    normalized_linear_data_entry = []
                    normalized_log_data_entry = []
                    if with_fit_FWHM and value.fit_can_be_used:
                        legend_entry += (
                            ", FWHM = "
                            + "{0:.{prec}f}".format(value.FWHM, prec=number_precision)
                            + " nm, "
                            + "{0:.{prec}f}".format(
                                value.bandwidth / 1e9, prec=number_precision
                            )
                            + " GHz"
                        )
                    if use_stacked_spectra and entry > 0:
                        # I assume that the spectra are put into the dictionary in order
                        log_shift_factor = np.max(value.log_meas_data) - np.min(
                            value.log_meas_data
                        )
                        cut_off_lin_shift_factor = np.max(
                            value.cut_off_linear_meas_data
                        )
                        lin_shift_factor = np.max(value.linear_meas_data)
                        normalized_lin_shift_factor = np.max(value.normalized_data)
                        normalized_log_shift_factor = np.max(value.normalized_log_data)
                        linear_data_entry = value.linear_meas_data + lin_shift_factor
                        log_data_entry = value.log_meas_data + log_shift_factor
                        normalized_linear_data_entry = (
                            value.normalized_data + normalized_lin_shift_factor
                        )
                        normalized_log_data_entry = (
                            value.normalized_log_data + normalized_log_shift_factor
                        )
                    else:
                        log_shift_factor = 0
                        cut_off_lin_shift_factor = 0
                        lin_shift_factor = 0
                        normalized_lin_shift_factor = 0
                        normalized_log_shift_factor = 0
                        linear_data_entry = value.linear_meas_data + lin_shift_factor
                        log_data_entry = value.log_meas_data + log_shift_factor
                        normalized_linear_data_entry = (
                            value.normalized_data + normalized_lin_shift_factor
                        )
                        normalized_log_data_entry = (
                            value.normalized_log_data + normalized_log_shift_factor
                        )
                    if not use_normalized_data:
                        if use_linear_scale:
                            plt.plot(
                                value.wl_data, linear_data_entry, label=legend_entry
                            )
                        else:
                            plt.plot(value.wl_data, log_data_entry, label=legend_entry)
                    else:
                        if use_linear_scale:
                            plt.plot(
                                value.wl_data,
                                normalized_linear_data_entry,
                                label=legend_entry,
                            )
                        else:
                            plt.plot(
                                value.wl_data,
                                normalized_log_data_entry,
                                label=legend_entry,
                            )
                    # legend_entries.append(legend_entry)
                    if x_min < 0:
                        x_min = value.wl_data[0]
                    if x_max < 0:
                        x_max = value.wl_data[-1]
                    if adjust_height_freely:
                        if not use_normalized_data:
                            if use_linear_scale:
                                y_min = 0
                                y_min = np.min([np.min(linear_data_entry), y_min])
                            else:
                                y_min = np.min([np.min(log_data_entry), y_min])
                            if use_linear_scale:
                                y_max = np.max([np.max(linear_data_entry), y_max])
                            else:
                                y_max = np.max([np.max(log_data_entry), y_max])
                        else:
                            if use_linear_scale:
                                y_min = np.min(
                                    [np.min(normalized_linear_data_entry), y_min]
                                )
                                y_max = np.max(
                                    [np.max(normalized_linear_data_entry), y_max]
                                )
                            else:
                                y_min = np.min(
                                    [np.min(normalized_log_data_entry), y_min]
                                )
                                y_max = np.max(
                                    [np.max(normalized_log_data_entry), y_max]
                                )
            else:
                local_value = self.spectra[
                    [*self.spectra][spectrum_number]
                ]  # nth_key(self.spectra, spectrum_number)
                if use_linear_scale:
                    plt.plot(
                        local_value.wl_data,
                        local_value.cut_off_linear_meas_data,
                        label=local_value.spectrum_name,
                    )
                else:
                    plt.plot(
                        local_value.wl_data,
                        local_value.log_meas_data,
                        label=local_value.spectrum_name,
                    )
            if put_legend_in_lines:
                if not legend_in_line_xvals:
                    labelLines(plt.gca().get_lines())
                else:
                    print(legend_in_line_xvals)
                    labelLines(plt.gca().get_lines(), xvals=legend_in_line_xvals)
            else:
                plt.legend()
            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))
            if add_wavenumbers_on_top:
                ax_top = ax.twiny()
                ax_top.set_xlabel("Wavenumbers [cm^-1]")
                ax_top.set_xticks(ax.get_xticks())
                ax_top.set_xbound(ax.get_xbound())
                ax_top.set_xticklabels([(int)(10e6 / x) for x in ax.get_xticks()])
            if use_stacked_spectra:
                plt.yticks([])
        plt.show()

    def save_spectrum(
        self,
        spectrum_number: int,
        file_name: str,
        legend_position: str = "upper center",
        file_format: str = "PNG",
        use_linear_scale: bool = False,
        use_cube_spectra: bool = False,
        use_stacked_spectra: bool = True,
        use_normalized_data: bool = False,
        add_wavenumbers_on_top: bool = False,
        x_min: float = -1,
        x_max: float = -1,
        y_min: float = -1,
        y_max: float = -1,
        use_grid: bool = False,
        with_fit_FWHM: bool = False,
        plot_title: str = "",
        number_precision: int = 3,
        put_legend_in_lines: bool = False,
        legend_in_line_xvals: list = [],
        reduce_to_every_nth_entry: int = 1,
    ) -> None:
        """save_spectrum _summary_

        Args:
            spectrum_number (int): _description_
            file_name (str): _description_
            legend_position (str, optional): _description_. Defaults to "upper center".
            file_format (str, optional): _description_. Defaults to "PNG".
            use_linear_scale (bool, optional): _description_. Defaults to False.
            use_cube_spectra (bool, optional): _description_. Defaults to False.
            use_stacked_spectra (bool, optional): _description_. Defaults to True.
            use_normalized_data (bool, optional): _description_. Defaults to False.
            add_wavenumbers_on_top (bool, optional): _description_. Defaults to False.
            x_min (float, optional): _description_. Defaults to -1.
            x_max (float, optional): _description_. Defaults to -1.
            y_min (float, optional): _description_. Defaults to -1.
            y_max (float, optional): _description_. Defaults to -1.
            use_grid (bool, optional): _description_. Defaults to False.
            with_fit_FWHM (bool, optional): _description_. Defaults to False.
            plot_title (str, optional): _description_. Defaults to "".
            number_precision (int, optional): _description_. Defaults to 3.
            put_legend_in_lines (bool, optional): _description_. Defaults to False.
            legend_in_line_xvals (list, optional): _description_. Defaults to [].
            reduce_to_every_nth_entry (int, optional): _description_. Defaults to 1.
        """

        fig = plt.figure()
        if use_cube_spectra:
            ax = plt.subplot(1, 1, 1, projection="3d")
            spectrum_keys = self.spectra.keys()
            for entry, key in enumerate(spectrum_keys):
                value = self.spectra[key]
                ax.plot(
                    value.wl_data,
                    np.ones(value.wl_data.size) * entry * 0.1,
                    value.normalized_data,
                )
                ax.add_collection3d(
                    plt.fill_between(
                        value.wl_data,
                        0.95 * value.normalized_data,
                        1.05 * value.normalized_data,
                        alpha=0.3,
                    ),
                    zs=entry,
                    zdir="y",
                )
        else:
            ax = plt.subplot(111)
            if y_max <= y_min:
                adjust_height_freely = True
            else:
                adjust_height_freely = False
            plt.xlabel("Wavelength [nm]")
            if not use_normalized_data:
                if use_linear_scale:
                    plt.ylabel("Intensity in [mW/nm]")
                else:
                    plt.ylabel("Intensity in [dBm/nm]")
            else:
                if use_linear_scale:
                    plt.ylabel("Intensity in [mW/nm], normalized")
                else:
                    plt.ylabel("Intensity in [dBm/nm], normalized")
            legend_entries = []
            ax.grid(use_grid)
            if (
                spectrum_number < 0 or spectrum_number > len(self.spectra) - 1
            ):  # plot all
                spectrum_entries = self.spectra.items()
                spectrum_keys = self.spectra.keys()
                for entry, key in enumerate(spectrum_keys):
                    value = self.spectra[key]
                    linear_data_entry = []
                    log_data_entry = []
                    normalized_linear_data_entry = []
                    normalized_log_data_entry = []
                    legend_entry = value.spectrum_name
                    if with_fit_FWHM and value.fit_can_be_used:
                        legend_entry += (
                            ", FWHM = "
                            + "{0:.{prec}f}".format(value.FWHM, prec=number_precision)
                            + " nm, "
                            + "{0:.{prec}f}".format(
                                value.bandwidth / 1e9, prec=number_precision
                            )
                            + " GHz"
                        )
                    legend_entries.append(legend_entry)
                    if use_stacked_spectra and entry > 0:
                        # I assume that the spectra are put into the dictionary in order
                        log_shift_factor = np.max(
                            list(self.spectra.items())[entry - 1][1].log_meas_data
                        ) - np.min(log_meas_data)
                        cut_off_lin_shift_factor = np.max(
                            list(self.spectra.items())[entry - 1][
                                1
                            ].cut_off_linear_meas_data
                        )
                        lin_shift_factor = np.max(
                            list(self.spectra.items())[entry - 1][1].linear_meas_data
                        )
                        normalized_lin_shift_factor = np.max(
                            list(self.spectra.items())[entry - 1][1].normalized_data
                        )
                        normalized_log_shift_factor = np.max(
                            list(self.spectra.items())[entry - 1][1].normalized_log_data
                        )
                        linear_data_entry = value.linear_meas_data + lin_shift_factor
                        log_data_entry = value.log_meas_data + log_shift_factor
                        normalized_linear_data_entry = (
                            value.normalized_data + normalized_lin_shift_factor
                        )
                        normalized_log_data_entry = (
                            value.normalized_log_data + normalized_log_shift_factor
                        )
                    else:
                        log_shift_factor = 0
                        cut_off_lin_shift_factor = 0
                        lin_shift_factor = 0
                        normalized_lin_shift_factor = 0
                        normalized_log_shift_factor = 0
                        linear_data_entry = value.linear_meas_data + lin_shift_factor
                        log_data_entry = value.log_meas_data + log_shift_factor
                        normalized_linear_data_entry = (
                            value.normalized_data + normalized_lin_shift_factor
                        )
                        normalized_log_data_entry = (
                            value.normalized_log_data + normalized_log_shift_factor
                        )
                    if not use_normalized_data:
                        if use_linear_scale:
                            plt.plot(
                                value.wl_data[::reduce_to_every_nth_entry],
                                linear_data_entry[::reduce_to_every_nth_entry],
                                label=legend_entry,
                            )
                        else:
                            plt.plot(
                                value.wl_data[::reduce_to_every_nth_entry],
                                log_data_entry[::reduce_to_every_nth_entry],
                                label=legend_entry,
                            )
                    else:
                        if use_linear_scale:
                            plt.plot(
                                value.wl_data[::reduce_to_every_nth_entry],
                                normalized_linear_data_entry[
                                    ::reduce_to_every_nth_entry
                                ],
                                label=legend_entry,
                            )
                        else:
                            plt.plot(
                                value.wl_data[::reduce_to_every_nth_entry],
                                normalized_log_data_entry[::reduce_to_every_nth_entry],
                                label=legend_entry,
                            )

                    if x_min < 0:
                        x_min = np.min(value.wl_data[0], x_min)
                    if x_max < 0:
                        x_max = np.max(value.wl_data[-1], x_max)
                    if adjust_height_freely:
                        if not use_normalized_data:
                            if use_linear_scale:
                                y_min = np.min([np.min(linear_data_entry), y_min])
                            else:
                                y_min = np.min([np.min(log_data_entry), y_min])
                            if use_linear_scale:
                                y_max = np.max([np.max(linear_data_entry), y_max])
                            else:
                                y_max = np.max([np.max(log_data_entry), y_max])
                        else:
                            if use_linear_scale:
                                y_min = np.min(
                                    [np.min(normalized_linear_data_entry), y_min]
                                )
                                y_max = np.max(
                                    [np.max(normalized_linear_data_entry), y_max]
                                )
                            else:
                                y_min = np.min(
                                    [np.min(normalized_log_data_entry), y_min]
                                )
                                y_max = np.max(
                                    [np.max(normalized_log_data_entry), y_max]
                                )
            else:
                local_value = self.spectra[
                    [*self.spectra][spectrum_number]
                ]  # nth_key(self.spectra, spectrum_number)
                if use_linear_scale:
                    ax.plot(local_value.wl_data, local_value.cut_off_linear_meas_data)
                else:
                    ax.plot(local_value.wl_data, local_value.log_meas_data)
                legend_entries.append(local_value.spectrum_name)
            if put_legend_in_lines:
                if not legend_in_line_xvals:
                    labelLines(plt.gca().get_lines())
                else:
                    print(legend_in_line_xvals)
                    labelLines(plt.gca().get_lines(), xvals=legend_in_line_xvals)
            else:
                if "outside" in legend_position:
                    outside_legend_position = (
                        legend_position.split(" ")[1]
                        + " "
                        + legend_position.split(" ")[2]
                    )
                    anchor_pos = (0, 0)
                    if "lower left" in legend_position:
                        anchor_pos = (1.04, 0)
                    if "center left" in legend_position:
                        anchor_pos = (1.04, 0.5)
                    if "upper left" in legend_position:
                        anchor_pos = (1.04, 1)
                    ax.legend(
                        legend_entries,
                        bbox_to_anchor=anchor_pos,
                        loc=outside_legend_position,
                    )
                else:
                    ax.legend(legend_entries, loc=legend_position)
            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))
            if add_wavenumbers_on_top:
                ax_top = ax.twiny()
                if file_format == "TIKZ":
                    ax_top.set_xlabel("Wavenumbers " + r"$\left\[cm^-1\right\]$")
                else:
                    ax_top.set_xlabel("Wavenumbers [cm^-1]")
                ax_top.set_xticks(ax.get_xticks())
                ax_top.set_xbound(ax.get_xbound())
                ax_top.set_xticklabels([int(10e6 / (x)) for x in ax.get_xticks()])
                # ax_top.set_xticklabels([int(x) for x in ax.get_xticks()])
                # ax_top.xaxis.set_major_locator(plt.MaxNLocator())

            if use_stacked_spectra:
                plt.yticks([])
        if plot_title:
            plt.title(plot_title)
        local_file_name = file_name
        if file_format == "PNG":
            if local_file_name[-4:] != ".png":
                local_file_name += ".png"
            plt.savefig(local_file_name, bbox_inches="tight", dpi=600)
        else:
            if file_format == "PGF":
                if local_file_name[-4:] != ".pgf":
                    local_file_name += ".pgf"
                plt.savefig(local_file_name, bbox_inches="tight", dpi=600)
            else:
                if local_file_name[-5:] != ".tikz":
                    local_file_name += ".tikz"
                tikzplotlib.clean_figure(target_resolution=600)
                tikzplotlib.save(local_file_name)
        plt.clf()

    def fit_spectrum(
        self,
        external_spectrum_number: int = -1,
        estimated_wavelength: float = 0,
        minimum_wavelength: float = -np.inf,
        maximum_wavelength: float = np.inf,
        fit_function: str = "Gaussian",
    ) -> None:
        """fit_spectrum _summary_

        Args:
            external_spectrum_number (int, optional): _description_. Defaults to -1.
            estimated_wavelength (float, optional): _description_. Defaults to 0.
            minimum_wavelength (float, optional): _description_. Defaults to -np.inf.
            maximum_wavelength (float, optional): _description_. Defaults to np.inf.
            fit_function (str, optional): _description_. Defaults to "Gaussian".
        """
        # find peak
        central_wavelength = estimated_wavelength
        if external_spectrum_number < 0:
            for spectrum_number in range(len(self.spectra)):
                spectrum = self.spectra[[*self.spectra][spectrum_number]]
                peaks, _ = find_peaks(spectrum.linear_meas_data, prominence=(0.001, 1))
                if len(spectrum.wl_data[peaks]) > 0:
                    central_wavelength = float(spectrum.wl_data[peaks][0])
                if fit_function == "Gaussian":
                    if (
                        minimum_wavelength > spectrum.wl_data[0]
                        or maximum_wavelength < spectrum.wl_data[-1]
                    ):
                        center_wl_val = spectrum.wl_data.flat[
                            np.abs(spectrum.wl_data - estimated_wavelength).argmin()
                        ]
                        center_wl_pos = spectrum.wl_data.tolist().index(center_wl_val)
                        if minimum_wavelength <= spectrum.wl_data[0]:
                            lower_wl_val = spectrum.wl_data[0]
                        else:
                            lower_wl_val = minimum_wavelength
                        if maximum_wavelength >= spectrum.wl_data[-1]:
                            upper_wl_val = spectrum.wl_data[-1]
                        else:
                            upper_wl_val = maximum_wavelength
                        closest_lower_wl_val = spectrum.wl_data.flat[
                            np.abs(spectrum.wl_data - lower_wl_val).argmin()
                        ]
                        lower_wl_pos = spectrum.wl_data.tolist().index(
                            closest_lower_wl_val
                        )
                        closest_upper_wl_val = spectrum.wl_data.flat[
                            np.abs(spectrum.wl_data - upper_wl_val).argmin()
                        ]
                        upper_wl_pos = spectrum.wl_data.tolist().index(
                            closest_upper_wl_val
                        )
                        target_wl_scale = np.asarray(
                            spectrum.wl_data[lower_wl_pos:upper_wl_pos]
                        )
                        target_val_scale = np.asarray(
                            spectrum.linear_meas_data[lower_wl_pos:upper_wl_pos]
                        )
                    else:
                        target_wl_scale = spectrum.wl_data[:]
                        target_val_scale = spectrum.linear_meas_data[:]

                    popt, _ = curve_fit(
                        self.gaussian_function,
                        target_wl_scale,
                        target_val_scale,
                        p0=[np.max(target_val_scale), 1, estimated_wavelength, 0],
                    )

                    print("Estimated bandwidth:", popt[1])
                    print("Estimated central wavelength:", popt[2])
                    plt.clf()
                    plt.plot(
                        self.spectra[[*self.spectra][spectrum_number]].wl_data,
                        self.spectra[[*self.spectra][spectrum_number]].linear_meas_data,
                        self.spectra[[*self.spectra][spectrum_number]].wl_data,
                        self.gaussian_function_compact(
                            self.spectra[[*self.spectra][spectrum_number]].wl_data, popt
                        ),
                    )
                    plt.show()

                    self.spectra[
                        [*self.spectra][spectrum_number]
                    ].fit_function = "Gaussian"
                    self.spectra[[*self.spectra][spectrum_number]].FWHM = popt[1]
                    self.spectra[
                        [*self.spectra][spectrum_number]
                    ].bandwidth = scco.speed_of_light / (
                        (popt[2] - popt[1] / 2) * 1e-9
                    ) - scco.speed_of_light / (
                        (popt[2] + popt[1] / 2) * 1e-9
                    )
                else:
                    popt, pcopt = curve_fit(
                        self.sech_function,
                        spectrum.wl_data,
                        spectrum.linear_meas_data,
                        p0=[
                            np.max(spectrum.linear_meas_data),
                            1,
                            estimated_wavelength,
                            0,
                        ],
                    )
                    self.spectra[[*self.spectra][spectrum_number]].fit_function = "Sech"
                    self.spectra[[*self.spectra][spectrum_number]].FWHM = (
                        2 * np.log(2 + np.sqrt(3)) * popt[1]
                    )
                    self.spectra[
                        [*self.spectra][spectrum_number]
                    ].bandwidth = scco.speed_of_light / (
                        (
                            popt[2]
                            - self.spectra[[*self.spectra][spectrum_number]].FWHM / 2
                        )
                        * 1e-9
                    ) - scco.speed_of_light / (
                        (
                            popt[2]
                            + self.spectra[[*self.spectra][spectrum_number]].FWHM / 2
                        )
                        * 1e-9
                    )
                self.spectra[
                    [*self.spectra][spectrum_number]
                ].fit_meas_data = self.gaussian_function_compact(spectrum.wl_data, popt)
                self.spectra[[*self.spectra][spectrum_number]].fit_data = popt
                self.spectra[[*self.spectra][spectrum_number]].fit_can_be_used = True
        else:
            spectrum_number = external_spectrum_number
            spectrum = self.spectra[[*self.spectra][spectrum_number]]
            peaks, _ = find_peaks(spectrum.linear_meas_data, prominence=(0.001, 1))
            if len(spectrum.wl_data[peaks]) > 0:
                central_wavelength = float(spectrum.wl_data[peaks][0])
            if fit_function == "Gaussian":
                if (
                    minimum_wavelength > spectrum.wl_data[0]
                    or maximum_wavelength < spectrum.wl_data[-1]
                ):
                    center_wl_val = spectrum.wl_data.flat[
                        np.abs(spectrum.wl_data - estimated_wavelength).argmin()
                    ]
                    center_wl_pos = spectrum.wl_data.tolist().index(center_wl_val)
                    if minimum_wavelength <= spectrum.wl_data[0]:
                        lower_wl_val = spectrum.wl_data[0]
                    else:
                        lower_wl_val = minimum_wavelength
                    if maximum_wavelength >= spectrum.wl_data[-1]:
                        upper_wl_val = spectrum.wl_data[-1]
                    else:
                        upper_wl_val = maximum_wavelength
                    closest_lower_wl_val = spectrum.wl_data.flat[
                        np.abs(spectrum.wl_data - lower_wl_val).argmin()
                    ]
                    lower_wl_pos = spectrum.wl_data.tolist().index(closest_lower_wl_val)
                    closest_upper_wl_val = spectrum.wl_data.flat[
                        np.abs(spectrum.wl_data - upper_wl_val).argmin()
                    ]
                    upper_wl_pos = spectrum.wl_data.tolist().index(closest_upper_wl_val)
                    target_wl_scale = np.asarray(
                        spectrum.wl_data[lower_wl_pos:upper_wl_pos]
                    )
                    target_val_scale = np.asarray(
                        spectrum.linear_meas_data[lower_wl_pos:upper_wl_pos]
                    )
                else:
                    target_wl_scale = spectrum.wl_data[:]
                    target_val_scale = spectrum.linear_meas_data[:]

                popt, pcopt = curve_fit(
                    self.gaussian_function,
                    target_wl_scale,
                    target_val_scale,
                    p0=[np.max(target_val_scale), 1, estimated_wavelength, 0],
                )

                print("Estimated bandwidth:", popt[1])
                print("Estimated central wavelength:", popt[2])
                plt.clf()
                plt.plot(
                    self.spectra[[*self.spectra][spectrum_number]].wl_data,
                    self.spectra[[*self.spectra][spectrum_number]].linear_meas_data,
                    self.spectra[[*self.spectra][spectrum_number]].wl_data,
                    self.gaussian_function_compact(
                        self.spectra[[*self.spectra][spectrum_number]].wl_data, popt
                    ),
                )
                plt.show()

                self.spectra[[*self.spectra][spectrum_number]].fit_function = "Gaussian"
                self.spectra[[*self.spectra][spectrum_number]].FWHM = popt[1]
                self.spectra[
                    [*self.spectra][spectrum_number]
                ].bandwidth = scco.speed_of_light / (
                    (popt[2] - popt[1] / 2) * 1e-9
                ) - scco.speed_of_light / (
                    (popt[2] + popt[1] / 2) * 1e-9
                )
            else:
                popt, pcopt = curve_fit(
                    self.sech_function,
                    spectrum.wl_data,
                    spectrum.linear_meas_data,
                    p0=[np.max(spectrum.linear_meas_data), 1, estimated_wavelength, 0],
                )
                self.spectra[[*self.spectra][spectrum_number]].fit_function = "Sech"
                self.spectra[[*self.spectra][spectrum_number]].FWHM = (
                    2 * np.log(2 + np.sqrt(3)) * popt[1]
                )
                self.spectra[
                    [*self.spectra][spectrum_number]
                ].bandwidth = scco.speed_of_light / (
                    (popt[2] - self.spectra[[*self.spectra][spectrum_number]].FWHM / 2)
                    * 1e-9
                ) - scco.speed_of_light / (
                    (popt[2] + self.spectra[[*self.spectra][spectrum_number]].FWHM / 2)
                    * 1e-9
                )
            self.spectra[
                [*self.spectra][spectrum_number]
            ].fit_meas_data = self.gaussian_function_compact(spectrum.wl_data, popt)
            self.spectra[[*self.spectra][spectrum_number]].fit_data = popt
            self.spectra[[*self.spectra][spectrum_number]].fit_can_be_used = True
