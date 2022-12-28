#!/usr/bin/env python3
""" _summary_

Returns:
    _type_: _description_
"""
from typing import Union
from enum import Enum
import numpy as np
import scipy.constants as scco
from scipy import fftpack
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


import tikzplotlib


class ReferenceMethod(Enum):
    """Reference_Method _summary_

    Args:
        Enum (_type_): _description_
    """

    USE_RIPPLES = 1
    USE_DISTANCE = 2


class AutoCorrelator:
    """_summary_"""

    def __init__(
        self,
        file_wide: str = "",
        file_close: str = "",
        central_wavelength: float = 2000,
        files: tuple[str, str] = ("", ""),
        peak_distance: float = -1,
        use_fft_for_peak: bool = True,
        plot_distance_data: bool = False,
        path_difference: float = 1e-3,
        use_normalized_data: bool = False,
        reference: ReferenceMethod = ReferenceMethod.USE_RIPPLES,
        print_peak_data: bool = False,
    ) -> None:
        """__init__ _summary_

        Args:
            file_wide (str, optional): _description_. Defaults to "".
            file_close (str, optional): _description_. Defaults to "".
            central_wavelength (float, optional): _description_. Defaults to 2000.
            files (list, optional): _description_. Defaults to [].
            peak_distance (float, optional): _description_. Defaults to -1.
            use_fft_for_peak (bool, optional): _description_. Defaults to True.
            plot_distance_data (bool, optional): _description_. Defaults to False.
            path_difference (float, optional): _description_. Defaults to 1e-3.
            use_normalized_data (bool, optional): _description_. Defaults to False.
            reference (ReferenceMethod, optional): _description_. Defaults to ReferenceMethod.USE_RIPPLES.
            print_peak_data (bool, optional): _description_. Defaults to False.
        """
        # pylint:disable=too-many-arguments, too-many-branches, dangerous-default-value
        self.gaussian_fit_calculated = False
        self.sech_fit_calculated = False
        self.time_correction_factor_wide = 0
        self.time_correction_factor_close = 0
        self.use_normalized_data = use_normalized_data

        # init of values used in later functions
        self.sech_pulse_fwhm = 0.0
        self.sech_fwhm = 0.0
        self.sech_data = list()

        self.gaussian_pulse_fwhm = 0.0
        self.gaussian_fwhm = 0.0
        self.gaussian_data = list()

        self.filter_frequency = 0.0

        self.raw_time_data = np.zeros(0)
        self.raw_meas_data = np.zeros(0)
        self.time_data = np.zeros(0)
        self.meas_data = np.zeros(0)

        self.gaussian_used = False
        self.sech_used = False

        # I assume we are just using wavelengths between 1e-9 and 100e-6
        if central_wavelength > 100e-6:
            # Central wavelength is given in nm
            self.central_wavelength = central_wavelength * 1e-9
        else:
            # Everything is fine
            self.central_wavelength = central_wavelength
        if reference is ReferenceMethod.USE_RIPPLES:
            if file_wide and file_close:
                self.file_wide = file_wide
                self.file_close = file_close
            else:
                if len(files) == 2:
                    self.get_file_destination(files)
                else:
                    self.file_wide = files[0]
                    self.file_close = files[0]
            if peak_distance < 0:
                if len(files) >= 2 or (file_wide and file_close):
                    self.get_peak_distance(
                        use_fft=use_fft_for_peak,
                        plot_data=plot_distance_data,
                        print_peak_data=print_peak_data,
                    )
                else:
                    self.get_peak_distance_from_single_file(
                        single_data_file=files[0],
                        use_fft=use_fft_for_peak,
                        plot_data=plot_distance_data,
                    )
            else:
                self.peak_distance = peak_distance
            self.prefactor = (
                1 / self.peak_distance * (self.central_wavelength) / scco.speed_of_light
            )
        else:
            if file_wide and file_close:
                self.peak_distance = self.get_peak_distance_from_path_distance(
                    path_difference,
                    (file_wide, file_close),
                    plot_data=plot_distance_data,
                )
            else:
                self.peak_distance = self.get_peak_distance_from_path_distance(
                    path_difference, files, plot_data=plot_distance_data
                )
            self.prefactor = self.peak_distance
        if print_peak_data:
            print("Peak distance:", self.peak_distance)

    def get_file_destination(self, files: tuple[str, str]) -> None:
        """get_file_destination _summary_

        Args:
            files (list[str]): _description_
        """
        if len(files) != 2:
            error_string = ""
            if len(files) == 1:
                error_string = "One file submitted. Please submit exactly two files."
            else:
                error_string = (
                    f"{len(files)} files submitted. Please submit exactly two files."
                )
            raise Exception(error_string)
        file_1 = files[0]
        file_2 = files[1]
        time_scale_1 = -1
        time_scale_2 = -1
        with open(file_1, "r", encoding="utf-8") as filereader_1:
            lines = filereader_1.readlines()
            for line in lines:
                if "Horizontal Scale" in line:
                    line_data = line.split(",")
                    time_scale_1 = float(line_data[1])
                    break

        with open(file_2, "r", encoding="utf-8") as filereader_2:
            lines = filereader_2.readlines()
            for line in lines:
                if "Horizontal Scale" in line:
                    line_data = line.split(",")
                    time_scale_2 = float(line_data[1])
                    break

        if time_scale_1 < 0 or time_scale_2 < 0:
            raise Exception(
                "Error reading files, please proceed with separate files. No time scale could be found."
            )
        if time_scale_1 > time_scale_2:
            self.file_wide = file_1
            self.file_close = file_2
        else:
            self.file_close = file_1
            self.file_wide = file_2
        print(f"File_close: {self.file_close}, File_wide: {self.file_wide}")

    def get_peak_distance_from_single_file(
        self,
        single_data_file: str,
        use_fft: bool = True,
        plot_data: bool = True,
        max_time_window: float = 1e-3,
    ) -> None:
        """get_peak_distance_from_single_file _summary_

        Args:
            single_data_file (str): _description_
            use_fft (bool, optional): _description_. Defaults to True.
            plot_data (bool, optional): _description_. Defaults to True.
            max_time_window (float, optional): _description_. Defaults to 1e-3.
        """
        file_data = self.get_raw_autocorrelation_data(single_data_file)
        # find center of spectrum/time
        center_time_val = file_data[0].flat[np.abs(file_data[0]).argmin()]
        lower_time_val = center_time_val - (max_time_window / 2)
        upper_time_val = center_time_val + (max_time_window / 2)
        closest_lower_time_val = file_data[0].flat[
            np.abs(file_data[0] - lower_time_val).argmin()
        ]
        lower_time_pos = file_data[0].tolist().index(closest_lower_time_val)
        closest_upper_time_val = file_data[0].flat[
            np.abs(file_data[0] - upper_time_val).argmin()
        ]
        upper_time_pos = file_data[0].tolist().index(closest_upper_time_val)
        target_time_scale = np.asarray(file_data[0][lower_time_pos:upper_time_pos])
        target_val_scale = np.asarray(file_data[1][lower_time_pos:upper_time_pos])
        self.get_peak_distance(
            use_fft=use_fft,
            plot_data=plot_data,
            internal_file_close_data=[target_time_scale, target_val_scale],
        )

    def get_peak_distance_from_path_distance(
        self, path_difference: float, files: tuple[str, str], plot_data: bool
    ) -> float:
        """get_peak_distance_from_path_distance _summary_

        Args:
            path_difference (_type_): _description_
            files (_type_): _description_
            plot_data (_type_): _description_

        Returns:
            float: _description_
        """
        file_data_I = self.get_raw_autocorrelation_data(files[0])
        file_data_II = self.get_raw_autocorrelation_data(files[1])

        if plot_data:
            plt.plot(file_data_I[0], file_data_I[1], file_data_II[0], file_data_II[1])
            plt.show()
            plt.clf()
        # find maximum position
        max_time_I = float(
            file_data_I[0][file_data_I[1].tolist().index(np.max(file_data_I[1]))]
        )
        max_time_II = float(
            file_data_II[0][file_data_II[1].tolist().index(np.max(file_data_II[1]))]
        )
        if max_time_I > max_time_II:
            self.file_wide = files[0]
            self.time_correction_factor_wide = max_time_I
            self.file_close = files[1]
            self.time_correction_factor_close = max_time_II
        else:
            self.file_wide = files[1]
            self.time_correction_factor_wide = max_time_II
            self.file_close = files[0]
            self.time_correction_factor_close = max_time_I
        return (2 * path_difference / scco.speed_of_light) / np.abs(
            max_time_II - max_time_I
        )

    def get_peak_distance(
        self,
        f_threshold: float = 1000,
        use_fft: bool = True,
        plot_data: bool = True,
        error_threshold: float = 0.1,
        internal_file_close_data: list = [],
        print_peak_data: bool = False,
    ) -> None:
        """get_peak_distance _summary_

        Args:
            f_threshold (float, optional): _description_. Defaults to 1000.
            use_fft (bool, optional): _description_. Defaults to True.
            plot_data (bool, optional): _description_. Defaults to True.
            error_threshold (float, optional): _description_. Defaults to 0.1.
            internal_file_close_data (list, optional): _description_. Defaults to [].
            print_peak_data (bool, optional): _description_. Defaults to False.

        Raises:
            Exception: _description_
        """
        # pylint:disable=dangerous-default-value
        if len(internal_file_close_data) < 2:
            time, data = self.get_raw_autocorrelation_data(self.file_close)
        else:
            time = internal_file_close_data[0]
            data = internal_file_close_data[1]
        smoothened_data = savgol_filter(data, 95, 3)
        fft_example_data = np.fft.fft(smoothened_data)
        frequencies = np.fft.fftfreq(len(smoothened_data), np.abs(time[1] - time[0]))
        positive_frequencies = np.where(frequencies > f_threshold, frequencies, 0)
        positive_fft_values = np.where(frequencies > f_threshold, fft_example_data, 0)
        positive_max_frequency = positive_frequencies[
            np.where(np.abs(positive_fft_values) == np.max(np.abs(positive_fft_values)))
        ][0]

        peaks, _ = find_peaks(smoothened_data, prominence=(0.001, 10))
        distance_peaks = time[peaks]
        aver_distance = 0
        for i in range(len(distance_peaks) - 1):
            aver_distance += distance_peaks[i + 1] - distance_peaks[i]
        aver_distance /= len(distance_peaks)
        if print_peak_data:
            print(f"Aver_distance: {aver_distance}")
            print(f"F_dist: {1./positive_max_frequency}")
        if plot_data:
            plt.plot(data)
            plt.plot(smoothened_data)
            plt.plot(peaks, smoothened_data[peaks], "x")
            plt.show()
            plt.clf()
        if (
            np.abs(aver_distance / (1.0 / positive_max_frequency)) < error_threshold
            or np.abs(aver_distance / (1.0 / positive_max_frequency))
            > (1.0 / error_threshold)
        ) and use_fft:
            raise Exception(
                "FFT-data and aver_distance-data do not correlate close enough. Exiting code"
            )
        if use_fft:
            self.peak_distance = 1 / positive_max_frequency
        else:
            self.peak_distance = aver_distance

    def butter_lowpass_filter(
        self, data, cutoff, sampling_frequency, order
    ) -> np.ndarray:
        """butter_lowpass_filter _summary_

        Args:
            data (_type_): _description_
            cutoff (_type_): _description_
            sampling_frequency (_type_): _description_
            order (_type_): _description_

        Returns:
            np.ndarray: _description_
        """
        normal_cutoff = cutoff / (sampling_frequency * 0.5)
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    def gaussian_function_compact(
        self, x: Union[np.ndarray, float], data: list[float]
    ) -> float:
        """gaussian_function_compact _summary_

        Args:
            x (_type_): _description_
            data (_type_): _description_

        Returns:
            float: _description_
        """
        return data[3] + data[0] * np.sqrt(2 / np.pi) / data[1] * np.exp(
            -4 * np.log(2) * ((x - data[2]) / data[1]) ** 2
        )

        # return data[3] + data[0] * np.exp(-0.5 * np.power((x - data[2]) / data[1], 2))

    def sech_function_compact(self, x, data) -> float:
        """sech_function_compact _summary_

        Args:
            x (_type_): _description_
            data (_type_): _description_

        Returns:
            float: _description_
        """
        y_val = (x - data[2]) / data[1]
        return data[3] + data[0] * 3 / (np.power((np.sinh(y_val) + 1e-13), 2)) * (
            y_val * np.cosh(y_val) / (np.sinh(y_val) + 1e-13) - 1
        )

    def gaussian_function(self, x, A, xc, w, y0) -> float:
        """gaussian_function _summary_

        Args:
            x (_type_): _description_
            A (_type_): _description_
            xc (_type_): _description_
            w (_type_): _description_
            y0 (_type_): _description_

        Returns:
            float: _description_
        """
        return self.gaussian_function_compact(x, [A, xc, w, y0])

    def sech_function(self, x, A, xc, w, y0) -> float:
        """sech_function _summary_

        Args:
            x (_type_): _description_
            A (_type_): _description_
            xc (_type_): _description_
            w (_type_): _description_
            y0 (_type_): _description_

        Returns:
            float: _description_
        """
        return self.sech_function_compact(x, [A, xc, w, y0])

    def get_raw_autocorrelation_data(
        self, filename: str, skip_lines: int = 15
    ) -> tuple[np.ndarray, np.ndarray]:
        """get_raw_autocorrelation_data _summary_

        Args:
            filename (str): _description_
            skip_lines (int, optional): _description_. Defaults to 15.

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        time_data = []
        meas_data = []
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line_number, line in enumerate(lines):
                if line_number > skip_lines and len(line.split(",")) == 3:
                    line_data = line.split(",")
                    time_data.append(float(line_data[0]))
                    meas_data.append(float(line_data[1]))
        if self.use_normalized_data:
            meas_data = np.asarray(meas_data)
            meas_data -= np.min(meas_data)
            meas_data /= np.max(meas_data)
        return np.asarray(time_data), np.asarray(meas_data)

    def butter_filter_autocorrelation_data(
        self, data: tuple[np.ndarray, np.ndarray], filter_frequency: float = 1000
    ) -> tuple[np.ndarray, np.ndarray]:
        """butter_filter_autocorrelation_data _summary_

        Args:
            data (tuple[np.ndarray, np.ndarray]): _description_
            filter_frequency (float, optional): _description_. Defaults to 1000.

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        sampling_frequency = 1.0 / (np.abs(data[0][1] - data[0][0]))
        if filter_frequency > 0:
            data[1][:] = self.butter_lowpass_filter(
                data[1][:], filter_frequency, sampling_frequency, 2
            )
        return data

    def fft_filter_autocorrelation_data(
        self, data: tuple[np.ndarray, np.ndarray], filter_frequency: float = 1000
    ) -> tuple[np.ndarray, np.ndarray]:
        """fft_filter_autocorrelation_data _summary_

        Args:
            data (tuple[np.ndarray, np.ndarray]): _description_
            filter_frequency (float, optional): _description_. Defaults to 1000.

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        w = fftpack.fftfreq(data[1].size, np.abs(data[0][1] - data[0][0]))
        f_signal = fftpack.fft(data[1])
        cut_f_signal = f_signal.copy()
        cut_f_signal[np.abs(w) > filter_frequency] = 0
        data[1][:] = np.abs(fftpack.ifft(cut_f_signal))
        return data

    def get_autocorrelation_data(
        self,
        estimated_pulse_duration: float = 1,
        include_gaussian: bool = True,
        include_sech: bool = True,
        filter_frequency: float = 1000,
        print_results: bool = True,
        use_fft_filter: bool = False,
        use_significant_digits: int = 3,
    ) -> None:
        """get_autocorrelation_data _summary_

        Args:
            estimated_pulse_duration (float, optional): _description_. Defaults to 1.
            include_gaussian (bool, optional): _description_. Defaults to True.
            include_sech (bool, optional): _description_. Defaults to True.
            filter_frequency (float, optional): _description_. Defaults to 1000.
            print_results (bool, optional): _description_. Defaults to True.
            use_fft_filter (bool, optional): _description_. Defaults to False.
            use_significant_digits (int, optional): _description_. Defaults to 3.
        """
        time_data, meas_data = self.get_raw_autocorrelation_data(
            self.file_wide, skip_lines=15
        )
        self.filter_frequency = filter_frequency
        self.raw_time_data = time_data.copy()
        self.raw_time_data = self.raw_time_data - self.time_correction_factor_wide
        time_data = time_data - self.time_correction_factor_wide
        self.raw_meas_data = meas_data.copy()
        if use_fft_filter:
            (filter_time_data, filter_meas_data) = self.fft_filter_autocorrelation_data(
                (time_data[:], meas_data[:]), filter_frequency=filter_frequency
            )
        else:
            (
                filter_time_data,
                filter_meas_data,
            ) = self.butter_filter_autocorrelation_data(
                (time_data[:], meas_data[:]), filter_frequency=filter_frequency
            )

        if print_results:
            print(f"Prefactor: {self.prefactor}")
        self.time_data = filter_time_data * self.prefactor
        self.raw_time_data = self.raw_time_data * self.prefactor
        self.meas_data = filter_meas_data

        self.gaussian_used = include_gaussian
        self.sech_used = include_sech
        if include_gaussian:
            self.apply_gaussian_fit(estimated_pulse_duration)
            if print_results:
                print_string = "Pulse FWHM for gaussian pulse:"
                print_string += (
                    f" {np.abs(self.gaussian_pulse_fwhm):0.{use_significant_digits}} ps"
                )
                print(print_string)
        if include_sech:
            self.apply_sech_fit(estimated_pulse_duration)
            if print_results:
                print_string = "Pulse FWHM for sech pulse: "
                print_string += (
                    f"{np.abs(self.sech_pulse_fwhm):0.{use_significant_digits}} ps"
                )
                print(print_string)

    def apply_gaussian_fit(self, estimated_pulse_duration: float = 1.0) -> None:
        """apply_gaussian_fit _summary_

        Args:
            estimated_pulse_duration (float, optional): _description_. Defaults to 1.0.
        """
        # pylint:disable=unbalanced-tuple-unpacking
        popt, _ = curve_fit(
            self.gaussian_function,
            1e12 * self.time_data,
            self.meas_data,
            p0=[np.max(self.meas_data), estimated_pulse_duration, 0, self.meas_data[0]],
        )
        self.gaussian_pulse_fwhm = (popt[1]) * np.sqrt(
            0.5
        )  # * 2 * np.sqrt(2 * np.log(2))
        self.gaussian_fwhm = popt[1]  # * 2 * np.sqrt(2 * np.log(2))
        self.gaussian_data = popt
        self.gaussian_fit_calculated = True

    def apply_sech_fit(self, estimated_pulse_duration: float = 1.0) -> None:
        """apply_sech_fit _summary_

        Args:
            estimated_pulse_duration (int, optional): _description_. Defaults to 1.
        """
        # pylint:disable=unbalanced-tuple-unpacking
        offset = 0
        if self.gaussian_fit_calculated:
            offset = self.gaussian_data[2]
        popt, _ = curve_fit(
            self.sech_function,
            1e12 * self.time_data,
            self.meas_data,
            p0=[
                np.max(self.meas_data),
                estimated_pulse_duration,
                offset,
                self.meas_data[0],
            ],
        )
        self.sech_pulse_fwhm = (popt[1]) * 2.7196 * 0.6482
        self.sech_fwhm = (popt[1]) * 2.7196
        self.sech_data = popt
        self.sech_fit_calculated = True
        # print(self.sech_data, np.max(self.meas_data), self.meas_data[0])

    def plot_figure(
        self,
        legend_location: str = "upper center",
        file_title: str = "",
        x_min: float = 1,
        x_max: float = -1,
        use_significant_digits: int = 3,
    ) -> None:
        """plot_figure _summary_

        Args:
            legend_location (str, optional): _description_. Defaults to "upper center".
            file_title (str, optional): _description_. Defaults to "".
            x_min (float, optional): _description_. Defaults to 1.
            x_max (float, optional): _description_. Defaults to -1.
            use_significant_digits (int, optional): _description_. Defaults to 3.
        """
        fig_ax = plt.subplot(111)

        fig_ax.plot(
            self.raw_time_data * 1e15,
            self.raw_meas_data,
            self.time_data * 1e15,
            self.meas_data,
        )
        if self.gaussian_used:
            fig_ax.plot(
                self.time_data * 1e15,
                self.gaussian_function_compact(
                    self.time_data * 1e12, self.gaussian_data
                ),
            )
        if self.sech_used:
            fig_ax.plot(
                self.time_data * 1e15,
                self.sech_function_compact(self.time_data * 1e12, self.sech_data),
            )
        legend_entries = []
        legend_entries.append("Raw data")
        legend_entries.append(
            f"Filtered data with Lowpass L(f < {np.abs(self.filter_frequency) / 1000:3.2} kHz)"
        )
        if self.gaussian_used:
            cur_legend_entry = ""
            cur_legend_entry += "Fitted gaussian with AC FWHM = "
            cur_legend_entry += f"{np.abs(self.gaussian_fwhm):.2}"
            cur_legend_entry += r" ps\nand pulse FWHM = "
            cur_legend_entry += (
                f"{np.abs(self.gaussian_pulse_fwhm):.{use_significant_digits}} ps"
            )
            legend_entries.append(cur_legend_entry)
        if self.sech_used:
            cur_legend_entry = ""
            cur_legend_entry += "Fitted sech with AC FWHM = "
            cur_legend_entry += f"{np.abs(self.sech_fwhm):.2}"
            cur_legend_entry += r" ps\nand pulse FWHM = "
            cur_legend_entry += (
                f"{np.abs(self.sech_pulse_fwhm):.{use_significant_digits}} ps"
            )
            legend_entries.append(cur_legend_entry)
        fig_ax.legend(legend_entries, loc=legend_location)
        plt.ylabel("Intensity")
        plt.xlabel("Delay [fs]")
        if file_title:
            plt.title(file_title)
        if x_min < 0 and x_max > 0:
            plt.xlim((x_min, x_max))
        else:
            x_max = self.time_data[-1]
            x_min = self.time_data[0]
            x_border = np.min([np.abs(x_max), np.abs(x_min)])
            plt.xlim((-1 * x_border * 1e15, x_border * 1e15))
        plt.show()
        plt.clf()

    def save_figure(
        self,
        legend_location: str = "upper center",
        file_format: str = "PNG",
        file_name: str = "",
        file_title: str = "",
        x_min: float = 1,
        x_max: float = -1,
        use_significant_digits: int = 3,
        reduce_to_every_nth_entry: int = 1,
    ) -> None:
        """save_figure _summary_

        Args:
            legend_location (str, optional): _description_. Defaults to "upper center".
            file_format (str, optional): _description_. Defaults to "PNG".
            file_name (str, optional): _description_. Defaults to "".
            file_title (str, optional): _description_. Defaults to "".
            x_min (float, optional): _description_. Defaults to 1.
            x_max (float, optional): _description_. Defaults to -1.
            use_significant_digits (int, optional): _description_. Defaults to 3.
            reduce_to_every_nth_entry (int, optional): _description_. Defaults to 1.
        """
        fig_ax = plt.subplot(111)

        if reduce_to_every_nth_entry > 1:
            fig_ax.plot(
                self.raw_time_data[0::reduce_to_every_nth_entry] * 1e15,
                self.raw_meas_data[0::reduce_to_every_nth_entry],
                self.time_data[0::reduce_to_every_nth_entry] * 1e15,
                self.meas_data[0::reduce_to_every_nth_entry],
            )
            if self.gaussian_used:
                fig_ax.plot(
                    self.time_data * 1e15,
                    self.gaussian_function_compact(
                        self.time_data * 1e12, self.gaussian_data
                    ),
                )
            if self.sech_used:
                fig_ax.plot(
                    self.time_data * 1e15,
                    self.sech_function_compact(self.time_data * 1e12, self.sech_data),
                )
        else:
            fig_ax.plot(
                self.raw_time_data * 1e15,
                self.raw_meas_data,
                self.time_data * 1e15,
                self.meas_data,
            )
            if self.gaussian_used:
                fig_ax.plot(
                    self.time_data * 1e15,
                    self.gaussian_function_compact(
                        self.time_data * 1e12, self.gaussian_data
                    ),
                )
            if self.sech_used:
                fig_ax.plot(
                    self.time_data * 1e15,
                    self.sech_function_compact(self.time_data * 1e12, self.sech_data),
                )
        legend_entries = []
        legend_entries.append("Raw data")
        legend_entries.append(
            "Filtered data with Lowpass L(f < "
            + "{0:3.2f}".format(np.abs(self.filter_frequency) / 1000)
            + " kHz)"
        )
        if self.gaussian_used:
            legend_entries.append(
                "Fitted gaussian with AC FWHM = "
                + "{0:.2f}".format(np.abs(self.gaussian_fwhm))
                + " ps\nand pulse FWHM = "
                + "{0:.{prec}f}".format(
                    np.abs(self.gaussian_pulse_fwhm), prec=use_significant_digits
                )
                + " ps"
            )
        if self.sech_used:
            legend_entries.append(
                "Fitted sech with AC FWHM = "
                + "{0:.2f}".format(np.abs(self.sech_fwhm))
                + " ps\nand pulse FWHM = "
                + "{0:.{prec}f}".format(
                    np.abs(self.sech_pulse_fwhm), prec=use_significant_digits
                )
                + " ps"
            )
        fig_ax.legend(legend_entries, loc=legend_location)
        plt.ylabel("Intensity")
        plt.xlabel("Delay [fs]")
        if file_title:
            plt.title(file_title)
        if x_min < 0 and x_max > 0:
            plt.xlim((x_min, x_max))
        else:
            x_max = self.time_data[-1]
            x_min = self.time_data[0]
            x_border = np.min([np.abs(x_max), np.abs(x_min)])
            plt.xlim((-1 * x_border * 1e15, x_border * 1e15))
        # ax.legend(loc = legend_location)
        # plt.show()
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
        plt.close()

        # plot_file_name = file_name
        # if file_format == 0:
        #     if plot_file_name[-4:] != ".png":
        #         plot_file_name += ".png"
        #     plt.savefig(plot_file_name, dpi=600, metadata={'Source': [self.file_wide],
        #     'Title': plot_file_name, 'Author': "Roland Richter"})
        #     plt.clf()
        # else:
        #     if plot_file_name[-5:] != ".tikz":
        #         plot_file_name += ".tikz"
        #     tikzplotlib.clean_figure(target_resolution=150)
        #     tikzplotlib.save(plot_file_name)
