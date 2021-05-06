#!/usr/bin/env python3

import h5py
import numpy as np
import numpy.fft as nfft
import scipy.constants as scco
from scipy.signal import butter, lfilter, freqz, filtfilt, find_peaks, savgol_filter
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate, fftpack
import tikzplotlib

import math
from string import Template
import subprocess

import types
import functools

from enum import Enum

class Reference_Method(Enum):
    USE_RIPPLES = 1
    USE_DISTANCE = 2

class autocorrelator:
    def __init__(self, file_wide = "", file_close = "", central_wavelength = 2000, files = [], peak_distance = -1, use_FFT_for_peak=True, plot_distance_data = False, path_difference = 1e-3, reference = Reference_Method.USE_RIPPLES, print_peak_data = False):
        self.time_correction_factor_wide = 0
        self.time_correction_factor_close = 0
        #I assume we are just using wavelengths between 1e-9 and 100e-6
        if central_wavelength > 100e-6:
            #Central wavelength is given in nm
            self.central_wavelength = central_wavelength * 1e-9
        else:
            #Everything is fine
            self.central_wavelength = central_wavelength
        if reference is Reference_Method.USE_RIPPLES:
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
                    self.get_peak_distance(use_FFT=use_FFT_for_peak, plot_data=plot_distance_data, print_peak_data=print_peak_data)
                else:
                    self.get_peak_distance_from_single_file(single_data_file = files[0], use_FFT = use_FFT_for_peak, plot_data = plot_distance_data)
            else:
                self.peak_distance = peak_distance
            self.prefactor = 1 / self.peak_distance * (self.central_wavelength) / scco.speed_of_light
        else:
            if file_wide and file_close:
                self.peak_distance = self.get_peak_distance_from_path_distance(path_difference, [file_wide, file_close], plot_data=plot_distance_data)
            else:
                self.peak_distance = self.get_peak_distance_from_path_distance(path_difference, files, plot_data=plot_distance_data)
            self.prefactor = self.peak_distance
        if print_peak_data:
            print("Peak distance:", self.peak_distance)

    def get_file_destination(self, files):
        if len(files) != 2:
            print("Not enough or too many files supplied. Please submit only two files. Exiting")
            exit()
        file_1 = files[0]
        file_2 = files[1]
        time_scale_1 = -1
        time_scale_2 = -1
        with open(file_1, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Horizontal Scale" in line:
                    line_data = line.split(',')
                    time_scale_1 = float(line_data[1])
                    break

        with open(file_2, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Horizontal Scale" in line:
                    line_data = line.split(',')
                    time_scale_2 = float(line_data[1])
                    break
        
        if time_scale_1 < 0 or time_scale_2 < 0:
            print("Error reading files, please proceed with separate files. No time scale could be found.")
            exit()
        if time_scale_1 > time_scale_2:
            self.file_wide = file_1
            self.file_close = file_2
        else:
            self.file_close = file_1
            self.file_wide = file_2
        print("File_close:", self.file_close, ", File_wide:", self.file_wide)

    def get_peak_distance_from_single_file(self, single_data_file, use_FFT = True, plot_data = True, max_time_window = 1e-3):
        file_data = self.get_raw_autocorrelation_data(single_data_file)
        #find center of spectrum/time
        center_time_val = file_data[0].flat[np.abs(file_data[0]).argmin()]
        center_time_pos = file_data[0].tolist().index(center_time_val)
        lower_time_val = center_time_val - (max_time_window / 2)
        upper_time_val = center_time_val + (max_time_window / 2)
        closest_lower_time_val = file_data[0].flat[np.abs(file_data[0] - lower_time_val).argmin()]
        lower_time_pos = file_data[0].tolist().index(closest_lower_time_val)
        closest_upper_time_val = file_data[0].flat[np.abs(file_data[0] - upper_time_val).argmin()]
        upper_time_pos = file_data[0].tolist().index(closest_upper_time_val)
        target_time_scale = np.asarray(file_data[0][lower_time_pos:upper_time_pos])
        target_val_scale = np.asarray(file_data[1][lower_time_pos:upper_time_pos])
        self.get_peak_distance(use_FFT = use_FFT, plot_data = plot_data, internal_file_close_data = [target_time_scale, target_val_scale])

    def get_peak_distance_from_path_distance(self, path_difference, files, plot_data):
        file_data_I = self.get_raw_autocorrelation_data(files[0])
        file_data_II = self.get_raw_autocorrelation_data(files[1])

        if plot_data:
            plt.plot(file_data_I[0], file_data_I[1], file_data_II[0], file_data_II[1])
            plt.show()
            plt.clf()
        #find maximum position
        max_time_I = file_data_I[0][file_data_I[1].tolist().index(np.max(file_data_I[1]))]
        max_time_II = file_data_II[0][file_data_II[1].tolist().index(np.max(file_data_II[1]))]
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
        return (2 * path_difference / scco.speed_of_light) / np.abs(max_time_II - max_time_I)


    def get_peak_distance(self, f_threshold = 1000, use_FFT = True, plot_data = True, error_threshold = 0.1, internal_file_close_data = [], print_peak_data = False):
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
        positive_max_frequency = positive_frequencies[np.where(np.abs(positive_fft_values) == np.max(np.abs(positive_fft_values)))][0]

        peaks, _ = find_peaks(smoothened_data, prominence=(0.001, 10))
        distance_peaks = time[peaks]
        aver_distance = 0
        for i in range(len(distance_peaks) - 1):
            aver_distance += distance_peaks[i + 1] - distance_peaks[i]
        aver_distance /= len(distance_peaks)
        if print_peak_data:
            print("Aver_distance:", aver_distance)
            print("F_dist:", 1 / positive_max_frequency)
        if plot_data:
            plt.plot(data)
            plt.plot(smoothened_data)
            plt.plot(peaks, smoothened_data[peaks], "x")
            plt.show()
            plt.clf()
        if (np.abs(aver_distance / (1. / positive_max_frequency)) < error_threshold or np.abs(aver_distance / (1. / positive_max_frequency)) > (1. / error_threshold)) and use_FFT:
            print("FFT-data and aver_distance-data do not correlate close enough. Exiting code")
            exit()
        if use_FFT:
            self.peak_distance = 1 / positive_max_frequency
        else:
            self.peak_distance = aver_distance

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        normal_cutoff = cutoff / (fs * 0.5)
        b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
        return filtfilt(b, a, data)

    def gaussian_function_compact(self, x, data):
        y_val = (x - data[2]) / data[1]
        return data[3] + data[0]*np.sqrt(2/np.pi)/data[1]*np.exp(-4 * np.log(2) *((x-data[2])/data[1])**2)
        
        #return data[3] + data[0] * np.exp(-0.5 * np.power((x - data[2]) / data[1], 2))

    def sech_function_compact(self, x, data):
        #return data[3] + data[0] * 1. / np.cosh((x - data[2]) / (data[1]))
        y_val = (x - data[2]) / data[1]
        return data[3] + data[0] * 3 / (np.power(np.sinh(y_val), 2)) * (y_val * np.cosh(y_val) / np.sinh(y_val) - 1) 

    def gaussian_function(self, x, A, xc, w, y0):
        return self.gaussian_function_compact(x, [A, xc, w, y0])

    def sech_function(self, x, A, xc, w, y0):
        return self.sech_function_compact(x, [A, xc, w, y0])

    def get_raw_autocorrelation_data(self, filename, skip_lines = 15):
        time_data = []
        meas_data = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line_number, line in enumerate(lines):
                if line_number > skip_lines and len(line.split(',')) == 3:
                    line_data = line.split(',')
                    time_data.append(float(line_data[0]))
                    meas_data.append(float(line_data[1]))
        return np.asarray(time_data), np.asarray(meas_data)

    def butter_filter_autocorrelation_data(self, data, filter_frequency = 1000):
        fs = 1. / (np.abs(data[0][1] - data[0][0]))
        if filter_frequency > 0:
            data[1][:] = self.butter_lowpass_filter(data[1][:], filter_frequency, fs, 2)
        return data

    def fft_filter_autocorrelation_data(self, data, filter_frequency = 1000):
        fs = 1. / (np.abs(data[0][1] - data[0][0]))
        w = fftpack.fftfreq(data[1].size, np.abs(data[0][1] - data[0][0]))
        #f_signal = np.fft.rfft(data[1])
        f_signal = fftpack.fft(data[1])
        cut_f_signal = f_signal.copy()
        #cut_f_signal[(w[:cut_f_signal.size] > filter_frequency)] = 0
        cut_f_signal[np.abs(w) > filter_frequency] = 0
        data[1][:] = np.abs(fftpack.ifft(cut_f_signal))
        return data

    def get_autocorrelation_data(self, estimated_pulse_duration = 1, include_gaussian = True, include_sech = True, filter_frequency = 1000, print_results = True, use_FFT_filter = False, use_significant_digits = 3):
        time_data, meas_data = self.get_raw_autocorrelation_data(self.file_wide, skip_lines = 15)
        self.filter_frequency = filter_frequency
        self.raw_time_data = time_data.copy()
        self.raw_time_data = self.raw_time_data - self.time_correction_factor_wide
        time_data = time_data - self.time_correction_factor_wide
        self.raw_meas_data = meas_data.copy()
        if use_FFT_filter:
            (filter_time_data, filter_meas_data) = self.fft_filter_autocorrelation_data((time_data[:], meas_data[:]), filter_frequency = filter_frequency)
        else:
            (filter_time_data, filter_meas_data) = self.butter_filter_autocorrelation_data((time_data[:], meas_data[:]), filter_frequency=filter_frequency)

        if print_results:
            print("Prefactor:", self.prefactor)
        self.time_data = filter_time_data * self.prefactor
        self.raw_time_data = self.raw_time_data * self.prefactor
        self.meas_data = filter_meas_data

        self.gaussian_used = include_gaussian
        self.sech_used = include_sech
        if include_gaussian:
            self.apply_gaussian_fit(estimated_pulse_duration)
            if print_results:
                print("Pulse FWHM for gaussian pulse:", '{0:.{prec}f}'.format(np.abs(self.gaussian_pulse_fwhm), prec=use_significant_digits), "ps")
        if include_sech:
            self.apply_sech_fit(estimated_pulse_duration)
            if print_results:
                print("Pulse FWHM for sech pulse:", '{0:.{prec}f}'.format(np.abs(self.sech_pulse_fwhm), prec=use_significant_digits), "ps")

    def apply_gaussian_fit(self, estimated_pulse_duration = 1):
        popt, pcopt = curve_fit(self.gaussian_function, 1e12 * self.time_data, self.meas_data, p0 = [np.max(self.meas_data), estimated_pulse_duration, 0, self.meas_data[0]])
        self.gaussian_pulse_fwhm = (popt[1]) * np.sqrt(0.5)# * 2 * np.sqrt(2 * np.log(2))
        self.gaussian_fwhm = (popt[1])# * 2 * np.sqrt(2 * np.log(2))
        self.gaussian_data = popt

    def apply_sech_fit(self, estimated_pulse_duration = 1):
        popt, pcopt = curve_fit(self.sech_function, 1e12 * self.time_data, self.meas_data, p0 = [np.max(self.meas_data), estimated_pulse_duration, 0, self.meas_data[0]])
        self.sech_pulse_fwhm = (popt[1]) * 2.7196 * 0.6482
        self.sech_fwhm = (popt[1]) * 2.7196
        self.sech_data = popt 

    def plot_figure(self, legend_location = 'upper center', file_title = "", x_min = 1, x_max = -1, use_significant_digits = 3):
        #plt.title("Autocorrelation for I = " + str(power_elem / 10) + " A")
        fig = plt.figure()
        ax = plt.subplot(111)
        
        ax.plot(self.raw_time_data * 1e15, self.raw_meas_data, self.time_data * 1e15, self.meas_data)
        if self.gaussian_used:
            ax.plot(self.time_data * 1e15, self.gaussian_function_compact(self.time_data * 1e12, self.gaussian_data))
        if self.sech_used:
            ax.plot(self.time_data * 1e15, self.sech_function_compact(self.time_data * 1e12, self.sech_data))
        Legend_entries = []
        Legend_entries.append('Raw data')
        Legend_entries.append('Filtered data with Lowpass L(f < ' + '{0:3.2f}'.format(np.abs(self.filter_frequency) / 1000) + ' kHz)')
        if self.gaussian_used:
            Legend_entries.append('Fitted gaussian with AC FWHM = ' + '{0:.2f}'.format(np.abs(self.gaussian_fwhm)) + ' ps\nand pulse FWHM = ' + '{0:.{prec}f}'.format(np.abs(self.gaussian_pulse_fwhm), prec=use_significant_digits) + " ps")
        if self.sech_used:
            Legend_entries.append('Fitted sech with AC FWHM = ' + '{0:.2f}'.format(np.abs(self.sech_fwhm)) + ' ps\nand pulse FWHM = ' + '{0:.{prec}f}'.format(np.abs(self.sech_pulse_fwhm), prec=use_significant_digits) + " ps")
        ax.legend(Legend_entries, loc=legend_location)
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
        #ax.legend(loc = legend_location)
        plt.show()
        plt.clf()

    def save_figure(self, legend_location = 'upper center', file_format = "PNG", file_name = "", file_title = "", x_min = 1, x_max = -1, use_significant_digits = 3, reduce_to_every_nth_entry = 1):
        #plt.title("Autocorrelation for I = " + str(power_elem / 10) + " A")
        fig = plt.figure()
        ax = plt.subplot(111)
        
        if reduce_to_every_nth_entry > 1:
            ax.plot(self.raw_time_data[0::reduce_to_every_nth_entry] * 1e15, self.raw_meas_data[0::reduce_to_every_nth_entry], self.time_data[0::reduce_to_every_nth_entry] * 1e15, self.meas_data[0::reduce_to_every_nth_entry])
            if self.gaussian_used:
                ax.plot(self.time_data * 1e15, self.gaussian_function_compact(self.time_data * 1e12, self.gaussian_data))
            if self.sech_used:
                ax.plot(self.time_data * 1e15, self.sech_function_compact(self.time_data * 1e12, self.sech_data))
        else:
            ax.plot(self.raw_time_data * 1e15, self.raw_meas_data, self.time_data * 1e15, self.meas_data)
            if self.gaussian_used:
                ax.plot(self.time_data * 1e15, self.gaussian_function_compact(self.time_data * 1e12, self.gaussian_data))
            if self.sech_used:
                ax.plot(self.time_data * 1e15, self.sech_function_compact(self.time_data * 1e12, self.sech_data))
        Legend_entries = []
        Legend_entries.append('Raw data')
        Legend_entries.append('Filtered data with Lowpass L(f < ' + '{0:3.2f}'.format(np.abs(self.filter_frequency) / 1000) + ' kHz)')
        if self.gaussian_used:
            Legend_entries.append('Fitted gaussian with AC FWHM = ' + '{0:.2f}'.format(np.abs(self.gaussian_fwhm)) + ' ps\nand pulse FWHM = ' + '{0:.{prec}f}'.format(np.abs(self.gaussian_pulse_fwhm), prec=use_significant_digits) + " ps")
        if self.sech_used:
            Legend_entries.append('Fitted sech with AC FWHM = ' + '{0:.2f}'.format(np.abs(self.sech_fwhm)) + ' ps\nand pulse FWHM = ' + '{0:.{prec}f}'.format(np.abs(self.sech_pulse_fwhm), prec=use_significant_digits) + " ps")
        ax.legend(Legend_entries, loc=legend_location)
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
        #ax.legend(loc = legend_location)
        #plt.show()
        local_file_name = file_name
        if file_format == "PNG":
            if local_file_name[-4:] != ".png":
                local_file_name += ".png"
            plt.savefig(local_file_name, bbox_inches="tight", dpi=600, metadata={'Source': [self.file_wide], 'Title': plot_file_name, 'Author': "Roland Richter"})
        else:
            if file_format == "PGF":
                if local_file_name[-4:] != ".pgf":
                    local_file_name += ".pgf"
                plt.savefig(local_file_name, bbox_inches="tight", dpi=600)
            else:
                if local_file_name[-5:] != ".tikz":
                    local_file_name += ".tikz"
                tikzplotlib.clean_figure(target_resolution=300)
                tikzplotlib.save(local_file_name)
        """
        plot_file_name = file_name
        if file_format == 0:
            if plot_file_name[-4:] != ".png":
                plot_file_name += ".png"
            plt.savefig(plot_file_name, dpi=600, metadata={'Source': [self.file_wide], 'Title': plot_file_name, 'Author': "Roland Richter"})
            plt.clf()
        else:
            if plot_file_name[-5:] != ".tikz":
                plot_file_name += ".tikz"
            tikzplotlib.clean_figure(target_resolution=150)
            tikzplotlib.save(plot_file_name)
        """
