#!/usr/bin/env python3

import h5py
import numpy as np
import scipy.constants as scco
from scipy.signal import butter, lfilter, freqz, filtfilt, find_peaks, savgol_filter
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
import tikzplotlib

import math
from string import Template
import subprocess

import types
import functools

from enum import Enum

from itertools import islice

from filter_mirrors import *

def nth_key(dct, n):
    it = iter(dct)
    next(islice(it, n, n), None)
    return next(it)

class OSA_spectrum:
    def __init__(self, wl_data, linear_meas_data, log_meas_data, spectrum_name):
        self.wl_data = wl_data
        self.linear_meas_data = linear_meas_data
        self.log_meas_data = log_meas_data
        self.spectrum_name = spectrum_name
        self.fit_function = "None"
        self.fit_meas_data = []
        self.fit_data = []
        self.FWHM = 0
        self.bandwidth = 0
        self.fit_can_be_used = False


class OSA:
    def __init__(self):
        self.spectra = {}
        return None

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        normal_cutoff = cutoff / (fs * 0.5)
        b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
        return filtfilt(b, a, data)

    def gaussian_function_compact(self, x, data):
        return data[3] + data[0]*np.sqrt(2/np.pi)/data[1]*np.exp(-4 * np.log(2) *((x-data[2])/data[1])**2)
        #return data[3] + data[0]*np.exp(-2*((x-data[2])/data[1])**2)

    def gaussian_function(self, x, A, xc, w, y0):
        return self.gaussian_function_compact(x, [A, xc, w, y0])

    def sech_function_compact(self, x, data):
        y_val = (x - data[2]) / data[1]
        return data[3] + data[0]/np.cosh(y_val)
        #return data[3] + data[0]*np.exp(-2*((x-data[2])/data[1])**2)

    def sech_function(self, x, A, xc, w, y0):
        return self.sech_function_compact(x, [A, xc, w, y0])

    def calc_spectrum_energy(self, spectrum_number):
        spectrum_energies = {}
        if spectrum_number < 0 or spectrum_number > (len(self.spectra) - 1):
            for i in range(len(self.spectra)):
                local_lin_meas_data = self.spectra[[*self.spectra][i]].linear_meas_data
                local_wl_data = self.spectra[[*self.spectra][i]].wl_data
                cur_energy = 0
                full_energy = 0
                for j, elem in enumerate(local_lin_meas_data):
                    cur_energy = elem
                    if j == 0 or j == len(local_lin_meas_data) - 1:
                        cur_energy *= 0.5
                    full_energy += cur_energy
                full_energy *= 1e-9
                spectrum_energies[self.spectra[[*self.spectra][i]].spectrum_name] = full_energy
        else:
            local_lin_meas_data = self.spectra[[*self.spectra][i]].linear_meas_data
            local_wl_data = self.spectra[[*self.spectra][i]].wl_data
            cur_energy = 0
            full_energy = 0
            for j, elem in enumerate(local_lin_meas_data):
                cur_energy = elem
                if j == 0 or j == len(local_lin_meas_data) - 1:
                    cur_energy *= 0.5
                full_energy += cur_energy
            full_energy *= 1e-9
            spectrum_energies[self.spectra[[*self.spectra][spectrum_number]].spectrum_name] = full_energy
        return spectrum_energies


    def get_raw_spectrum_data(self, filename, spectrum_name, stack_spectra):
        wl_data = []
        meas_data = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line_number, line in enumerate(lines):
                if line_number > 30 and (len(line.split(',')) or len(line.split('\t'))) == 2:
                    line_data = line.split(',')
                    if len(line_data) != 2:
                        line_data = line.split('\t')
                    wl_data.append(float(line_data[0]))
                    meas_data.append(float(line_data[1]))
        meas_data = np.asarray(meas_data)
        wl_data = np.asarray(wl_data)
        if meas_data[0] < 0:
            linear_meas_data = np.power(10, meas_data / 10) * 1e-3
            log_meas_data = meas_data
        else:
            linear_meas_data = np.asarray(meas_data)
            log_meas_data = 10 * np.log10(meas_data)
        if stack_spectra:
            #I assume that the spectra are put into the dictionary in order
            if len(self.spectra) > 0:
                log_shift_factor = np.max(list(self.spectra.items())[-1][1].log_meas_data)
                lin_shift_factor = np.max(list(self.spectra.items())[-1][1].linear_meas_data)
                linear_meas_data = linear_meas_data + lin_shift_factor
                #if log_shift_factor < 0:
                #    log_shift_factor *= -1
                log_meas_data = log_meas_data + log_shift_factor
        wl_data = np.asarray(wl_data)
        self.spectra[spectrum_name] = OSA_spectrum(wl_data, linear_meas_data, log_meas_data, spectrum_name)
        

    def get_spectrum_data(self, filename, spectrum_name, stack_spectra = False):
        self.get_raw_spectrum_data(filename, spectrum_name, stack_spectra = stack_spectra)
        return len(self.spectra)

    def apply_filter_mirror(self, spectrum_number, filter_mirror):
        if spectrum_number < 0 or spectrum_number > (len(self.spectra) - 1):
            for i in range(len(self.spectra)):
                local_lin_meas_data = self.spectra[[*self.spectra][i]].linear_meas_data
                local_wl_data = self.spectra[[*self.spectra][i]].wl_data
                interpolated_data = filter_mirror.apply_mirror(local_wl_data, local_lin_meas_data)
                self.spectra[[*self.spectra][i]].linear_meas_data = np.asarray(interpolated_data[1][:])
                self.spectra[[*self.spectra][i]].log_meas_data = 10 * np.log10(interpolated_data[1][:] / 1e-3)
        else:
            local_lin_meas_data = self.spectra[[*self.spectra][spectrum_number]].linear_meas_data
            local_wl_data = self.spectra[[*self.spectra][spectrum_number]].wl_data
            interpolated_data = filter_mirror.apply_mirror(local_wl_data, local_lin_meas_data)
            self.spectra[[*self.spectra][spectrum_number]].linear_meas_data = np.asarray(interpolated_data[1][:])
            self.spectra[[*self.spectra][spectrum_number]].log_meas_data = 10 * np.log10(interpolated_data[1][:] / 1e-3)

    def plot_spectrum(self, spectrum_number, use_linear_scale = False, x_min = -1, x_max = -1, y_min = -1, y_max = -1, use_grid = False):
        plt.figure()
        plt.xlabel("Wavelength [nm]")
        if y_max <= y_min:
            adjust_height_freely = True
        else:
            adjust_height_freely = False
        if use_linear_scale:
            plt.ylabel("Intensity in [mW/nm]")
        else:
            plt.ylabel("Intensity in [dBm/nm]")
        legend_entries = []
        plt.grid(use_grid)
        if spectrum_number < 0 or spectrum_number > len(self.spectra) - 1: # plot all
            for key, value in self.spectra.items():
                if use_linear_scale:
                    plt.plot(value.wl_data, value.linear_meas_data)
                else:
                    plt.plot(value.wl_data, value.log_meas_data)
                legend_entries.append(value.spectrum_name)
                if x_min < 0:
                    x_min = value.wl_data[0]
                if x_max < 0:
                    x_max = value.wl_data[-1]
                if adjust_height_freely:
                    if use_linear_scale:
                        y_min = np.min([np.min(value.linear_meas_data), y_min])
                    else:
                        y_min = np.min([np.min(value.log_meas_data), y_min])
                    if use_linear_scale:
                        y_max = np.max([np.max(value.linear_meas_data), y_max])
                    else:
                        y_max = np.max([np.max(value.log_meas_data), y_max])
        else:
            local_value = self.spectra[[*self.spectra][spectrum_number]]#nth_key(self.spectra, spectrum_number)
            if use_linear_scale:
                plt.plot(local_value.wl_data, local_value.linear_meas_data)
            else:
                plt.plot(local_value.wl_data, local_value.log_meas_data)
            legend_entries.append(local_value.spectrum_name)
        plt.legend(legend_entries)
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.show()

    def save_spectrum(self, spectrum_number, file_name, legend_position = "upper center", file_format = "PNG", use_linear_scale = False, x_min = -1, x_max = -1, y_min = -1, y_max = -1, use_grid = False, with_fit_FWHM = False, plot_title = ""):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.legend(loc = legend_position)
        if y_max <= y_min:
            adjust_height_freely = True
        else:
            adjust_height_freely = False
        plt.xlabel("Wavelength [nm]")
        if use_linear_scale:
            plt.ylabel("Intensity in [mW/nm]")
        else:
            plt.ylabel("Intensity in [dBm/nm]")
        legend_entries = []
        plt.grid(use_grid)
        if spectrum_number < 0 or spectrum_number > len(self.spectra) - 1: # plot all
            for key, value in self.spectra.items():
                if use_linear_scale:
                    plt.plot(value.wl_data, value.linear_meas_data)
                else:
                    plt.plot(value.wl_data, value.log_meas_data)
                legend_entry = value.spectrum_name
                if with_fit_FWHM and value.fit_can_be_used:
                    legend_entry += ", FWHM = " + '{0:.2f}'.format(value.FWHM) + " nm, " + '{0:.2f}'.format(value.bandwidth / 1e9) + " THz"
                legend_entries.append(legend_entry)

                if x_min < 0:
                    x_min = np.min(value.wl_data[0], x_min)
                if x_max < 0:
                    x_max = np.max(value.wl_data[-1], x_max)
                if adjust_height_freely:
                    if use_linear_scale:
                        y_min = np.min([np.min(value.linear_meas_data), y_min])
                    else:
                        y_min = np.min([np.min(value.log_meas_data), y_min])
                    if use_linear_scale:
                        y_max = np.max([np.max(value.linear_meas_data), y_max])
                    else:
                        y_max = np.max([np.max(value.log_meas_data), y_max])
        else:
            local_value = self.spectra[[*self.spectra][spectrum_number]]#nth_key(self.spectra, spectrum_number)
            if use_linear_scale:
                plt.plot(local_value.wl_data, local_value.linear_meas_data)
            else:
                plt.plot(local_value.wl_data, local_value.log_meas_data)
            legend_entries.append(local_value.spectrum_name)
        plt.legend(legend_entries)
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        if plot_title:
            plt.title(plot_title)
        local_file_name = file_name
        if file_format == "PNG":
            if local_file_name[-4:] != ".png":
                local_file_name += ".png"
            plt.savefig(local_file_name, bbox_inches="tight", dpi=600)
        else:
            if local_file_name[-5:] != ".tikz":
                local_file_name += ".tikz"
            tikzplotlib.save(local_file_name)
        plt.clf()

    def fit_spectrum(self, external_spectrum_number = -1, estimated_wavelength = 0, fit_function = 'Gaussian'):
        #find peak
        central_wavelength = estimated_wavelength
        if external_spectrum_number < 0:
            for spectrum_number in range(len(self.spectra)):
                spectrum = self.spectra[[*self.spectra][spectrum_number]]
                peaks, _ = find_peaks(spectrum.linear_meas_data, prominence=(0.001, 1))
                if len(spectrum.wl_data[peaks]) > 0:
                    central_wavelength = float(spectrum.wl_data[peaks][0])
                if fit_function == "Gaussian":
                    popt, pcopt = curve_fit(self.gaussian_function, spectrum.wl_data, spectrum.linear_meas_data, p0 = [np.max(spectrum.linear_meas_data), 1, central_wavelength, 0])
                    
                    print(popt[1])
                    print(popt)
                    plt.clf()
                    plt.plot(self.spectra[[*self.spectra][spectrum_number]].wl_data, self.spectra[[*self.spectra][spectrum_number]].linear_meas_data, self.spectra[[*self.spectra][spectrum_number]].wl_data, self.gaussian_function_compact(self.spectra[[*self.spectra][spectrum_number]].wl_data, popt))
                    plt.show()
                    
                    self.spectra[[*self.spectra][spectrum_number]].fit_function = "Gaussian"
                    self.spectra[[*self.spectra][spectrum_number]].FWHM = popt[1]
                    self.spectra[[*self.spectra][spectrum_number]].bandwidth = scco.speed_of_light / ((central_wavelength - popt[1] / 2) * 1e-9) - scco.speed_of_light / ((central_wavelength + popt[1] / 2) * 1e-9)
                else:
                    popt, pcopt = curve_fit(self.sech_function, spectrum.wl_data, spectrum.linear_meas_data, p0 = [np.max(spectrum.linear_meas_data), 1, central_wavelength, 0])
                    self.spectra[[*self.spectra][spectrum_number]].fit_function = "Sech"
                    self.spectra[[*self.spectra][spectrum_number]].FWHM = 2 * np.log(2 + np.sqrt(3)) * popt[1]
                    self.spectra[[*self.spectra][spectrum_number]].bandwidth = scco.speed_of_light / ((central_wavelength - self.spectra[[*self.spectra][spectrum_number]].FWHM / 2) * 1e-9) - scco.speed_of_light / ((central_wavelength + self.spectra[[*self.spectra][spectrum_number]].FWHM / 2) * 1e-9)
                self.spectra[[*self.spectra][spectrum_number]].fit_meas_data = self.gaussian_function_compact(spectrum.wl_data, popt)
                self.spectra[[*self.spectra][spectrum_number]].fit_data = popt
                self.spectra[[*self.spectra][spectrum_number]].fit_can_be_used = True
        else:
            spectrum_number = external_spectrum_number
            spectrum = self.spectra[[*self.spectra][spectrum_number]]
            peaks, _ = find_peaks(spectrum.linear_meas_data, prominence=(0.001, 1))
            if len(spectrum.wl_data[peaks]) > 0:
                central_wavelength = float(spectrum.wl_data[peaks][0])
            if fit_function == "Gaussian":
                popt, pcopt = curve_fit(self.gaussian_function, spectrum.wl_data, spectrum.linear_meas_data, p0 = [np.max(spectrum.linear_meas_data), 1, central_wavelength, 0])
                
                print(popt[1])
                print(popt)
                plt.clf()
                plt.plot(self.spectra[[*self.spectra][spectrum_number]].wl_data, self.spectra[[*self.spectra][spectrum_number]].linear_meas_data, self.spectra[[*self.spectra][spectrum_number]].wl_data, self.gaussian_function_compact(self.spectra[[*self.spectra][spectrum_number]].wl_data, popt))
                plt.show()
                
                self.spectra[[*self.spectra][spectrum_number]].fit_function = "Gaussian"
                self.spectra[[*self.spectra][spectrum_number]].FWHM = popt[1]
                self.spectra[[*self.spectra][spectrum_number]].bandwidth = scco.speed_of_light / ((central_wavelength - popt[1] / 2) * 1e-9) - scco.speed_of_light / ((central_wavelength + popt[1] / 2) * 1e-9)
            else:
                popt, pcopt = curve_fit(self.sech_function, spectrum.wl_data, spectrum.linear_meas_data, p0 = [np.max(spectrum.linear_meas_data), 1, central_wavelength, 0])
                self.spectra[[*self.spectra][spectrum_number]].fit_function = "Sech"
                self.spectra[[*self.spectra][spectrum_number]].FWHM = 2 * np.log(2 + np.sqrt(3)) * popt[1]
                self.spectra[[*self.spectra][spectrum_number]].bandwidth = scco.speed_of_light / ((central_wavelength - self.spectra[[*self.spectra][spectrum_number]].FWHM / 2) * 1e-9) - scco.speed_of_light / ((central_wavelength + self.spectra[[*self.spectra][spectrum_number]].FWHM / 2) * 1e-9)
            self.spectra[[*self.spectra][spectrum_number]].fit_meas_data = self.gaussian_function_compact(spectrum.wl_data, popt)
            self.spectra[[*self.spectra][spectrum_number]].fit_data = popt
            self.spectra[[*self.spectra][spectrum_number]].fit_can_be_used = True