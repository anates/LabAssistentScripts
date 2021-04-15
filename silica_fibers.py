#!/usr/bin/env python3

import h5py
import numpy as np
import scipy.constants as scco
from scipy.signal import butter, lfilter, freqz
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Qt5Cairo')
from matplotlib import pyplot as plt
from scipy import interpolate

import math
from string import Template
import subprocess

import types
import functools

from enum import Enum

#mirror_spectrum_file = "/home/roland/Dokumente/Uni/Verstärker-Tests/109612 ASCII.csv"
#mirror_II_spectrum_file = "/home/roland/Dokumente/Uni/Verstärker-Tests/1800 mirror.csv"
#import mirror data
'''
mirror_wl_data = []
mirror_ref_data = []
mirror_selection = 0
with open(mirror_spectrum_file, 'r') as f:
    data_lines = f.readlines()
    for line_number, line in enumerate(data_lines):
        data = line.split(',')
        if len(data) == 5 and line_number > 1:
            mirror_wl_data.append(float(data[0]))
            mirror_ref_data.append(float(data[mirror_selection + 1]))
mirror_wl_data = np.flip(np.asarray(mirror_wl_data))
mirror_ref_data = np.flip(np.asarray(mirror_ref_data)) / 100
'''

class SMF28:
    def __init__(self):
        self.dispersion_data_file = "Dispersion_SMF28.csv"
        self.name = "SMF-28"
        self.core_diameter = 0
        self.cladding_diameter = 0
        self.coating_diameter = 0


class fiber:
    def __init__(self, fiber_dispersion_data, core_size, cladding_sizes, coating_size, name):
        self.dispersion_data = fiber_dispersion_data
        self.core_size = core_size
        self.cladding_sizes = cladding_sizes
        self.name = name
        self.coating_size = coating_size
        self.beta_vals = []
        self.D_vals = []
        self.wl_vals = []
        self.create_fiber()

    @classmethod
    def fromFiberClass(fiber, fiber_template):
        return fiber(fiber_template.dispersion_data_file, fiber_template.core_diameter, fiber_template.cladding_diameter, fiber_template.coating_diameter, fiber_template.name)

    def create_fiber(self):
        fiber_wl_data = []
        fiber_ref_data = []
        with open(self.dispersion_data, 'r') as f:
            data_lines = f.readlines()
            for line_number, line in enumerate(data_lines):
                data = line.split(';')
                if len(data) == 2:
                    fiber_wl_data.append(float(data[0].replace(",", ".")) * 1e-9)
                    fiber_ref_data.append(float(data[1].replace(",", ".")))
        self.D_vals = np.asarray(fiber_ref_data) * 1e-12 / (1e-9 * 1e3)
        self.wl_vals = np.asarray(fiber_wl_data)
        self.beta_vals = self.D_vals * (-1 * np.power(self.wl_vals, 2) / (2 * np.pi * scco.speed_of_light))
        self.beta_vals_interpolator = interp1d(self.wl_vals, self.beta_vals)
        self.D_vals_interpolator = interp1d(self.wl_vals, self.D_vals)

    def determine_wavelength_range(self, wavelength):
        if wavelength < 0:
            return wavelength
        if wavelength > 0.1 and wavelength < 10:
            return wavelength * 1e-6 #We use um as wavelength
        if wavelength > 100:
            return wavelength * 1e-9 # We use nm as wavelength
        if wavelength < 0.1:
            return wavelength # We use m as wavelength

    def calc_beta_val(self, wavelength):
        return self.beta_vals_interpolator(self.determine_wavelength_range(wavelength))

    def calc_D_val(self, wavelength):
        return self.D_vals_interpolator(self.determine_wavelength_range(wavelength))