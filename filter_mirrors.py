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

class mirror_filter_I_LMA:
    def __init__(self):
        self.mirror_data_file = "/home/roland/Dokumente/Uni/Verstärker-Tests/109612 ASCII.csv"
        self.scaling_factor = 1
        self.wl_shift = 7
        self.name = "First mirror"
        self.line_parameters = [5, 1]

class mirror_filter_II_LMA:
    def __init__(self):
        self.mirror_data_file = "/home/roland/Dokumente/Uni/Verstärker-Tests/1800 mirror.csv"
        self.scaling_factor = 95 / 58
        self.wl_shift = -6
        self.name = "Second mirror"
        self.line_parameters = [3, 2, 2700]

class filter_mirror:
    def __init__(self, mirror_data, scaling_factor, wl_shift, name, line_parameters = []):
        self.mirror_data_file = mirror_data
        self.scaling_factor = scaling_factor
        self.wl_shift = wl_shift
        self.name = name
        self.line_parameters = line_parameters
        self.create_mirror()

    @classmethod
    def fromMirrorClass(filter_mirror, mirror_template):
        return filter_mirror(mirror_template.mirror_data_file, mirror_template.scaling_factor, mirror_template.wl_shift, mirror_template.name, mirror_template.line_parameters)

    def apply_mirror(self, wl_data, meas_data):
        ret_meas_data = []
        for i, elem in enumerate(wl_data):
            ret_meas_data.append(self.mirror_interpolator(elem) * meas_data[i])
        return (np.asarray(wl_data), np.asarray(ret_meas_data))

    def create_mirror(self):
        mirror_wl_data = []
        mirror_ref_data = []
        with open(self.mirror_data_file, 'r') as f:
            data_lines = f.readlines()
            for line_number, line in enumerate(data_lines):
                data = line.split(',')
                if len(self.line_parameters) == 2:
                    if len(data) == self.line_parameters[0] and line_number > self.line_parameters[1]:
                        mirror_wl_data.append(float(data[0]) + self.wl_shift)
                        mirror_ref_data.append(float(data[1]))
                else:
                    if len(self.line_parameters) == 3:
                        if len(data) == self.line_parameters[0] and line_number > self.line_parameters[1] and line_number < self.line_parameters[2]:
                            mirror_wl_data.append(float(data[0]) + self.wl_shift)
                            mirror_ref_data.append(float(data[1]))
        self.mirror_wl_data = np.flip(np.asarray(mirror_wl_data))
        self.mirror_ref_data = 1 / (1 - self.scaling_factor * np.flip(np.asarray(mirror_ref_data)) / 100)
        self.mirror_interpolator = interp1d(self.mirror_wl_data, self.mirror_ref_data)