#!/usr/bin/env python3

import h5py
import numpy as np
import scipy.constants as scco
from scipy.signal import butter, lfilter, freqz, filtfilt
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

class LaserDiode:
    def __init__(self, current_points, power_points):
        self.current_points = current_points
        self.power_points = power_points
        self.power_function = interpolate.interp1d(self.current_points, self.power_points, kind='linear', fill_value="extrapolate")

    def get_power(self, input_current):
        power = self.power_function(input_current)
        if power < 0:
            return 0
        else:
            return self.power_function(input_current)

    def get_power_from_voltage(self, input_voltage): # if LD is based on voltage
        return self.get_power(input_voltage / 0.8)

class DILAS_I(LaserDiode):
    def __init__(self):
        measured_current_points = [10, 12.5, 15, 17.5, 20]
        measured_power_points = [1630, 3220, 4770, 6380, 7930]
        super().__init__(measured_current_points, measured_power_points)
        self.name = "Dilas Diode I"

    def get_power(self, input_current):
        return super().get_power(input_current)

class DILAS_II(LaserDiode):
    def __init__(self):
        #measured_current_points = [10, 12.5, 15, 17.5, 20]
        #measured_power_points = [1830, 3400, 4940, 6510, 8040]
        measured_current_points = [8, 10, 12, 14, 16, 18, 20]
        measured_power_points = [258, 1978, 3698, 5418, 7138, 8858, 10578]
        super().__init__(measured_current_points, measured_power_points)
        self.name = "Dilas Diode II"

    def get_power(self, input_current):
        return super().get_power(input_current)

class DILAS_III(LaserDiode):
    def __init__(self):
        #measured_current_points = [10, 12.5, 15, 17.5, 20]
        #measured_power_points = [2170, 3970, 5800, 7510, 9300]
        measured_current_points = [8, 10, 12, 14, 16, 18, 20]
        measured_power_points = [570, 2470, 4370, 6270, 8170, 10070, 11970]
        super().__init__(measured_current_points, measured_power_points)
        self.name = "Dilas Diode III"

    def get_power(self, input_current):
        return super().get_power(input_current)
        
class nLight(LaserDiode):
    def __init__(self):
        measured_current_points = [0.720, 0.785, 0.845, 0.910]#, 1.13, 1.41, 1.69, 1.96, 2.24]
        measured_power_points = [20, 252, 615, 990]#, 1860, 3120, 4390, 5610, 6820]
        super().__init__(measured_current_points, measured_power_points)
        self.name = "nLight-Diode"

    def get_power(self, input_voltage):
        return super().get_power(input_voltage / 0.8)


