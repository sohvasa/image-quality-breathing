#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:21:43 2022

@author: soham
"""
# Dependencies:
from cmath import *
import numpy as np


# Constants
j = complex(0, 1)
e = exp(1).real


# Default Values
sigma_default = 0.0  # Real component. When 0, the result is the Fourier transform
ends_default = np.asarray([0, 0])


# Forward Transform - Time Domain to Laplace Domain
def Laplace(data, is_inverse, sigma=sigma_default, frequency_stamps=None, time_stamps=None, ends=ends_default):
    # Resolve empty data scenario
    data = np.asarray(data)
    if data.size <= 1:
        return data

    # Add time data if missing
    if time_stamps is None:
        time_stamps = np.arange(0, data.size)  # Size doesn't change between forward and inverse
    else:
        time_stamps = np.asarray(time_stamps).real
        if time_stamps.size is not data.size:
            time_stamps = np.arange(0, data.size)

    # Add frequency stamps if missing
    if frequency_stamps is None:
        frequency_stamps = np.asarray(np.arange(0.0, data.size)).real  # Size doesn't change between forward and inverse
        frequency_stamps *= 2 * pi / np.max(frequency_stamps)  # Restrict the integral range to 0 -> 2pi
    else:
        frequency_stamps = np.asarray(frequency_stamps).real
    frequency_stamps = sigma + frequency_stamps * j

    # Create the vector of powers exp(1) is raised to. Also create the delta times / frequencies
    if is_inverse is False:
        power = -Get_Powers(time_stamps, frequency_stamps)
        delta = np.diff(time_stamps)
    else:
        power = Get_Powers(frequency_stamps, time_stamps)
        delta = np.diff(frequency_stamps)
    delta = np.concatenate([[np.average(delta)], delta])  # Ensure a start value is present

    # Perform a numerical approximation of the Laplace transform
    laplace = data * np.power(e, power) * delta
    laplace = laplace.transpose()
    laplace[[0, -1]] *= 0.5  # Trapezium rule => average 1st and last wrt zero
    laplace = laplace.transpose()
    laplace = np.sum(laplace, 1)  # Integrate

    # If inverse function, then normalise and ensure the result is real
    if is_inverse is True:
        laplace *= 1 / (2 * pi * j)  # Scale
        laplace = laplace.real  # Ensure time series is real only

        # Correct for edge cases
        laplace[0] = ends[0]
        laplace[-1] = ends[-1]

    # Return the result
    return laplace


# Used to derive the vector of powers exp(1) is to be raised to
def Get_Powers(values1, values2):
    # For forward Laplace, 1 = time, 2 = frequency
    # For inverse Laplace, 1 = frequency, 2 = time
    power = np.ones([values1.size, values2.size])
    power = (power * values2).transpose() * values1
    return power


if __name__ == "__main__":
    a = np.arange(3, 13)
    b = Laplace(a, False, sigma=0.5)
    c = Laplace(b, True, sigma=0.5, ends=np.asarray([3, 12]))
    print(np.asarray(a))
    print(b)
    print(c)