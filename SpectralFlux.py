"""
SpectralFlux.py
Compute the spectral flux between consecutive spectra
This technique can be for onset detection

rectify - only return positive values
"""

import math

def cuberoot(x):
    if x < 0:
        x = abs(x)
        cube_root = x**(1/3)*(-1)
    else:
        cube_root = x**(1/3)
    return cube_root

def spectralFlux(spectra, rectify=False):
    """
    Compute the spectral flux between consecutive spectra
    """
    spectralFlux = []

    # Compute flux for zeroth spectrum
    flux = 0
    for bin in spectra[0]:
        flux = flux + abs(bin)

    spectralFlux.append(flux)

    # Compute flux for subsequent spectra
    for s in range(1, len(spectra)):
        prevSpectrum = spectra[s - 1]
        spectrum = spectra[s]

        flux = 0
        for bin in range(0, len(spectrum)):
            diff = abs(spectrum[bin]) - abs(prevSpectrum[bin])
            #diff = spectrum[bin] - prevSpectrum[bin]
            #diff = cuberoot(spectrum[bin])  - cuberoot(prevSpectrum[bin])
            #diff = round(diff,10)
            # If rectify is specified, only return positive values
            if rectify and diff < 0:
                diff = 0

            flux = flux + abs(diff)

        spectralFlux.append(flux)

    return spectralFlux