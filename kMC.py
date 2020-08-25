from surface import Surface
from math import log
import numpy as np
from copy import deepcopy


class KMC:
    def __init__(self, dt, length):
        if length % 2 != 0:
            print('Length has to be an even number!')
            raise ValueError
        else:
            self.site_distance = 2.087e-10  # m Ni-O bond is 2.087A
            self.NA = 6.02e23
            # Area should be used for flux and as a chess board model
            self.Area = (length * self.site_distance) ** 2
            self.t_final = dt

    def get_flux(self, C_CO, C_H2, p_CO, p_H2, surface):
        t = 0
        step = 0
        surface.update(C_CO, C_H2, p_CO, p_H2)
        while t < self.t_final:
            r = np.random.uniform()
            while r == 0.0:
                r = np.random.uniform()
            if surface.W_tot == 0:
                return [0, 0, 0, 0]
            dt = - (1 / surface.W_tot) * log(r)
            surface.execute_event(C_CO, C_H2, p_CO, p_H2)
            step += 1
            t += dt

        CO_flux = - (surface.num_CO_consumed / self.NA) / (
            t * self.Area)
        CO2_flux = (surface.num_CO2_produced / self.NA) / (
            t * self.Area)
        H2_flux = - (surface.num_H2_consumed / self.NA) / (
            t * self.Area)
        H2O_flux = (surface.num_H2O_produced / self.NA) / (
            t * self.Area)

        return [CO_flux, CO2_flux, H2_flux, H2O_flux]
