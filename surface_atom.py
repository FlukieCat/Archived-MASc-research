import random
import numpy as np


class SurfaceAtom:
    def __init__(self, loc_self, loc_neibr, length):
        self.len = length
        self.loc = loc_self
        self.loc_neibr = loc_neibr
        self.neibr_diag =\
            [loc_neibr[0], loc_neibr[2], loc_neibr[5], loc_neibr[7]]
        self.neibr_direc =\
            [loc_neibr[1], loc_neibr[3], loc_neibr[4], loc_neibr[6]]
        self.occupancy = 0
        self.adsorbate = None
        self.neibr_factor = 0.5
        self.s0CO = 1
        self.s0H2 = 1 
        NA = 6.02e23
        kb = 1.3806e-23
        self.T = 1223 # K
        self.Asite = ((2 * 2.087e-10) ** 2) / 2
        self.k_a_CO_coeff = self.s0CO * 1.01e5 * self.Asite / ( #change atm into Pa 1.01e5
            2 * np.pi * 0.028 / NA * kb * self.T) ** 0.5
        self.k_a_H2_coeff = self.s0H2 * 1.01e5 * self.Asite / (  # change atm into Pa 1.01e5
            2 * np.pi * 0.002 / NA * kb * self.T) ** 0.5
        self.k_mod = 1e-13
        self.k_a_CO = 0
        self.k_a_H2 = 0
        self.k_d_H = 2.463E+10 * self.k_mod
        self.k_r_CO = 4.471E+04 * self.k_mod
        self.k_r_H = 1.905E+06 * self.k_mod
        self.events = []

    def __repr__(self):
        return "SurfaceAtom({}, {})".format(
            self.loc, self.loc_neibr)

    def __str__(self):
        return "[{}, {}] occupied by {}".format(
            self.loc.x, self.loc.y, self.adsorbate)

    def calculate_k_ad(self, p_CO, p_H2):
        self.k_a_CO = self.k_a_CO_coeff * p_CO * self.k_mod
        self.k_a_H2 = self.k_a_H2_coeff * p_H2 * self.k_mod

    def is_empty(self):
        if self.occupancy == 0:
            return True
        else:
            return False

    def set_adsorbate(self, dweller):
        if self.adsorbate == 'H2' or self.adsorbate is None:
            self.adsorbate = dweller
            self.occupancy = 1
        else:
            print('Operation failed: \
                [{}, {}] is already occupied by {} (occupancy == {}).'.format(
                self.loc.x, self.loc.y, self.adsorbate, self.occupancy))

    def take_adsorbate(self):
        if self.occupancy == 1:
            self.adsorbate = None
            self.occupancy = 0
        else:
            print('Operation failed: \
                [{}, {}] is empty (adsorbate is {}, occupancy == {}).'.format(
                self.loc.x, self.loc.y, self.adsorbate, self.occupancy))

    def find_empty_direc_neibr(self, surface):
        pool = []
        for loc in self.neibr_direc:
            if surface.config[loc.x, loc.y].occupancy == 0:
                pool.append(loc)
        try:
            return random.choice(pool)
        except IndexError:
            print(self)
            print(self.loc)
            print(self.events)
            print('====================')
            for loc in self.neibr_direc:
                print(surface.config[loc.x, loc.y])

    def count_empty_direc_neibr(self, surface):
        n = 0
        for loc in self.neibr_direc:
            if surface.config[loc.x, loc.y].occupancy == 0:
                n += 1
        return n

    def has_empty_direc_neibr(self, surface):
        for loc in self.neibr_direc:
            if surface.config[loc.x, loc.y].occupancy == 0:
                return 1
        return 0

    def has_neibr(self, surface):
        for loc in self.loc_neibr:
            if not surface.config[loc.x, loc.y].is_empty():
                return 1
        return 0

    def find_direc_H_neibr(self, surface):
        pool = []
        for loc in self.neibr_direc:
            if surface.config[loc.x, loc.y].adsorbate == 'H':
                pool.append(loc)
        try:
            return random.choice(pool)
        except IndexError:
            print(self)
            print(self.loc)
            print(self.events)
            print('====================')
            for loc in self.neibr_direc:
                print(surface.config[loc.x, loc.y])

    def count_direc_H_neibr(self, surface):
        n = 0
        for loc in self.neibr_direc:
            if surface.config[loc.x, loc.y].adsorbate == 'H':
                n += 1
        return n

    def has_direc_H_neibr(self, surface):
        for loc in self.neibr_direc:
            if surface.config[loc.x, loc.y].adsorbate == 'H':
                return 1
        return 0

    def adsorption(self, surface, dweller):
        if dweller == 'CO':
            self.set_adsorbate('CO')
            affected_region = [self.loc] + self.loc_neibr
            return affected_region
        elif dweller == 'H2':
            self.set_adsorbate('H2')
            affected_region = [self.loc] + self.loc_neibr
            return affected_region
        elif dweller == 'H':
            loc = self.find_empty_direc_neibr(surface)
            self.set_adsorbate('H')
            surface.config[loc.x, loc.y].set_adsorbate('H')
            affected_region = \
                self.loc_neibr + surface.config[loc.x, loc.y].loc_neibr
            return affected_region

    def reaction(self, surface, reactant):
        if reactant is 'CO':
            self.take_adsorbate()
            affected_region = [self.loc] + self.loc_neibr
            return affected_region
        else:
            loc = self.find_direc_H_neibr(surface)
            self.take_adsorbate()
            surface.config[loc.x, loc.y].take_adsorbate()
            affected_region = \
                self.loc_neibr + surface.config[loc.x, loc.y].loc_neibr
            return affected_region


class Ni(SurfaceAtom):
    def __init__(self, loc_self, loc_neibr, length):
        super().__init__(loc_self, loc_neibr, length)

    def __repr__(self):
        return "Ni({}, {})".format(
            self.loc, self.loc_neibr)

    def __str__(self):
        return 'Ni ({})'.format(str(self.adsorbate))

    def has_4_CO_neibrs(self, surface):
        count = 0
        for nei in self.neibr_diag:
            if surface.config[nei.x, nei.y].adsorbate is 'CO':
                count += 1
        if count < 4:
            return False
        else:
            return True

    def calculate_W(self, surface, C_CO, C_H2):
        empty = self.is_empty()
        CO_blocked = self.has_4_CO_neibrs(surface)
        n = self.count_empty_direc_neibr(surface)
        m = self.count_direc_H_neibr(surface)
        q = self.has_neibr(surface)  # return 1 if True, 0 is False
        if empty:
            if CO_blocked:
                self.events = [0, 0, (1 - q) * self.k_a_H2 * C_H2,
                               q * self.neibr_factor * self.k_a_H2 * C_H2,
                               0, 0, 0]
            else:
                self.events = [
                    (1 - q) * self.k_a_CO * C_CO,
                    q * self.neibr_factor * self.k_a_CO * C_CO,
                    (1 - q) * self.k_a_H2 * C_H2,
                    q * self.neibr_factor * self.k_a_H2 * C_H2,
                    0, 0, 0]
        else:
            if self.adsorbate == 'CO':
                self.events = [0, 0, 0, 0, 0, n * self.k_r_CO, 0]
            elif self.adsorbate == 'H2':
                self.events = [0, 0, 0, 0, n * self.k_d_H, 0, 0]
            elif self.adsorbate == 'H':
                self.events = [0, 0, 0, 0, 0,  m * self.k_r_H]


class Oxygen(SurfaceAtom):
    def __init__(self, loc_self, loc_neibr, length):
        super().__init__(loc_self, loc_neibr, length)

    def __repr__(self):
        return "O({}, {})".format(
            self.loc, self.loc_neibr)

    def __str__(self):
        return 'O ({})'.format(str(self.adsorbate))

    def calculate_W(self, surface, C_CO, C_H2):
        empty = self.is_empty()
        # n = self.count_empty_direc_neibr(surface)
        m = self.count_direc_H_neibr(surface)
        # if empty and n == 0:
        #     self.events = [0, 0, 0, 0, 0, 0]
        # elif empty and n > 0:
        #     if self.has_neibr(surface):
        #         self.events = [
        #             0, self.neibr_factor * self.k_a_H2 * C_H2, 0, 0, 0, 0]
        #     else:
        #         self.events = [
        #             0, self.k_a_H2 * C_H2, 0, 0, 0, 0]
        # elif not empty:
        #     if m == 0:
        #         self.events = [0, 0, 0, 0, 0, 0]
        #     else:
        #         self.events = [0, 0, 0, 0, 0, self.k_r_H]
        if empty:
            self.events = [0, 0, 0, 0, 0, 0, 0]
        else:
            self.events = [0, 0, 0, 0, 0, 0,  m * self.k_r_H]
