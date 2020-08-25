import numpy as np
import collections
import random
from surface_atom import SurfaceAtom, Ni, Oxygen
from math import log

Location = collections.namedtuple('Location', 'x y')


class Surface:
    def __init__(self, index, dt, length, p_CO, p_H2):
        if length % 2 != 0:
            print('Length has to be an even number!')
            raise ValueError
        else:
            self.index = index
            self.site_distance = 2.087e-10  # m Ni-O bond is 2.087A
            self.NA = 6.02e23
            # Area should be used for flux and as a chess board model
            self.Area = (length * self.site_distance) ** 2
            self.t_final = dt
            self.num_CO_consumed = 0
            self.num_CO_ad = 0
            self.num_H2_consumed = 0
            self.num_H_ad = 0
            self.num_H2_ad = 0
            self.num_CO2_produced = 0
            self.num_H2O_produced = 0
            self.row_pointer = 0
            self.col_pointer = 0
            self.len = length
            self.W_tot = 0
            self.W_matrix = np.zeros((length, length))
            self.config = np.full((length, length), SurfaceAtom)
            for x in range(length):
                for y in range(length):
                    loc_self = Location(x=x, y=y)
                    loc_neibr = self.cal_neibr_loc(x, y)
                    if (x % 2 == 0 and y % 2 == 0):
                        self.config[x, y] = Ni(loc_self, loc_neibr, self.len)
                    elif (x % 2 == 0 and y % 2 == 1):
                        self.config[x, y] = Oxygen(loc_self, loc_neibr, self.len)
                    elif (x % 2 == 1 and y % 2 == 0):
                        self.config[x, y] = Oxygen(loc_self, loc_neibr, self.len)
                    else:
                        self.config[x, y] = Ni(loc_self, loc_neibr, self.len)
            for atom in self:
                atom.calculate_k_ad(p_CO, p_H2)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.row_pointer < self.len:
                if self.col_pointer < self.len:
                    row = self.row_pointer
                    col = self.col_pointer
                    self.col_pointer += 1
                else:
                    self.row_pointer += 1
                    self.col_pointer = 0
                    row = self.row_pointer
                    col = self.col_pointer
                    self.col_pointer += 1
            return self.config[row, col]
        except IndexError:
            self.row_pointer = 0
            self.col_pointer = 0
            raise StopIteration

    def __str__(self):
        output = '|'
        for i in range(self.len):
            for j in range(self.len):
                output += (str(self.config[i, j]) + '|')
            output += '\n|'
        return output

    def cal_neibr_loc(self, x, y):
        N = x - 1
        S = x + 1
        W = y - 1
        E = y + 1
        if x == 0:
            N = (self.len - 1)
        elif x == (self.len - 1):
            S = 0

        if y == 0:
            W = (self.len - 1)
        elif y == (self.len - 1):
            E = 0

        #  ----------> y
        # x|
        #  | l1 l2 l3
        #  | l4 me l5
        #  V l6 l7 l8

        l1 = Location(x=N, y=W)
        l2 = Location(x=N, y=y)
        l3 = Location(x=N, y=E)
        l4 = Location(x=x, y=W)
        l5 = Location(x=x, y=E)
        l6 = Location(x=S, y=W)
        l7 = Location(x=S, y=y)
        l8 = Location(x=S, y=E)

        return [l1, l2, l3, l4, l5, l6, l7, l8]

    def update(self, C_CO, C_H2, p_CO, p_H2):
        for atom in self:
            atom.calculate_k_ad(p_CO, p_H2)
            atom.calculate_W(self, C_CO, C_H2)
            self.W_matrix[atom.loc.x, atom.loc.y] = sum(atom.events)
            self.W_tot += sum(atom.events)

    def select_atom(self):
        selected_atom = random.choices(
            self.config.flatten(), weights=self.W_matrix.flatten(), k=1)
        return selected_atom[0]

    def execute_event(self, C_CO, C_H2, p_CO, p_H2):
        selected_atom = self.select_atom()
        events = selected_atom.events.copy()
        W = sum(events)
        if W == 0:
            return
        else:
            W_select = W * random.uniform(0, 1)
            i = 1
            while W_select > sum(events[:i]):
                if i < len(events) - 1:
                    i += 1
                else:
                    i = 7
                    break
            W_old = 0
            W_new = 0
            if i == 1 or i == 2:
                affected_region = selected_atom.adsorption(self, 'CO')
                self.num_CO_consumed += 1
                self.num_CO_ad += 1
                for a in affected_region:
                    W_old += sum(self.config[a.x, a.y].events)
                    self.config[a.x, a.y].calculate_k_ad(p_CO, p_H2)
                    self.config[a.x, a.y].calculate_W(self, C_CO, C_H2)
                    self.W_matrix[a.x, a.y] = sum(self.config[a.x, a.y].events)
                    W_new += sum(self.config[a.x, a.y].events)
                self.W_tot = self.W_tot - W_old + W_new
            elif i == 3 or i == 4:
                affected_region = selected_atom.adsorption(self, 'H2')
                self.num_H2_consumed += 1
                self.num_H2_ad += 1
                for a in affected_region:
                    W_old += sum(self.config[a.x, a.y].events)
                    self.config[a.x, a.y].calculate_k_ad(p_CO, p_H2)
                    self.config[a.x, a.y].calculate_W(self, C_CO, C_H2)
                    self.W_matrix[a.x, a.y] = sum(self.config[a.x, a.y].events)
                    W_new += sum(self.config[a.x, a.y].events)
                self.W_tot = self.W_tot - W_old + W_new
            elif i == 5:
                self.num_H2_ad -= 1
                self.num_H_ad += 2
                affected_region = selected_atom.adsorption(self, 'H')
                for a in affected_region:
                    W_old += sum(self.config[a.x, a.y].events)
                    self.config[a.x, a.y].calculate_k_ad(p_CO, p_H2)
                    self.config[a.x, a.y].calculate_W(self, C_CO, C_H2)
                    self.W_matrix[a.x, a.y] = sum(self.config[a.x, a.y].events)
                    W_new += sum(self.config[a.x, a.y].events)
                self.W_tot = self.W_tot - W_old + W_new
            elif i == 6:
                affected_region = selected_atom.reaction(self, 'CO')
                self.num_CO2_produced += 1
                self.num_CO_ad -= 1
                for a in affected_region:
                    W_old += sum(self.config[a.x, a.y].events)
                    self.config[a.x, a.y].calculate_k_ad(p_CO, p_H2)
                    self.config[a.x, a.y].calculate_W(self, C_CO, C_H2)
                    self.W_matrix[a.x, a.y] = sum(self.config[a.x, a.y].events)
                    W_new += sum(self.config[a.x, a.y].events)
                self.W_tot = self.W_tot - W_old + W_new
            else:
                affected_region = selected_atom.reaction(self, 'H')
                self.num_H2O_produced += 1
                self.num_H_ad -= 2
                for a in affected_region:
                    W_old += sum(self.config[a.x, a.y].events)
                    self.config[a.x, a.y].calculate_k_ad(p_CO, p_H2)
                    self.config[a.x, a.y].calculate_W(self, C_CO, C_H2)
                    self.W_matrix[a.x, a.y] = sum(self.config[a.x, a.y].events)
                    W_new += sum(self.config[a.x, a.y].events)
                self.W_tot = self.W_tot - W_old + W_new