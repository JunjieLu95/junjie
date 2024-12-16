# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:08:02 2024

@author: Junjie
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from junjie.tools import single_fit
from junjie.num.graph.resonances import extract_func_peak
from mesopylib.num.graphs.graph_bonds import Bond
from mesopylib.num.graphs.graph_vertices import Vertex_neumann
from mesopylib.num.graphs.graph_num import Graph_VNA_effH
from mesopylib.num.graphs.graphs_dirichlet import generate_random_bonds_equal
from mesopylib.extract.fit_resonances import cLorentz

class Superwidth_Graph:
    def __init__(self, l_subtot, l_d, target_ratio, phi0, n_periodic, seed, 
                 flag_minl_v0=False, flag_duplicate_min=False):
        
        self.l_d = l_d                     # dangling bond
        self.l_subtot = l_subtot           # total length excluding dangling bond
        self.target_ratio = target_ratio   # the ratio between the min length of the bonds (attaching to the vetex) and the dangling bond
        self.phi0 = phi0                   # phase of bond
        self.n_phi = len(phi0)
        self.n_periodic = n_periodic
        self.seed = seed
        self.flag_minl_v0 = flag_minl_v0
        self.flag_duplicate_min = flag_duplicate_min
        
        self.eplison = 0
    
    def generate_lengths(self, n_l, n_v, flag_minl_v0=False, flag_duplicate_min=False):
        """
        Generate and optionally adjust the lengths of bonds.
        n_l: the number of lengths
        n_v: the valancy of vertex, since the main focus is on the bond of vertex 0
        flag_minl_v0: - If True, ensures the minimum length is in lengths_v0 (the bonds attaching to v0).
                      - If False, ensures the minimum length is NOT in lengths_v0.
        flag_duplicate_min: if True, ensures that there are exactly two minimum lengths in lengths[:n_v].
        """
        valid = False
        while not valid:
            lengths = generate_random_bonds_equal(n_l, realizations=1, seed=self.seed, t_length=self.l_subtot)[0, :]
            lengths_v0 = lengths[:n_v]
            
            min_index = np.argmin(lengths)

            target_range = range(0, n_v) if flag_minl_v0 else range(n_v, n_l)
            if min_index not in target_range:
                random_index = np.random.choice(target_range)
                lengths[min_index], lengths[random_index] = lengths[random_index], lengths[min_index]
                
            
            if flag_duplicate_min:
                min_length = np.min(lengths_v0)
                if np.count_nonzero(lengths_v0 == min_length) < 2:
                    non_min_indices = [i for i in range(n_v) if lengths_v0[i] != min_length]
                    if non_min_indices:
                        random_non_min_index = np.random.choice(non_min_indices)
                        lengths[random_non_min_index] = min_length
            
            r = np.min(lengths_v0) / self.l_d
            scale_factor = self.target_ratio / r
            lengths *= scale_factor
            
            if np.max(lengths) <= 120:   # avoid bug of calculation of np.sin(k*l), it causes problem when k is complex and l is so large
                valid = True
            else:
                self.seed += 1
        
        return lengths
    
    def Tetrahedral_graph_exp(self):
        """
        Tetrahedral graph setup, and get the ratio between the min length of three bonds and the dangling bond
        """
        self.theta = 1/3
        self.lengths = self.generate_lengths(7, 2, self.flag_minl_v0, self.flag_duplicate_min)

        bonds = [Bond(ilength, name='b' + str(i)) for i, ilength in enumerate(self.lengths)]
        
        vn = [Vertex_neumann(3, name='v' + str(i + 1)) for i in range(5)] 
        
        vn[0].connect2vertex(0, 0, vn[1], bonds[0])
        vn[0].connect2vertex(1, 0, vn[3], bonds[1])
        
        vn[1].connect2vertex(1, 0, vn[2], bonds[2])
        vn[1].connect2vertex(2, 0, vn[4], bonds[3])
        
        vn[2].connect2vertex(1, 1, vn[3], bonds[4])
        vn[2].connect2vertex(2, 1, vn[4], bonds[5])
        
        vn[3].connect2vertex(2, 2, vn[4], bonds[6])

        self.vna = Graph_VNA_effH(vn, connect=[0], couplings=[1])
        self.cal_ratio()
    
    def Tetrahedral_graph(self):
        """
        Tetrahedral graph setup, vetrex 0 is the one coupling to VNA
        """
        self.theta = 2/4
        self.lengths = self.generate_lengths(6, 3, self.flag_minl_v0, self.flag_duplicate_min)
                
        bonds = [Bond(ilength, name='b' + str(i)) for i, ilength in enumerate(self.lengths)]
    
        vn = [Vertex_neumann(4, name='v0')] 
        for i in range(3):
            vn.append(Vertex_neumann(3, name='v' + str(i + 1)))

        vn[0].connect2vertex(0,0,vn[1],bonds[0])
        vn[0].connect2vertex(1,0,vn[2],bonds[1])
        vn[0].connect2vertex(2,0,vn[3],bonds[2])
        
        vn[1].connect2vertex(1,1,vn[2],bonds[3])
        vn[1].connect2vertex(2,1,vn[3],bonds[4])
        
        vn[2].connect2vertex(2,2,vn[3],bonds[5])

        self.vna = Graph_VNA_effH(vn, connect=[0], couplings=[1])
        self.cal_ratio()
        
    def Fully_connected_v4(self):
        """
        The fully connected graph with 4 vertices, vetrex 0 is the one coupling to VNA
        """
        self.theta = 3/5
        self.lengths = self.generate_lengths(10, 4, self.flag_minl_v0, self.flag_duplicate_min)
                
        bonds = [Bond(ilength, name='b' + str(i)) for i, ilength in enumerate(self.lengths)]
    
        vn = [Vertex_neumann(5, name='v0')] 
        for i in range(4):
            vn.append(Vertex_neumann(4, name='v' + str(i + 1)))

        vn[0].connect2vertex(0, 0, vn[1], bonds[0])
        vn[0].connect2vertex(1, 0, vn[2], bonds[1])
        vn[0].connect2vertex(2, 0, vn[3], bonds[2])
        vn[0].connect2vertex(3, 0, vn[4], bonds[3])
    
        vn[1].connect2vertex(1, 1, vn[2], bonds[4])
        vn[1].connect2vertex(2, 1, vn[3], bonds[5])
        vn[1].connect2vertex(3, 1, vn[4], bonds[6])
    
        vn[2].connect2vertex(2, 2, vn[3], bonds[7])
        vn[2].connect2vertex(3, 2, vn[4], bonds[8])
    
        vn[3].connect2vertex(3, 3, vn[4], bonds[9])

        self.vna = Graph_VNA_effH(vn, connect=[0], couplings=[1])
        self.cal_ratio()
    
    def Fully_connected_v5(self):
        """
        The fully connected graph with 5 vertices, vetrex 0 is the one coupling to VNA
        """
        self.theta = 4/6
        self.lengths = self.generate_lengths(15, 5, self.flag_minl_v0, self.flag_duplicate_min)
                
        bonds = [Bond(ilength, name='b' + str(i)) for i, ilength in enumerate(self.lengths)]
        
        vn = [Vertex_neumann(6, name='v0')] 
        for i in range(5):
            vn.append(Vertex_neumann(5, name='v' + str(i + 1)))
         
        vn[0].connect2vertex(0, 0, vn[1], bonds[0])
        vn[0].connect2vertex(1, 0, vn[2], bonds[1])
        vn[0].connect2vertex(2, 0, vn[3], bonds[2])
        vn[0].connect2vertex(3, 0, vn[4], bonds[3])
        vn[0].connect2vertex(4, 0, vn[5], bonds[4])
    
        vn[1].connect2vertex(1, 1, vn[2], bonds[5])
        vn[1].connect2vertex(2, 1, vn[3], bonds[6])
        vn[1].connect2vertex(3, 1, vn[4], bonds[7])
        vn[1].connect2vertex(4, 1, vn[5], bonds[8])
    
        vn[2].connect2vertex(2, 2, vn[3], bonds[9])
        vn[2].connect2vertex(3, 2, vn[4], bonds[10])
        vn[2].connect2vertex(4, 2, vn[5], bonds[11])
    
        vn[3].connect2vertex(3, 3, vn[4], bonds[12])
        vn[3].connect2vertex(4, 3, vn[5], bonds[13])
    
        vn[4].connect2vertex(4, 4, vn[5], bonds[14])
    
        self.vna = Graph_VNA_effH(vn, connect=[0], couplings=[1])
        self.cal_ratio()
    
    def cal_ratio(self):
        """
        Return the ratio between the length of min bond attaching to vertex 0, 
        and the length of dangling bond
        """
        l_v0 = [self.l_d]
        for i in self.vna.vertices[0].bonds:
            if i[0] is not None and hasattr(i[0], 'l12'):
                l_v0.append(i[0].l12)
        self.r = np.min(l_v0)/self.l_d
    
    def cal_beta(self, phi):
        """
        beta = alpha/z = -tan(phi/2)   *need to be check!!!!!!  This is derived according to the simulation
        """
        return -np.tan(phi/2)
    
    def cal_radius(self):
        """
        See Corollary 8 in the manuscript.
        Return the radius of disks.
        Here we choose eplison=0, and theta=1/3.
        """
        disks = np.zeros((self.n_periodic, self.n_phi))
        for i in range(self.n_periodic):
            for j in range(self.n_phi):
                beta = self.cal_beta(self.phi0[j])
                disks[i,j] = (1+self.eplison)/2/self.l_d/self.theta * np.abs(beta/2/self.theta)**self.r
                # disks[i,j] = (1+self.eplison)/2/self.l_d/self.theta * np.abs(beta)**self.r
        return np.real(disks)
    
    def cal_prediction(self, phi, nn):
        """
        Using the approximation to predict the k value, when phi is closed to 0
        see the document for the detail of derivation

        """
        beta = self.cal_beta(phi)
        b2t = beta/2/self.theta
        return 1j/2/self.l_d*(np.log(np.abs(b2t))) - np.angle(b2t)/2/self.l_d + (nn+1)*np.pi/self.l_d - np.pi/4/self.l_d   #nn+1 is for postive k
         
    def cal_S1(self, phi):
        """
        Graph 1 is a bond with variable phase
        """
        return np.exp(1j * phi)

    def cal_S2(self, k):
        """
        Graph 2 is a graph with one dangling bond and complex graph
        """
        return np.exp(2j * k * self.l_d) * self.vna.calcS(k)[:, 0]
    
    def cal_S(self, k, phi):
        """
        Using T-junction structure,
        one port attaches lead,
        the other two ports attach graph 1 and 2 (S1 and S2)
        """
        S1 = self.cal_S1(phi)
        S2 = self.cal_S2(k)
        S = -(3 * S1 * S2 + S1 + S2 - 1) / (S1 * S2 - S1 - S2 - 3)
        return S
    
    def S_cal_func(self, k_grid, phi):
        """
        Function used with "extract_resonance_grid"

        """
        S = self.cal_S(k_grid, phi)
        return S
    
    def extract_resonance_grid(self, flag_plot=False):
        """
        Use the grid method to extract resonances
        """

        n_periodics = np.arange(self.n_periodic)
        k_predict = np.zeros((self.n_periodic, self.n_phi), dtype=complex)
        k_numeric = np.zeros((self.n_periodic, self.n_phi), dtype=complex)
        
        for i in range(self.n_periodic):
            for j in range(self.n_phi):
                phi = self.phi0[j]
                k_predict[i, j] = self.cal_prediction(phi, n_periodics[i])

                try:
                    # First extraction
                    k_rough_re = np.real(k_predict[i, j])
                    k_rough_im = np.imag(k_predict[i, j])
                    kr_rough = (k_rough_re-0.3, k_rough_re+0.3, 101)  #Large window
                    ki_rough = (k_rough_im-0.4, k_rough_im+0.1)   # Large window
                    
                    
                    k_rough = extract_func_peak(kr_rough, ki_rough, graph_info=[phi], kr_tolrance=0, 
                                               ki_tolrance=0, cal_func=self.S_cal_func, extract_type='poles', 
                                               n_discrete=101, flag_plot=flag_plot)
                    
                    # Second extraction
                    k_precise_re = np.real(k_rough)
                    k_precise_im = np.imag(k_rough)
                    kr_precise  = (k_precise_re-0.05, k_precise_re+0.05, 101)  # Small window
                    ki_precise  = (k_precise_im-0.05, k_precise_im+0.05)   # Small window
                    
                    result = extract_func_peak(kr_precise, ki_precise, graph_info=[phi], kr_tolrance=0, 
                                               ki_tolrance=0, cal_func=self.S_cal_func, extract_type='poles', 
                                               n_discrete=151, flag_plot=flag_plot)
                    if result is not None:
                        if isinstance(result, np.ndarray) and result.size == 1:
                            k_numeric[i, j] = result.item()  # Convert single-element array to scalar
                        elif isinstance(result, complex):
                            k_numeric[i, j] = result  # Directly use the result if already a scalar
                        else:
                            raise ValueError("Unexpected result format")
                    else:
                        k_numeric[i, j] = np.nan
                        print(f"Extraction failed for window {i}, setting result as NaN.")
                except Exception as e:
                    k_numeric[i, j] = np.nan
                    print(f"An error occurred for window {i}: {str(e)}")
                    
        return k_predict, k_numeric
    
    
    def cal_S_shifted_k(self, phi, k_guess, shift_imag=-1e-9j, ss=1, flag_fp=False):
        """
        Calculate the spectrum for the considered graph,
        with k_guess, we roughly know where the poles are,
        so we set a frq range with 2*ss length, then we can get the spectrum
        
        PS: We need to make sure there are enough points in the resonance peak
        and the peak is almost in the center. 
        Tip: When dealing the case that the peak of resonance is too sharp, 
        it is better to adjust the imag part of frq.
        
        If flag_fp=True, use np.arange with step 0.0002, find peaks, and select the one closest
        to the real part of k_guess. Then choose 5000 points around the peak.

        """
        if flag_fp:
            k = np.arange(np.real(k_guess) - ss, np.real(k_guess) + ss, 0.0002) + 1j * np.imag(k_guess) + shift_imag
    
            S = self.S_cal_func(k, phi)
    
            peaks, _ = find_peaks(np.abs(S))
            
            peak_real_parts = np.real(k[peaks])
            closest_peak_idx = peaks[np.argmin(np.abs(peak_real_parts - np.real(k_guess)))]
    
            start_idx = max(0, closest_peak_idx - 2500)
            end_idx = min(len(k), closest_peak_idx + 2500)
    
            k = k[start_idx:end_idx]
            S = S[start_idx:end_idx]
        else:
            
            k = np.linspace(np.real(k_guess) - ss, np.real(k_guess) + ss, 2000) + 1j * np.imag(k_guess) + shift_imag
            S = self.S_cal_func(k, phi)        
      
        return k, S
    
    def extract_resonance_cLorentz(self, flag_plot=False):
        """
        Return the numerical results and prediction results of poles.

        """
        n_periodics = np.arange(self.n_periodic)
        k_predict = np.zeros((self.n_periodic, self.n_phi), dtype=complex)
        k_numeric = np.zeros((self.n_periodic, self.n_phi), dtype=complex)
                
        for i in range(self.n_periodic):
            for j in range(self.n_phi):
                phi = self.phi0[j]
                k_p0 = self.cal_prediction(phi, n_periodics[i])
                k_predict[i, j] = k_p0
                
                try:
                    # First extraction
                    k_c, S = self.cal_S_shifted_k(phi, k_p0, shift_imag=-0.02j, ss=1, flag_fp=True)
                    
                    k_rough = single_fit(S, k_c)[1] + np.imag(k_c[0])*1j

                    if flag_plot:
                        plt.figure('First Lorentz fit')
                        plt.plot(k_c.real, np.abs(S))
                        plt.plot(k_c.real, np.abs(cLorentz(k_c.real, single_fit(S, k_c))))
                        plt.xlabel('k')
                        plt.ylabel('|S|')
                    
                    # Second extraction
                    k_c, S = self.cal_S_shifted_k(phi, k_rough, shift_imag=-0.0001j, ss=0.002)
                    # k_c, S = self.cal_S_shifted_k(phi, k_rough, shift_imag=-0.0005j, ss=0.002)
                    
                    k_num = single_fit(S, k_c)[1] + np.imag(k_c[0])*1j

                    if flag_plot:
                        plt.figure('Second Lorentz fit')
                        plt.plot(k_c.real, np.abs(S))
                        plt.plot(k_c.real, np.abs(cLorentz(k_c.real, single_fit(S, k_c))))
                    
                    k_numeric[i, j] = k_num
                    
                    if np.isnan(k_num) or np.real(k_num)>3000 or np.real(k_num)<-3000:
                        print((i, j))
                        print('fit error')
                except:
                    k_numeric[i, j] = np.nan   
                    print((i, j))
                    print('fit error')
        
        return k_predict, k_numeric
        