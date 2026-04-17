import requests

import datetime
import time

import numpy as np
import scipy as sci
import scipy.stats
from scipy.stats import norm
from scipy import stats, signal
from scipy.integrate import quad
import scipy.constants as cst

from sympy import Symbol, series, exp

from qutip import *

import datetime as dt

import math
import pandas as pd
import seaborn as sns

from ipywidgets import Output
from IPython.display import clear_output, display
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl

import allantools

from IPython.display import display, HTML

class QC():

    def __init__(self,
                 dark=False):
        print("QC loaded.")
        #sns.set()
        #if dark==True:
        #    COLOR = 'white'
        #    BCKCOLOR = 'white'
        #    GRIDCOLOR = 'grey'
        #else:
        #    COLOR = 'black'
        #    BCKCOLOR = 'white'
        #    GRIDCOLOR = 'black'
        #mpl.rcParams['text.color'] = COLOR
        #mpl.rcParams['axes.labelcolor'] = COLOR
        #mpl.rcParams['axes.grid'] = True
        #mpl.rcParams['xtick.color'] = COLOR
        #mpl.rcParams['ytick.color'] = COLOR
        #mpl.rcParams['axes.facecolor'] = BCKCOLOR
        #mpl.rcParams['grid.color'] = GRIDCOLOR
        #mpl.rcParams['grid.linestyle'] = '-'
        #mpl.rcParams['grid.linewidth'] = 0.5
        #mpl.rcParams['errorbar.capsize'] = 2.0
        #mpl.rcParams['lines.markersize'] = 4.0
        #mpl.rcParams['patch.facecolor'] = BCKCOLOR
        #mpl.rcParams['patch.force_edgecolor'] = True
        #mpl.rcParams["savefig.dpi"] = 125
        #sns.set()
        #params = {'legend.fontsize': 'x-large',
        #  'figure.figsize': (9, 6),
        # 'axes.labelsize': 'x-large',
        # 'axes.titlesize':'x-large',
        #'xtick.labelsize':'x-large',
        # 'ytick.labelsize':'x-large'}
        #mpl.rcParams.update(params)
        
    # Define QPN for spin states
    # True sigma_x,y,z values <sigma_l>_true
    # Number of measurement repetitions r

    def UV_AOM_Specs(self, beam_dia, f_mod, verbose=True):
        import numpy as np
        '''
        For UV AOMs from Intraaction: ASM-2202B3

        beam_dia: beam diameter in mm
        f_mod: Modulation freq. in 2 pi MHz

        RETURNS Pulse rise time Tr, Moulation strength M, and Intensity ration max/min = C
        '''
        V = 5950
        Tr=beam_dia*110
        M=np.exp(-3.485*10**(-2)*beam_dia**2*f_mod**2)
        C=(M+1)/(1-M)
        if verbose:
            print("Speed of sound (mm/µs):", V/1000)
            print("Pulse rise time T_r (ns):", Tr)
            print("Modulation freq. (2 pi MHz):", f_mod)
            print("Modulation strength M:", np.round(M,2))
            print("Achivable contrast I_max/I_min: ", np.round(C))
        return Tr, M, C
    
    def generate_sinusoidal(self, time, modulation_frequency, modulation_depth):
        """
        Generates a sinusoidal waveform with specified modulation frequency and depth to simulate AOM response.

        :param time: Array of time points at which the waveform is evaluated.
        :param modulation_frequency: Frequency of the modulation in Hz.
        :param modulation_depth: Depth of the modulation, unitless (ratio or percentage).

        :return: Array of waveform values at each time point.
        """

        modulated_amplitude = 1 - (1-1/modulation_depth)/2 - (1-1/modulation_depth)/2 * np.cos(2 * np.pi * modulation_frequency * time)

        return modulated_amplitude

    def QPN(self, sigma_true, r):
        return 2*np.sqrt(1/r*((sigma_true+1)/2)*(1-((sigma_true+1)/2)))

    def gen_sigma(self, sigma_true, r):
        TQPN=self.QPN(sigma_true, r)
        return np.random.normal(loc=sigma_true, scale=TQPN), TQPN

    def significant_digit(self, x):
        if math.isnan(x) or math.isinf(x):
            return 0
        try:
            y = -int(math.floor(math.log10(abs(x))))
        except:
            y = 0
        return y

    def prnt_rslts(self, arg_names, popt, perr, verbose=False):
        m,v,e = arg_names, popt, perr
        n = max(self.significant_digit(e),0)
        #s_v = '%%%i.%if'%(6+n,n)
        #s_e = '%%%i.%if'%(1+n,n)
        #txt += '%5s'%m + ' '+r'= ' + s_v%v + r' $\pm$ ' + s_e%e + '\n'
        if e==float('Inf'):
            s_v = '%%.%if'%(n)
            txt = '%5s'%m + ' '+r'= ' + s_v%v + '(inf)'
        else:
            s_v = '%%.%if'%(n)
            txt = '%5s'%m + ' '+r'= ' + s_v%v + '(%i)'%(e*10**n)
        if verbose==True: print(txt)
        return txt

    def _ts2str(self, t_ts, str_format='%Y-%m-%d %H:%M:%S'):
        if isinstance(t_ts, (int, float)):
            t_ts = time.localtime(t_ts)
        return time.strftime(str_format,t_ts)
    
    def lamb_dicke(self, kvec, mvec, omega_m, mass):
        '''computes the Lamb Dicke parameter
        @ var kvec: effective laser wavevector
        @ var mvec: mode vector 
        @ var omega_m: motional frequency in 2 pi Hz
        @ var mass: Atomic mass in kg

        returns: Lamb-Dicke parameter
        '''
        return np.dot(kvec, mvec) * 2 * np.sqrt(2) * np.pi / (279.6 * cst.nano) * np.sqrt(cst.hbar/(2*mass*omega_m))
    
    def size_of_vaccum(self, freq):
        '''
        For single 25Mg+ at COM freq (in MHz)
        
        returns: size of ground-state wave function (in m)
        '''
        return np.sqrt(cst.hbar/(2*25*cst.u * 2* np.pi * freq*cst.mega))
    
    def classical_amplitude(self, alpha, freq):
        '''
        For displacement with alpha of single 25Mg+ at freq (in MHz)
        
        returns: classical amplitude (in m)
        '''
        return np.abs(alpha)*2*self.size_of_vaccum(freq)

    def sqr(self, a_r, phi):
        return a_r*np.exp(1j*2*phi)

    def cohr(self, a_a, phi):
        return a_a*np.exp(1j*phi)

    # define Entanglement of Formation (EoF) 
    # for two spatial degrees of two ions (see, e.g., 
    # M.W., PRL 2019 and C.F., PRA 2018)
    def xi(self,r, nbarp, nbarm, wp, wm):
        l1=np.exp(-r)*(np.sqrt((1+2*nbarm)*(1+2*nbarp)*wp))/(2*np.sqrt(wm))
        l2=np.exp(r)*(np.sqrt((1+2*nbarm)*(1+2*nbarp)*wm))/(2*np.sqrt(wp))
        return np.min(np.array(l1,l2))

    def EoF_2ions(self, xi):
        eof = (0.5+xi)**2/(2*xi)*np.log((0.5+xi)**2/(2*xi))-(0.5-xi)**2/(2*xi)*np.log((0.5-xi)**2/(2*xi))
        return eof
    #-----

    #print(EoF_2ions(xi(0.84, 0.001, 0.001, 2.81, 2.81)))

    #Quantum entanglement
    #Ryszard Horodecki, Paweł Horodecki, Michał Horodecki, and Karol Horodecki 
    #Rev. Mod. Phys. 81, 865 – Published 17 June 2009
    #see Eq. 168

    def EoF_2ModeSqueeze(self,r):
        ret=np.cosh(r)**2*np.log2(np.cosh(r)**2)-np.sinh(r)**2*np.log2(np.sinh(r)**2)
        return ret

    def H_binEntropy(self, x):
        '''
        Binary Entropy, below Eq. 133 of
        [Horodecki, R., Horodecki, P., Horodecki, M. & Horodecki, K. 
        Quantum entanglement. Rev. Mod. Phys. 81, 865–942 (2009).]
        '''
        return -x*np.log2(x)-(1-x)*np.log2(1-x)

    def EoF(self, rho):
        '''
        Entanglement of Formation of two qubits, see Eq. 133 of
        [Horodecki, R., Horodecki, P., Horodecki, M. & Horodecki, K. 
        Quantum entanglement. Rev. Mod. Phys. 81, 865–942 (2009).]
        '''
        #print((1+np.sqrt(1-concurrence(rho)**2))/2)
        if (1+np.sqrt(1-concurrence(rho)**2))/2 < 1:
            if np.isnan(self.H_binEntropy((1+np.sqrt(1-concurrence(rho)**2))/2))==True:
                ret = 0
            else:
                ret=self.H_binEntropy((1+np.sqrt(1-concurrence(rho)**2))/2)
        else:
            ret=0
        return ret
    #EoF(final_rho.ptrace([1,2]))

    def show_state_prop(self, rho=None, rho_motion=None, rho_spin=None, ptrace_sel=None, w0=None, verbose=True):
        if ptrace_sel != None and rho != None:
            rho_motion = rho.ptrace(ptrace_sel[0])
            rho_spin = rho.ptrace(ptrace_sel[1])
            combine = True
            props = []
        else:
            combine = False

        ################################################            
        # Spin DoF
        ################################################ 
        if rho_spin != None:
            if len(rho_spin.dims[0]) == 1:
                S = entropy_vn(rho_spin)
                s_xyz = [expect(sigmax(), rho_spin),expect(sigmay(), rho_spin),expect(sigmaz(), rho_spin)]
                if verbose == True:
                    print('Entropy (vN) of state:', round(S,2))
                    print('<s_x,y,z> =', round(s_xyz[0],3),round(s_xyz[1],3),round(s_xyz[2],3))
                    fig, ax=hinton(rho_spin)
                    fig.set_size_inches(4.5,3.75)
                    plt.show()
                props_S = [s_xyz[0],s_xyz[1],s_xyz[2], S]
                if combine==True:
                    props.append(props_S)
                else:
                    props=props_S
            if len(rho_spin.dims[0]) == 2:
                rho_spin_A = rho_spin.ptrace(0)
                rho_spin_B = rho_spin.ptrace(1)
                Pdd=rho_spin.diag()[0]
                Puddu=rho_spin.diag()[1]+rho_spin.diag()[2]
                Puu=rho_spin.diag()[3]
                S = entropy_vn(rho_spin)
                S_A = entropy_vn(rho_spin_A)
                S_B = entropy_vn(rho_spin_B)
                s_xyz_A = [expect(sigmax(), rho_spin_A),
                                  expect(sigmay(), rho_spin_A),
                                  expect(sigmaz(), rho_spin_A)]
                s_xyz_B = [expect(sigmax(), rho_spin_B),
                                  expect(sigmay(), rho_spin_B),
                                  expect(sigmaz(), rho_spin_B)]
                eof = self.EoF(rho_spin)
                fid=0.5*(np.abs(rho_spin[0,0])+np.abs(rho_spin[3,3]))+np.abs(rho_spin[0,3])
                if verbose == True:
                    print('Entropy of state:', round(S,2))
                    print('EoF: ', round(eof,3))
                    print('Entropy of A/B:', round(S_A,2),round(S_B,2))
                    print('P_dd, P_uddu, P_uu:', round(Pdd,2),round(Puddu,2),round(Puu,2))
                    print('<sx>_A/B =', round(s_xyz_A[0],2),round(s_xyz_B[0],2))
                    print('<sy>_A/B =', round(s_xyz_A[1],2),round(s_xyz_B[1],2))
                    print('<sz>_A/B =', round(s_xyz_A[2],2),round(s_xyz_B[2],2))
                    print('Bell-state fidelity',
                          round(fid,2),
                          round(fidelity(self.bell_state(0), rho_spin),2),
                          round(fidelity(self.bell_state(1), rho_spin),2),
                          round(fidelity(self.bell_state(2), rho_spin),2),
                          round(fidelity(self.bell_state(3), rho_spin),2),
                          round(fidelity(self.bell_state(4), rho_spin),2))
                    fig, ax=hinton(rho_spin)
                    fig.set_size_inches(4.5,3.75)
                    plt.ylim(-1,1)
                    plt.show()
                props_S = [S, eof, 
                           s_xyz_A[0], s_xyz_A[1], s_xyz_A[2], S_A, 
                           s_xyz_B[0], s_xyz_B[1], s_xyz_B[2], S_B,
                           Pdd, Puddu, Puu, fid]
                if combine==True:
                    props.append(props_S)
                else:
                    props=props_S

        ################################################            
        # Motional DoF
        ################################################            
        if rho_motion != None:
            if len(rho_motion.dims[0]) == 1:
                n_cut = rho_motion.dims[0][0]
                a = destroy(n_cut)
                x = (a.dag() + a) * np.sqrt(1/2)
                #x = position(n_cut)
                X=expect(x, rho_motion)
                p = 1j * (-a + a.dag()) * np.sqrt(1/2)
                #p = momentum(n_cut) 
                P=expect(p, rho_motion)

                var_x=(variance(x, rho_motion))
                var_p=(variance(p, rho_motion))
                nbar = np.abs(expect(a.dag()*a, rho_motion))
               
                S = entropy_vn(rho_motion)
                props_M = [n_cut, nbar, var_x, var_p, S, X, P]
                if combine==True:
                    props.append(props_M)
                else:
                    props=props_M
                if verbose == True:
                    print('Fck-state n_cut:',n_cut, '( pop. in last:', round(rho_motion.diag()[-1], 4),')')
                    print('Entropy of state:', round(S,2))
                    print('<n> =', round(nbar,2))
                    print('X =', round(np.real(X),2))
                    print('P =', round(np.real(P),2))
                    print('Var_x =', round(np.real(var_x),2))
                    print('Var_p =', round(np.real(var_p),2))
                    print('(Var_x^2+Var_p^2)^0.5 =', round(np.real((var_x**2+var_p**2)**0.5),2))
                    fig, ax = plot_wigner_fock_distribution(rho_motion, colorbar=True, alpha_max=2*(nbar)+1.5); 
                    fig.set_size_inches(4.,2)
                    draw_circle = plt.Circle((0., 0.), 1, fill=False, color='black', ls='--')
                    ax[1].set_aspect(1)
                    ax[1].add_artist(draw_circle)
                    if w0 != None:
                        import matplotlib.ticker as mticker
                        ax[1].set_xlabel('Position (nm)')
                        ax[1].set_ylabel('Momentum (zN µs)')
                        xticks=np.array(ax[1].get_xticks().tolist())*round(1/np.sqrt(2)*np.sqrt(cst.hbar/(2*25*cst.atomic_mass*w0*cst.mega))*10**9,0).tolist()
                        yticks=np.array(ax[1].get_yticks().tolist())*round(1/np.sqrt(2)*np.sqrt(cst.hbar*25*cst.atomic_mass*w0*cst.mega/(2))*10**21*10**6,0)
                        ax[1].xaxis.set_major_locator(mticker.FixedLocator(ax[1].get_xticks()))
                        ax[1].set_xticklabels(xticks)
                        ax[1].yaxis.set_major_locator(mticker.FixedLocator(ax[1].get_yticks()))
                        ax[1].set_yticklabels(yticks)
                    plt.show()        

            if len(rho_motion.dims[0]) == 2:
                n_cut = rho_motion.dims[0][0]
                #Define operators
                a_A = tensor(destroy(n_cut),qeye(n_cut))
                a_B = tensor(qeye(n_cut),destroy(n_cut))
                a_p = 1/np.sqrt(2)*(a_A + a_B)
                a_m = 1/np.sqrt(2)*(a_A - a_B)

                #Calcvulate nbars
                nbar_A = np.abs(expect(a_A.dag()*a_A, rho_motion))
                nbar_B = np.abs(expect(a_B.dag()*a_B, rho_motion))
                nbar_p = np.abs(expect(a_p.dag()*a_p, rho_motion))
                nbar_m = np.abs(expect(a_m.dag()*a_m, rho_motion))

                rho_motion_A = rho_motion.ptrace(0)
                rho_motion_B = rho_motion.ptrace(1)
                S = entropy_vn(rho_motion)
                S_A = entropy_vn(rho_motion_A)
                S_B = entropy_vn(rho_motion_B)
                EN_A = np.log2(partial_transpose(rho_motion, [1,0]).norm())
                EN_B = np.log2(partial_transpose(rho_motion, [0,1]).norm())

                props_M = [n_cut, nbar_p, nbar_m, S, nbar_A, S_A, EN_A, nbar_B, S_B, EN_B]
                if combine==True:
                    props.append(props_M)
                else:
                    props=props_M
                if verbose == True:
                    print('Fck-state n_cut=',n_cut, '( pop. in last:', round(rho_motion.diag()[-1], 4),')')
                    print('Total entropy S:', round(S,2))

                    print('<n>_A/B =',round(nbar_A,2),round(nbar_B,2))
                    print('<n>_+ =',round(nbar_p,2))
                    print('<n>_- =',round(nbar_m,2))

                    print('Partial entropy S_A/B: ',round(S_A,2),round(S_B,2))
                    print('Log. Negativity E_N,A/B: ',round(EN_A,2),round(EN_A,2))

                    fig, ax = plot_wigner_fock_distribution(rho_motion_A, colorbar=True, alpha_max=2*(nbar_A)+1.5)
                    if w0 != None:
                        ax[1].set_xlabel('Position (nm)')
                        ax[1].set_ylabel('Momentum (zN µs)')
                        ax[1].set_xticklabels(ax[1].get_xticks()*round(np.sqrt(cst.hbar/(2*25*cst.atomic_mass*w0*cst.mega))*10**9,0));
                        ax[1].set_yticklabels(ax[1].get_yticks()*round(np.sqrt(cst.hbar*25*cst.atomic_mass*w0*cst.mega/(2))*10**21*10**6,0));
                    fig, ax = plot_wigner_fock_distribution(rho_motion_B, colorbar=True, alpha_max=2*(nbar_B)+1.5)
                    if w0 != None:
                        ax[1].set_xlabel('Position (nm)')
                        ax[1].set_ylabel('Momentum (zN µs)')
                        ax[1].set_xticklabels(ax[1].get_xticks()*round(np.sqrt(cst.hbar/(2*25*cst.atomic_mass*w0*cst.mega))*10**9,0));
                        ax[1].set_yticklabels(ax[1].get_yticks()*round(np.sqrt(cst.hbar*25*cst.atomic_mass*w0*cst.mega/(2))*10**21*10**6,0));
                    plt.show()
        return props

    def initialise_single_mode(self, n_th=0.0, Fck=0, sq_ampl=0., sq_phi=0, dis_ampl=0., dis_phi=0., Prec=0.01, Ncut=None, verbose = True):
        def c_vec(ampl, phi):
            return ampl*np.exp(1j*phi)
        
        def ini_rho_motion(n_cut):
            if Fck<2:
                a = destroy(n_cut)
                return displace(n_cut, c_vec(dis_ampl, dis_phi)).dag()*squeeze(n_cut, c_vec(sq_ampl, sq_phi)).dag()*a.dag()**Fck*thermal_dm(n_cut,n_th)*a**Fck*squeeze(n_cut, c_vec(sq_ampl, sq_phi))*displace(n_cut, c_vec(dis_ampl, dis_phi))
            if Fck>1:
                return displace(n_cut, c_vec(dis_ampl, dis_phi)).dag()*squeeze(n_cut, c_vec(sq_ampl, sq_phi)).dag()*fock_dm(n_cut,Fck)*squeeze(n_cut, c_vec(sq_ampl, sq_phi))*displace(n_cut, c_vec(dis_ampl, dis_phi))

        n_cut=1+int(Fck)
        
        if Ncut==None:
            while True:
                n_cut +=1
                rho_motion=ini_rho_motion(n_cut)
                pop=rho_motion.diag()[-1] 
                if pop < Prec:
                    n_cut+=1
                    rho_motion=ini_rho_motion(n_cut)
                    pop=rho_motion.diag()[-1]
                    if pop < Prec:
                        n_cut+=5
                        rho_motion=ini_rho_motion(n_cut)
                        break
        else:
            rho_motion=ini_rho_motion(Ncut)
                 
        props = self.show_state_prop(rho_motion=rho_motion, rho_spin=None, verbose=verbose)
        return rho_motion, props

    def initialise_two_modes(self, n_th=0.0, Fck=[0,0], sq_ampl=0., sq_phi=0, coupl=0,dis_ampl=1, dis_phi=0., Prec=0.01, verbose = True):
        def c_vec(ampl, phi):
            return ampl*np.exp(1j*phi)

        def ini_rho_motion(n_cut):
            rho0 = tensor(thermal_dm(n_cut, n_th), thermal_dm(n_cut, n_th))
            a_A = tensor(destroy(n_cut),qeye(n_cut))
            a_B = tensor(qeye(n_cut),destroy(n_cut))
            a_p = 1/2**0.5*(a_A+a_B)
            a_m = 1/2**0.5*(a_A-a_B)
            if coupl == 0:
                S2 = squeezing(a_A, a_B, c_vec(sq_ampl, sq_phi))
            if coupl == -1:
                S2 = squeezing(a_m, a_m, c_vec(sq_ampl, sq_phi))
            if coupl == 1:
                S2 = squeezing(a_p, a_p, c_vec(sq_ampl, sq_phi))
            return S2.dag()*a_A.dag()**Fck[0]*a_B.dag()**Fck[1]*rho0*a_A**Fck[0]*a_B**Fck[1]*S2

        n_cut=1+np.max(Fck)
        while True:
            n_cut +=1
            rho_motion=ini_rho_motion(n_cut)
            pop=rho_motion.diag()[-1] 
            if pop < Prec:
                n_cut+=1
                rho_motion=ini_rho_motion(n_cut)
                pop=rho_motion.diag()[-1]
                if pop < Prec:
                    n_cut+=2
                    rho_motion=ini_rho_motion(n_cut)
                    break

        props = self.show_state_prop(rho_motion=rho_motion, rho_spin=None, verbose=verbose)

        return rho_motion, props

    def initialise_spins(self, no=1, angle=[0,0,0], verbose=True):
        #from qutip.qip.operations import rx, ry, rz
        if no==1:
            rho_spin = (basis(2,0)*basis(2,0).dag())
            rho_spin = rz(angle[2]).dag()*ry(angle[1]).dag()*rx(angle[0]).dag()*rho_spin*rx(angle[0])*ry(angle[1])*rz(angle[2])
            props = self.show_state_prop(rho_motion=None, rho_spin=rho_spin, verbose=verbose)

        if no==2:
            rho_spin = (basis(2,0)*basis(2,0).dag())
            rho_spin = rz(angle[2]).dag()*ry(angle[1]).dag()*rx(angle[0]).dag()*rho_spin*rx(angle[0])*ry(angle[1])*rz(angle[2])
            rho_spin = tensor(rho_spin,rho_spin)
            props = self.show_state_prop(rho_motion=None, rho_spin=rho_spin, verbose=verbose)

        return rho_spin, props

    def detect_motional_state(self, output, near_dur, ptrace_sel=None, w0=None, verbose=True, dmp=False):
        times=output.times
        npts=len(times)
        tstep=times[-1]/npts
        props_l=[]
        for i in range(len(near_dur)):
            elem=int(near_dur[i]/tstep)
            rho_motion=output.states[elem]
            if verbose ==True:
                print('---------------------------------')
                print('State props. at duration: ',round(times[elem],3))
                print('---------------------------------')
            if ptrace_sel==None:
                props_l.append(self.show_state_prop(rho_motion=rho_motion, rho_spin=None, w0 = w0, verbose=verbose))
            else:
                props_l.append(self.show_state_prop(rho_motion=rho_motion, rho_spin=None, w0 = w0, verbose=verbose))
            if verbose ==True: 
                plt.show()
                if dmp==True:
                    n_c = rho_motion.dims[0][0]
                    a_A = tensor(destroy(n_c),qeye(n_c))
                    a_B = tensor(qeye(n_c),destroy(n_c))
                    a_p = 1/2**0.5*(a_A+a_B)
                    a_m = 1/2**0.5*(a_A-a_B)
                    hinton(wigner_covariance_matrix(a_A,a_B,rho=rho_motion));
                    plt.show()
        return props_l

    def detect_spin_state(self, output, near_dur, ptrace_sel=None, verbose=True, quick=False):
        times=output.times
        npts=len(times)
        tstep=times[-1]/npts
        props_l=[]
        for i in range(len(near_dur)):
            elem=int(near_dur[i]/tstep)
            if verbose ==True: 
                print('---------------------------------')
                print('State props. at duration: ',round(times[elem],3))
                print('---------------------------------')
            if ptrace_sel==None:
                props_l.append(self.show_state_prop(rho_motion=None, rho_spin=output.states[elem], verbose=verbose))
                
            else:
                props_l.append(self.show_state_prop(rho_motion=None, rho_spin=output.states[elem].ptrace(ptrace_sel), verbose=verbose))
            if verbose ==True: plt.show()
        return props_l

    def detect_motion_spin_state(self, output, near_dur, ptrace_sel=[[0],[1]], w0=None, verbose=True):
        times=output.times
        npts=len(times)
        tstep=times[-1]/npts
        props_l=[]
        rho_l=[]
        for i in range(len(near_dur)):
            elem=int(near_dur[i]/tstep)
            if verbose ==True: 
                print('---------------------------------')
                print('State props. at duration: ',round(times[elem],3))
                print('---------------------------------')
            props_l.append(self.show_state_prop(rho=output.states[elem], rho_motion=None, rho_spin=None, 
                                           ptrace_sel=ptrace_sel, w0 = w0, verbose=verbose))
            plt.show()
            rho_l.append(output.states[elem])
        props_l=np.array(props_l, dtype=object)
        return props_l, rho_l

    def trace_motional_props(self, output, ptrace_sel=None, verbose = True):
        times = output.times
        npts = len(times)
        if ptrace_sel == None:
            rho_motion_l = output.states
            no_modes = len(output.states[0].dims[0])
        else:
            rho_motion_l = []
            for i in range(npts):
                rho_motion_l.append(output.states[i].ptrace(ptrace_sel))
            no_modes = len(ptrace_sel)

        if no_modes == 1:
            prop_l = []
            for i in range(npts):
                prop_l.append(self.show_state_prop(rho_motion=rho_motion_l[i], rho_spin=None, verbose=False))
            prop_l=np.array(prop_l)
            prop_lbl = ['Fck cutoff n_cut', 
                   'Mode <n>',
                   'Var x',
                   'Var p',
                   'Entropy S', '<X>', '|<P>|']
            style_l = [[],
                ['Grey', 'solid', 3.],
                ['Blue', 'solid', 1.],
                ['Red', 'solid', 1.],
                ['Black', 'dotted', 1.],                
                ['Blue', 'dotted', 1.],
                ['Red', 'dotted', 1.],
            ]
            if verbose == True:
                fig, ax = plt.subplots(1, 1, sharex=True)
                fig.set_size_inches(4.5,2)
                for i in [1,2,3,4,5]:
                    plt.plot(times, np.real(prop_l[:,i]), color=style_l[i][0], ls=style_l[i][1], lw=style_l[i][2],label=prop_lbl[i])
                for i in [6]:
                    plt.plot(times, np.abs(np.real(prop_l[:,i])), color=style_l[i][0], ls=style_l[i][1], lw=style_l[i][2],label=prop_lbl[i])
                plt.legend(loc=(.95,.8))
                plt.xlabel('Evolution duration (us)')
                plt.ylabel('Amplitude (a.u.)')
                plt.grid(visible=True)
                plt.show()

            m_values=[]
            for i in range(len(prop_l[0])):
                m_values.append(np.max(prop_l[:,i]))
            m_values=np.array(m_values)

        if no_modes == 2:
            prop_l = []
            for i in range(npts):
                prop_l.append(self.show_state_prop(rho_motion=rho_motion_l[i], rho_spin=None, verbose=False))
            prop_l=np.array(prop_l)
            prop_lbl = ['Fck cutoff n_cut', 
                   'CoM mode <n>_+', 
                   'OoP mode <n>_-', 
                   'Total entropy S', 
                   'Mode 1 <n>_1', 
                   'Entropy S_1', 
                   'Log. Neg. E_N,1', 
                   'Mode 2 <n>_2',
                   'Entropy S_2', 
                   'Log. Neg. E_N,2']
            style_l = [[],
                ['Black', 'solid', 3.],
                ['Grey', 'solid', 3.],
                ['Black', 'dotted', 1.],
                ['Orange', 'solid', 3.],
                ['Orange', 'solid', 2.],
                ['Orange', 'solid', 1.],
                ['Magenta', 'dashed', 3.],
                ['Magenta', 'dashed', 2.],
                ['Magenta', 'dashed', 1.],
            ]
            if verbose == True:
                for i in [3,4,5,7,8]:#1,2,3,
                    plt.plot(times, prop_l[:,i], color=style_l[i][0], ls=style_l[i][1], lw=style_l[i][2],label=prop_lbl[i])
                plt.legend(loc=(.95,.4))
                plt.xlabel('Evolution duration (us)')
                plt.ylabel('Amplitude (a.u.)')
                plt.grid(visible=True)
                plt.show()

            m_values=[]
            for i in range(len(prop_l[0])):
                m_values.append(np.max(prop_l[:,i]))
            m_values=np.array(m_values)
        return prop_l, m_values

    def trace_spin_props(self, output, ptrace_sel=None, verbose = True, data=None):
        times = output.times
        npts = len(times)
        if ptrace_sel == None:
            rho_spin_l = output.states
            no_spins = len(output.states[0].dims[0])
        else:
            rho_spin_l = []
            for i in range(npts):
                rho_spin_l.append(output.states[i].ptrace(ptrace_sel))
            no_spins = len(ptrace_sel)

        if no_spins == 1:
            prop_l = []
            for i in range(npts):
                prop_l.append(self.show_state_prop(rho_motion=None, rho_spin=rho_spin_l[i], verbose=False))
            prop_l=np.array(prop_l)
            prop_lbl = ['<sx>',
                        '<sy>',
                        '<sz>',
                   'Entropy S']
            style_l = [
                ['Grey', 'solid', 2.],
                ['Orange', 'solid', 2.],
                ['Navy', 'solid', 3.],
                ['Black', 'dotted', 1.]
            ]
            if verbose == True:
                fig, ax = plt.subplots(1, 1, sharex=True)
                fig.set_size_inches(4.5,2)
                for i in [0,1,2,3]:
                    plt.plot(times, prop_l[:,i], color=style_l[i][0], ls=style_l[i][1], lw=style_l[i][2],label=prop_lbl[i])
                plt.legend(loc=(.95,.8))
                plt.xlabel('Evolution duration (us)')
                plt.ylabel('Amplitude (a.u.)')
                plt.grid(visible=True)
                plt.show()

            m_values=[]
            for i in range(len(prop_l[0])):
                #m_values.append(np.max(prop_l[:,i]))
                m_values.append(prop_l[:,i][-1])
            m_values=np.array(m_values)

        if no_spins == 2:
            prop_l = []
            for i in range(npts):
                prop_l.append(self.show_state_prop(rho_motion=None, rho_spin=rho_spin_l[i], verbose=False))
            prop_l=np.array(prop_l)
                        #props_S = [S, eof, 
                        #   s_xyz_A[0], s_xyz_A[1], s_xyz_A[2], S_A, 
                        #   s_xyz_B[0], s_xyz_B[1], s_xyz_B[2], S_B]
            prop_lbl = ['Entropy S', 
                   'EoF', 
                   '<sx>_A', '<sy>_A', '<sz>_A', 'Entropy S_A',
                   '<sx>_B', '<sy>_B', '<sz>_B', 'Entropy S_B',
                    r'P($|\downarrow\downarrow\rangle$)',r'P($|\downarrow \uparrow\rangle$)+P($|\uparrow \downarrow\rangle$)',r'P($|\uparrow \uparrow\rangle$)','Bell-state fidelity']
            style_l = [
                ['Grey', 'dashed', 1.],
                ['Red', 'dashed', 1.],
                ['Orange', 'solid', 2.],
                ['Orange', 'solid', 2.],
                ['Orange', 'solid', 3.],
                ['Orange', 'solid', 1.],
                ['Magenta', 'dashed', 2.],
                ['Magenta', 'dashed', 2.],
                ['Magenta', 'dashed', 3.],
                ['Magenta', 'dotted', 1.],
                ['Darkred', 'solid', 2.],
                ['Navy', 'solid', 2.],
                ['Orange', 'solid', 2.],
                ['Black', 'dashed', 1.],
            ]
            if verbose == True:
                for i in [10,11,12,13,0,1]:#,2,3,4,5,6,7,8,9]:
                    plt.plot(times, prop_l[:,i], color=style_l[i][0], ls=style_l[i][1], lw=style_l[i][2],label=prop_lbl[i], alpha=.75)
                plt.legend(loc=(.95,.2))
               # plt.ylim(-1,1)
                plt.xlabel('Evolution duration (us)')
                plt.ylabel('State prop. & Amp. (a.u.)')
                if data==None:
                    plt.show()
                
                if data!=None:
                    xVals, y_dd, y_du, y_uu, y_dd_e, y_du_e, y_uu_e = data
                    plt.errorbar(xVals, y_dd, y_dd_e, color='Darkred', marker ='o', linestyle='', label=r'P$(|\downarrow\downarrow\rangle)_{exp}$')
                    plt.errorbar(xVals, y_du, y_du_e, color='Navy', marker ='o', linestyle='', label=r'P$(|\downarrow \uparrow\rangle$)+P($|\uparrow \downarrow\rangle)_{exp}$')
                    plt.errorbar(xVals, y_uu, y_uu_e, color='Orange', marker ='o', linestyle='', label=r'P$(|\uparrow \uparrow\rangle)_{exp}$')
                    plt.legend(loc=(1,0))
                    #plt.xlabel(r'Scan para (a.u.)')
                    #plt.ylabel('State prop. ')
                   # plt.ylim(-0.02,1.02)
                    plt.show()
                    
            m_values=[]
            for i in range(len(prop_l[0])):
                #m_values.append(np.max(prop_l[:,i]))
                m_values.append(prop_l[:,i][-1])
            m_values=np.array(m_values)
        return prop_l, m_values

    
    def bell_state(self, No=0):
        if No==0: bell_dm = Qobj([[0.5,0.,0.,-0.5*1j],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.5*1j,0.,0.,0.5]],dims=[[2,2], [2,2]])
        if No==1: bell_dm=bell_state(state='00')*bell_state(state='00').dag()
        if No==2: bell_dm=bell_state(state='01')*bell_state(state='01').dag()
        if No==3: bell_dm=bell_state(state='10')*bell_state(state='10').dag()
        if No==4: bell_dm=bell_state(state='11')*bell_state(state='11').dag()
        return bell_dm
    
    def analyse_output(self, tlist, output, do_plot = True):
    
        N = output.states[0].dims[0][0]
        # operators
        a_1  = tensor(destroy(N), qeye(2), qeye(2))

        sm_1 = tensor(qeye(N), destroy(2), qeye(2))
        sm_2 = tensor(qeye(N), qeye(2), destroy(2))

        sz_1 = tensor(qeye(N), sigmaz(), qeye(2))
        sz_2 = tensor(qeye(N), qeye(2), sigmaz())

        S1up = []
        S2up = []
        nbars1 = []
        ns = []
        fid0 = []
        fid1 = []
        fid2 = []
        fid3 = []
        fid4 = []
        Pdd_l = []
        Pud_du_l = []
        Puu_l = []

        for rho in output.states:
            nbars1.append(np.abs(expect(a_1.dag() * a_1,rho)))
            S1up.append(np.abs(expect(sm_1.dag() * sm_1,rho)))
            S2up.append(np.abs(expect(sm_2.dag() * sm_2,rho)))
            fid0.append(fidelity(rho.ptrace([1,2]),self.bell_state(0)))
            fid1.append(fidelity(rho.ptrace([1,2]),self.bell_state(1)))
            fid2.append(fidelity(rho.ptrace([1,2]),self.bell_state(2)))
            fid3.append(fidelity(rho.ptrace([1,2]),self.bell_state(3)))
            fid4.append(fidelity(rho.ptrace([1,2]),self.bell_state(4)))
            [Pdd, Pdu, Pud, Puu] = rho.ptrace([1,2]).diag()
            Pdd_l.append(Pdd)
            Pud_du_l.append(Pdu+Pud)
            Puu_l.append(Puu)
            ns.append(rho.ptrace(0).diag())

        figures = np.array([max(fid1)])

        fid = np.array(fid1)
        elNo = np.where(fid == max(fid))[0][0]
        dur4maxFid = tlist[elNo]

        S1up = np.array(S1up)
        S2up = np.array(S2up)
        nbars1 = np.array(nbars1)
        if do_plot == True:
            print('Maximal Bell state fidelity @ dur ', round(dur4maxFid/cst.micro,3),' µs:',
                  round(figures[0],3))

            # Plot it
            fig, ax = plt.subplots(3,1,figsize=(9,12), sharex = True)
            #ax.set_xlim((0,tmax * 2 * np.pi / Omega /cst.micro))
            ax[0].set_ylim((-0.01,1.01))
            ax[0].plot(tlist/cst.micro, 1-0.5*(S1up+S2up), 
                       linewidth=3.0, color = 'grey', label = 'Fluo.')
            ax[0].plot(tlist/cst.micro, fid1, linewidth=3.0, 
                       linestyle='--', label = r'cp. to $(|\downarrow\downarrow\rangle + |\uparrow\uparrow\rangle)/\sqrt{2}$')
            ax[0].plot(tlist/cst.micro, fid3, linewidth=3.0, 
                       linestyle='--', label = r'cp. to $(|\downarrow\uparrow\rangle + |\uparrow\downarrow\rangle)/\sqrt{2}$')
            ax[0].legend(loc = 4)
            bell_state(state='11')
            ax[0].set_ylabel('Fluo., Fidelity');
            ax[1].set_ylim((-0.01,1.01))
            ax[1].plot(tlist/cst.micro, Puu_l, linewidth=3.0, 
                       color = 'red', label = r'$|\uparrow\uparrow\rangle$')
            ax[1].plot(tlist/cst.micro, Pud_du_l, linewidth=3.0, 
                       color = 'grey', label = r'$|\downarrow\uparrow\rangle + |\uparrow\downarrow\rangle$')
            ax[1].plot(tlist/cst.micro, Pdd_l, linewidth=3.0, 
                       color = 'navy', label = r'$|\downarrow\downarrow\rangle$')
            ax[1].legend(loc = 4)
            ax[1].set_ylabel('Spin Props.');
            ax[2].plot(tlist/cst.micro, nbars1, linewidth=3.0, color = 'grey')
            ax[2].set_xlabel('Interaction duration (us)')
            ax[2].plot(tlist/cst.micro, ns, linewidth=1.0, color = 'navy', alpha = 0.5)
            ax[2].set_ylabel('Motional <n> and Props.');

        details = [S1up, S2up, nbars1, fid0, fid1, fid2, fid3, fid4]

        return figures, details

    def show_final_state(self, tlist, output, dur):
        stepNo=int(len(tlist)/(tlist[-1]/cst.micro)*dur);
        print("At duration (µs):", round(tlist[stepNo]/cst.micro,2))
        rho=output.states[stepNo]
        motion1 = rho.ptrace(0);
        spins = rho.ptrace([1,2])

        print('Purity of state:',round(rho.purity(),3),'and Fidelity:',round(fidelity(spins,self.bell_state(1)),3))

        plot_wigner_fock_distribution(motion1, colorbar=True);
        #hinton(rho);

        hinton(spins, title='Spin degrees',label_top=False)
        #hinton(motion1, title='Motional degree' ,label_top=False)
        plt.show()
        return rho
    
    def wQP(self, t, args):
        """calculates and returns the modulated frequency like in "Lit early universe"

        t time at which the frequency is calculated
        args: a list {w0, dwQ, dtQ, dwP, dtP, delay} or a dictionary with the following keys:
            w0 the unmodulated frequency
            dwQ (strength) and dtQ (duration) of a gaussian shaped quench centered around t=0
            dwP (strength) and dtP (duration) of a parametric modulation of frequency 2 w0 which starts at t = delay
            dtP shoud be an integer multiple of pi/(2 w0) to avoid uncontinuity at t=delay+dtP

        units: all frequencies are circular frequencies with unit MHz, times have unit \mu s"""

        if type(args) == list:
            w0, dwQ, dtQ, wP, dwP, dtP, delay = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        elif type(args) == dict:
            w0, dwQ, dtQ, wP, dwP, dtP, delay = args['w0'], args['dwQ'], args['dtQ'], args['wP'], args['dwP'], args['dtP'], args['delay']
        else:
            return("wrong input form for args, list or dict")

        # freq += dwQ/(np.sqrt(2*np.pi)*dtQ)*np.exp(-0.5*(t/dtQ)**2)
        freq = w0 + dwQ*np.exp(-0.5*(t/dtQ)**2) # quench
        freq += dwP*np.sin(wP*(t-delay))*np.heaviside(t-delay,1)*np.heaviside(dtP-(t-delay),1) # parametric
        return(freq)

    def wQPdot(self, t, args):
        """calculates the time derivative of w(t, args) at time t
        check help(wQP) for further information on args"""
        if type(args) == list:
            w0, dwQ, dtQ, wP, dwP, dtP, delay = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        elif type(args) == dict:
            w0, dwQ, dtQ, wP, dwP, dtP, delay = args['w0'], args['dwQ'], args['dtQ'], args['dwP'], args['dwP'], args['dtP'], args['delay']
        else:
            return("wrong input form for args, list or dict")

        # freqD = - dwQ/(np.sqrt(2*np.pi)*dtQ)*np.exp(-0.5*(t/dtQ)**2) * t/(dtQ**2)
        freqD = - dwQ*np.exp(-0.5*(t/dtQ)**2) * t/(dtQ**2) # quench
        freqD += w0*dwP*np.cos(wP*(t-delay))*np.heaviside(t-delay,1)*np.heaviside(dtP-(t-delay),1) # parametric
        return(freqD)

    # defining the hamiltonian of the phonon evolution for vaiable w(t)
    def H(self, t, args):
        """calculates the hamiltonian of a harmonic oscillator with modulated frequency
        has an additional term which takes a force proportional to 1/w^2 into account

        args (dictonary which carries all arguments except t):
            t time at which the Hamiltonian is calculated (unit \mu s)
            n dimension of the hilbert space (or cutoff dimension for the numerical calculations)
            f0 proportionality constant of the additional force (unit N MHz^2)
            omega(t, omegaArgs) frequency, modulated in time, described by the list of arguments omegaArgs
            omegaDt(t, omegaArgs) time derivative of the frequency
            => in args you need: n, f0, omega, omegaDt, omegaArgs
        This form of imput is necessary to use H in further calculations (mesolve)"""

        f0 = args['f0']
        n = args['n']
        omega = args['omega']
        omegaDt = args['omegaDt']
        omegaArgs = args['omegaArgs']

        ad = create(n)
        a = destroy(n)
        # H0, for the first two terms see Silveri 2017 Quantum_systems_under_frequency_modulation
        ham = omega(t, omegaArgs)*(ad*a+0.5*qeye(n))
        # additional term because of w(t) not constant
        ham += 1j/4*omegaDt(t, omegaArgs)/omega(t, omegaArgs)*(a*a-ad*ad)
        # Force term (9**10^-9 = x0, extent of ground state wave function), see Wittmann diss
        # with compensation term -f0/w0^2 (e.g. no force in the case of no modulation)
        ham += 9*(f0/(omega(t, omegaArgs)**2) - f0/(omegaArgs[0]**2))*(ad + a)
        # ham += (9*10**-9)/(10**6)*(f0/(omega(t, omegaArgs)**2))*(ad + a)
        return(ham)

    # defining the hamiltonian of the phonon evolution for vaiable w(t) for two-coupled oscillators
    def H2osc(self, t, args):
        """calculates the hamiltonian of two-coupled oscillators with modulated frequency
        has an additional term which takes a force proportional to 1/w^2 into account

        args (dictonary which carries all arguments except t):
            t time at which the Hamiltonian is calculated (unit \mu s)
            n dimension of the hilbert space (or cutoff dimension for the numerical calculations)
            f0 proportionality constant of the additional force (unit N MHz^2)
            omega(t, omegaArgs) frequency, modulated in time, described by the list of arguments omegaArgs
            omegaDt(t, omegaArgs) time derivative of the frequency
            => in args you need: n, f0, omega, omegaDt, omegaArgs
        This form of imput is necessary to use H in further calculations (mesolve)"""
        w0 = args['w0']
        f0 = args['f0']
        n = args['n']
        omega = args['omega']
        omegaDt = args['omegaDt']
        omegaArgs = args['omegaArgs']

        wax = 2*np.pi*1.3
        OmegaEx_12 = (w0-np.sqrt(w0**2-wax**2))/2

        ad_1 = tensor(create(n), qeye(n))
        a_1 = tensor(destroy(n), qeye(n))
        ad_2 = tensor(qeye(n), create(n))
        a_2 = tensor(qeye(n), destroy(n))
        eins=tensor(qeye(n), qeye(n))

        # H0, for the first two terms see Silveri 2017 Quantum_systems_under_frequency_modulation
        ham = (omega(t, omegaArgs)-OmegaEx_12)*((ad_1*a_1+0.5*eins)+(ad_2*a_2+0.5*eins))
        #ham = (omega(t, omegaArgs))*((ad_1*a_1+0.5*eins)+(ad_2*a_2+0.5*eins))
        #coupling term
        ham += OmegaEx_12 * (ad_1*a_2 + a_1*ad_2)
        # additional term because of w(t) not constant
        ham += 1j/4*omegaDt(t, omegaArgs)/(omega(t, omegaArgs)-OmegaEx_12)*((a_1*a_1-ad_1*ad_1)+(a_2*a_2-ad_2*ad_2))
        # Force term (9**10^-9 = x0, extent of ground state wave function), see Wittemer diss
        # with compensation term -f0/w0^2 (e.g. no force in the case of no modulation)
        ham += 9*(f0/(omega(t, omegaArgs)**2) - f0/(omegaArgs[0]**2))*((ad_1 + a_1)+(ad_2 + a_2))
        # ham += (9*10**-9)/(10**6)*(f0/(omega(t, omegaArgs)**2))*(ad + a)
        return(ham)
    
    def squeeze_to_entangle_singleSpin_singleMode(self, Omega = 0.050 * 2 * np.pi,
                         omega_1 = 2.2 * 2 * np.pi,
                         omega_z = -2.2 * 2 * np.pi,
                         r_spin=[0*np.pi,0*np.pi,0*np.pi],
                         n_th = 0.001, Fck=0, 
                         sq_a =0.2, sq_phi = 0,
                         dis_a=0, dis_phi=0.,
                         phi_drive = 0,
                         tmax = 4, nosteps=1, FockPrec=0.005,
                         state_in = None, do_plot=True,LD_regime = False):
        rho=None
        eta_1 = self.lamb_dicke([0,0,1], [0,0,1], omega_1*cst.mega, 25 * cst.atomic_mass)
        tend=tmax * 2 * np.pi / Omega / eta_1
        tlist = np.linspace(0, tend, int(tend * nosteps))
        if do_plot == True:
            print('Thermal initial state with <n> = ', n_th)
            print('Sq. ampl. = ', sq_a, "phs = ", np.round(sq_phi,3))
            print('Eff. eta = ', round(eta_1 * np.sqrt(2 * n_th + 1),3))
            print('Rabi rate (2pi MHz):', round(Omega/(2*np.pi),3))
            print('Mode freq. (2pi MHz):', round(omega_1/(2*np.pi),3))
            print('Laser detunung: (2pi MHz):', round(omega_z/(2*np.pi),3))
        if state_in == None:
        # intial state
            rho_spin_0, props_s=self.initialise_spins(no=1, angle=r_spin, verbose=do_plot)
            rho_motion_0, props_m=self.initialise_single_mode(n_th=n_th, Fck=Fck, sq_ampl=sq_a, sq_phi=sq_phi, dis_ampl=dis_a, dis_phi=dis_phi, Prec=FockPrec, verbose = do_plot)
            rho_0 = tensor(rho_motion_0, rho_spin_0)
            N=props_m[0]
        else:
            rho_0 = state_in
            N=state_in.dims[0][0]

        # operators
        a  = tensor(destroy(N), qeye(2))
        sm = tensor(qeye(N), destroy(2))
        sz = tensor(qeye(N), sigmaz())

        # Hamiltonian
        
        # Hamiltonian
        if LD_regime == True:
            C = (1+1j * eta_1 * (a.dag() + a))
        else:
            C = (1j * eta_1 * (a.dag() + a) + 1j).expm()
        H = omega_z/2 * sz + omega_1 * a.dag() * a + Omega/2 * (sm.dag() * C + sm * C.dag())

        c_op_list = []

        # Solve it...
        if do_plot == True: 
            print('Solving began... ')
            output = mesolve(H, rho_0, tlist, c_op_list, [], progress_bar = True)
        else:
            output = mesolve(H, rho_0, tlist, c_op_list, [], progress_bar = None)
        if do_plot == True: 
            print('done.')
        prop_l_S, m_values_S=self.trace_spin_props(output, ptrace_sel=[1], verbose = do_plot)
        prop_l_M, m_values_M=self.trace_motional_props(output, ptrace_sel=[0], verbose = do_plot)
#        print(m_values_S,m_values_M)
#        self.detect_motion_spin_state(output, near_dur=[tlist[-2]], ptrace_sel=[[0],[1]], w0=omega_1);
        return output, m_values_S, m_values_M
    
    
    def squeeze_to_entangle_twoSpins_singleMode(self, Omega = 100 * 2 * np.pi,
                          omega_1 = 2.5 * 2 * np.pi,
                          omega_z = 1 * 2.5 * 2 * np.pi,
                          r_spin=[0*np.pi,0*np.pi,0*np.pi],
                          n_th = 0.001, Fck=0, 
                          sq_a = .8, sq_phi = 0*np.pi/16, 
                          dis_a=0, dis_phi=0.,
                          phi_drive = 0*np.pi,
                          tmax = 5, nosteps = 1, FockPrec=0.0025,
                          LD_regime = True, state_in = None, do_plot = True
                       ):
        
        eta_1 = self.lamb_dicke([0,0,1], [0,0,0.707], omega_1*cst.mega, 25 * cst.atomic_mass)
        tend=tmax * 2 * np.pi / Omega / eta_1
        tlist = np.linspace(0, tend, int(tend * nosteps))

        if do_plot == True:
            print('Thermal initial state with <n> = ', n_th)
            print('Sq. ampl. = ', sq_a)
            print('Motion EoF:', round(self.EoF_2ions(self.xi(sq_a, n_th, n_th, 2.81, 2.8)),2))
            print('Effective eta = ', round(eta_1 * np.sqrt(2 * n_th + 1),2))
            print('Rabi rate (2pi MHz):', round(Omega/(2*np.pi),2))
            print('Mode freq (2pi MHz):', round(omega_1/(2*np.pi),2))
            print('Laser detunung: (2pi MHz):', round(omega_z/(2*np.pi),2))
            progbar = True
        else:
            progbar = None
        if state_in == None:
            # intial state
            rho_spin_0, props_s=self.initialise_spins(no=2, angle=r_spin, verbose=do_plot)
            rho_motion_0, props_m=self.initialise_single_mode(n_th=n_th, Fck=Fck, 
                                                         sq_ampl=sq_a, sq_phi=sq_phi, 
                                                         dis_ampl=dis_a, dis_phi=dis_phi, 
                                                         Prec=FockPrec, verbose = do_plot)
            rho_0 = tensor(rho_motion_0, rho_spin_0)
            N=props_m[0]
        else:
            rho_0 = state_in
            N=state_in.dims[0][0]
            
        # operators
        a_1  = tensor(destroy(N), qeye(2), qeye(2))

        sm_1 = tensor(qeye(N), destroy(2), qeye(2))
        sm_2 = tensor(qeye(N), qeye(2), destroy(2))

        sz_1 = tensor(qeye(N), sigmaz(), qeye(2))
        sz_2 = tensor(qeye(N), qeye(2), sigmaz())


        # Hamiltonian
        if LD_regime == True:
            C = (1+1j * eta_1 * (a_1.dag() + a_1)+ 1j*phi_drive)
        else:
            C = (1j * eta_1/np.sqrt(2) * (a_1 + a_1.dag())+ 1j*phi_drive).expm()

        #H = omega_z/2 * (sz_1 + sz_2) + omega_1 * (a_1.dag() * a_1 + 1/2) + Omega/2 * (
        #    sm_1.dag() * C + sm_1 * C.dag() + sm_2.dag() * C + sm_2 * C.dag()
        #)
        H = omega_z/2 * (sz_1 + sz_2) + omega_1 * (a_1.dag() * a_1 + 1/2) + Omega/2 * (
            sm_1.dag() * C + sm_1 * C.dag() + sm_2.dag() * C + sm_2 * C.dag()
        )

        c_op_list = []

        # Solve it...
        output = mesolve(H, rho_0, tlist, c_op_list, [], progress_bar = progbar)
        #if do_plot == True:
        prop_l_S, m_values_S=self.trace_spin_props(output, ptrace_sel=[1,2], verbose = do_plot)
        prop_l_M, m_values_M=self.trace_motional_props(output, ptrace_sel=[0], verbose = do_plot)
        return output, m_values_S, m_values_M
    
    
    def squeeze_to_entangle_twoSpins_twoModes(self, w0=2.84, det=-1, sq_ampl=0.001, verbose=True):
        ##### ---Control parameters:
        nbar_ini=0.001;
        omega_ax = 2*np.pi*1.3
        print("Axial conf. -- tuning of coupling strength (MHz):",round(omega_ax/(2*np.pi),3))
        omega_CoM = 2*np.pi*w0 #---single ion mode freqs.
        omega_OoP = np.sqrt(omega_CoM**2-omega_ax**2)
        print("MF DoF: mode freqs. (MHz):",round(omega_CoM/(2*np.pi),3),round(omega_OoP/(2*np.pi),3))
        omega_cpl = (omega_CoM-np.sqrt(omega_CoM**2-omega_ax**2))/2 #---Coulomb coupling rate of ions
        print('Coupling strength (MHz):',round(omega_cpl/(2*np.pi),3)," (set by axial freq of",round(omega_ax/(2*np.pi),3)," MHz)")
        Omega = 2*np.pi*0.2 #---Raman laser Rabi rate
        print("Rabi rate (MHz):",round(Omega/(2*np.pi),3))
        
        omega_z = det*2*np.pi #---Spin splitting
        print("Laser detuning near (MHz):",round(omega_z/(2*np.pi),3))
        
        eta_CoM = self.lamb_dicke([0,0,1], [0,0,0.707], omega_CoM*cst.mega, 25*cst.atomic_mass)
        eta_OoP = self.lamb_dicke([0,0,1], [0,0,0.707], omega_OoP*cst.mega, 25*cst.atomic_mass)
        print(eta_CoM,eta_OoP)
        #---Initial states:
        psi_ext_ini, props = self.initialise_two_modes(n_th=0.0, Fck=[0,0], 
                                                  sq_ampl=sq_ampl, sq_phi=0., coupl=0, 
                                                  dis_ampl=0., dis_phi=0., Prec=0.01, 
                                                  verbose = verbose)
        n_fck=props[0]

     
        #psi_ext_ini = S2.dag()*psi0*S2
        #---
        #psi_ext_ini = tensor(thermal_dm(n_fck, 0.001), thermal_dm(n_fck, 0.001))
        psi_int_ini = tensor((basis(2,0) * basis(2,0).dag()), (basis(2,0) * basis(2,0).dag()))
        psi_ini = tensor(psi_ext_ini, psi_int_ini)

        #---Define (normal modes+spins) operators:
        a_p  = tensor(destroy(n_fck), qeye(n_fck), qeye(2), qeye(2))
        a_m  = tensor(qeye(n_fck), destroy(n_fck), qeye(2), qeye(2))
        
        sz_A = tensor(qeye(n_fck), qeye(n_fck), sigmaz(), qeye(2))
        sm_A = tensor(qeye(n_fck), qeye(n_fck), destroy(2), qeye(2))
        
        sz_B = tensor(qeye(n_fck), qeye(n_fck), qeye(2), sigmaz())
        sm_B = tensor(qeye(n_fck), qeye(n_fck), qeye(2), destroy(2))
        
#         # operators
#        a  = tensor(destroy(N), qeye(2))
#        sm = tensor(qeye(N), destroy(2))
#        sz = tensor(qeye(N), sigmaz())
#
#        # Hamiltonian
#        C = (1j * eta_1 * (a.dag() + a) + 1j*phi_drive).expm()
#        H = omega_z/2 * sz + omega_1 * a.dag() * a + Omega/2 * (sm.dag() * C + sm * C.dag())

        
        
        #---Hamiltonians:
        #---Internal DoF (Ion A and B)
        Hint = omega_z/2 * (sz_A + sz_B)

        #---External DoF (Ion A and B)
        #Each ion
        Hext = omega_CoM * (a_p.dag()*a_p + 1/2) + omega_OoP * (a_m.dag()*a_m + 1/2)
        #Ion C. couplings
        #Hext += (omega_CoM - omega_cpl) * (a_A.dag()*a_A + a_B.dag()*a_B) + omega_cpl*(a_A.dag()*a_B + a_A*a_B.dag())

        #---Laser couplings (Ion A and B)
        #C = 1+1j * eta_CoM * (a_p.dag() + a_p) + 1j * eta_OoP * (a_m.dag() + a_m)
        C = (1j * eta_CoM * (a_p.dag() + a_p) + 1j * eta_OoP * (a_m.dag() + a_m)).expm()
        #C_A = (1j * eta_CoM * (a_A.dag() + a_A)).expm()
        #C_B = (1j * eta_CoM * (a_B.dag() + a_B)).expm()
        #C = (1j * eta_CoM * (a_p.dag() + a_p) + 1j * eta_OoP * (a_m.dag() + a_m)).expm()
        #C += ().expm()
        
        #HiAct_A = Omega/2*(sm_A.dag()*C + sm_A*C.dag())
        #HiAct_B = Omega/2*(sm_B.dag()*C + sm_B*C.dag())
        HiAct=Omega/2*(sm_A.dag()*C + sm_A*C.dag() + sm_B.dag()*C + sm_B*C.dag())
        #HiAct = Omega/2*((sm_A.dag() + sm_B.dag())*C + (sm_A + sm_B)*C.dag())

        H = [Hint, Hext, HiAct]

        tmax = 1
        tend=tmax * 2 * np.pi / Omega / eta_CoM
        tlist = np.linspace(0, tend, 100)

        if verbose == False:
            progbar=None
        else:
            progbar=True

        if verbose ==True: print('Start solving...')
        output = mesolve(H, psi_ini, tlist, [], [], progress_bar = progbar)
        if verbose ==True: print('...done.')

        prop_l_S, m_values_S=self.trace_spin_props(output, ptrace_sel=[2,3], verbose = verbose)
        prop_l_M, m_values_M=self.trace_motional_props(output, ptrace_sel=[0,1], verbose = verbose)

        return output, m_values_S, m_values_M

    
    def get_basis_fct(self, n_cut,state):
        from joblib import Parallel, delayed
        import multiprocessing

        inputs = np.linspace(0,n_cut,n_cut+1, dtype=int)
        def processInput(i):
            output, _, _ = self.squeeze_to_entangle_twoSpins_singleMode(Omega = .13 * 2 * np.pi,
                                 omega_1 = 2.75 * 2 * np.pi,
                                 omega_z = -1*2.75 * 2 * np.pi,
                                 r_spin=[state*np.pi,0.*np.pi,0.*np.pi],
                                 n_th = 0.0, Fck=i,
                                 sq_a = 0.0, sq_phi = -0.0*np.pi,
                                 dis_a = 0.0, dis_phi=0.0*np.pi,
                                 tmax = 2.2, nosteps = 3., FockPrec=0.1,
                                 LD_regime = False, do_plot = False)
            fname='fck_'+str(state)+'_'+str(i)
            qsave(output, fname)
            print('Fock state #:',i,' done.')
            return output

        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
        return results

    def load_basis_fct(self, n_cut,state):
        from joblib import Parallel, delayed
        import multiprocessing
        inputs = np.linspace(0,n_cut,n_cut+1, dtype=int)
        def processInput(i):
            fname='fck_'+str(state)+'_'+str(i)
            output=qload(fname)
            return output
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
        return results

    def get_flops_for_each_Fck(self, basis_fct, Fck=0, dur=0, verbose=False):
        #ret=qc.detect_spin_state(basis_fct[Fck], near_dur=[dur], ptrace_sel=[1,2], verbose=False)
        #ret=np.array(ret[0])[10:13]
        output=basis_fct[Fck]
        times=output.times
        npts=len(times)
        tstep=times[-1]/npts
        elem=int(dur/tstep)    
        diag=output.states[elem].ptrace([1,2]).diag()
        ret=np.array([diag[0],diag[1]+diag[2],diag[3]])
        if verbose==True: print('Pdd, Puddu, Puu:',ret)
        return ret

    def summed_fck_states(self, fck_pop, basis_fct, dur, dur_scl):
        summed=0
        nfck=np.min([len(basis_fct),len(fck_pop)])
        #print(nfck)
        for i in np.linspace(0,nfck-1, nfck,dtype=int):
            summed+=fck_pop[i]*self.get_flops_for_each_Fck(basis_fct, Fck=i, dur=dur_scl*dur, verbose=False)
        return summed

    def get_fit_fct(self, data, rho_m, basis_fct, dur_scl):
        xVals, y_dd, y_du, y_uu, y_dd_e, y_du_e, y_uu_e = data
        diag=rho_m.diag()

        times=xVals
        Pdd=[]
        Puddu=[]
        Puu=[]

        for durs in times:
            ret=self.summed_fck_states(diag, basis_fct,durs, dur_scl)
            Pdd.append(ret[0])
            Puddu.append(ret[1])
            Puu.append(ret[2])
        Pdd=np.array(Pdd)
        Puddu=np.array(Puddu)
        Puu=np.array(Puu)
        return times, Pdd, Puddu, Puu

    def plot_two_ions_sim_vs_exp(self, basis_fct, model_par, data=[], model_err=None):
        import itertools


        [dur_scl,n_th,sq_ampl,dis_ampl]=model_par
        rho_m, _=self.initialise_single_mode(n_th=n_th,sq_ampl=sq_ampl,dis_ampl=dis_ampl, verbose=True);
        times, Pdd, Puddu, Puu=self.get_fit_fct(data, rho_m, basis_fct, dur_scl=dur_scl)
        xVals, y_dd, y_du, y_uu, y_dd_e, y_du_e, y_uu_e = data

        fig, ax = plt.subplots(3, 1, figsize=(6,9), gridspec_kw = {'height_ratios':[3,3,3]}, sharex=True);

        if model_err != None:
            from joblib import Parallel, delayed
            import multiprocessing
            inputs = np.linspace(0,50,50+1, dtype=int)
            def processInput(i):
                sampl_model_par=[]
                for par in range(len(model_par)):
                    sampl_model_par.append(np.random.normal(model_par[par], model_err[par]))

                #print(sampl_model_par)
                [dur_scl,n_th,sq_ampl,dis_ampl]=sampl_model_par
                rho_m, _=self.initialise_single_mode(n_th=n_th,sq_ampl=sq_ampl,dis_ampl=dis_ampl, verbose=False);
                times, Pdd, Puddu, Puu=self.get_fit_fct(data, rho_m, basis_fct,dur_scl=dur_scl)
                #ax[0].plot(times,Pdd+Puddu+Puu, color='Black', lw=1, alpha=0.02)
                x = times
                y = Pdd
                new_x, new_y = zip(*sorted(zip(x, y)))
                ax[0].plot(new_x, new_y, color='Darkred', lw=5, alpha=0.025, marker ='', linestyle='-')
                y = Puddu
                new_x, new_y = zip(*sorted(zip(x, y)))
                ax[1].plot(new_x, new_y, color='Navy', lw=5, alpha=0.025, marker ='', linestyle='-')
                y = Puu
                new_x, new_y = zip(*sorted(zip(x, y)))
                ax[2].plot(new_x, new_y, color='Orange', lw=5, alpha=0.025, marker ='', linestyle='-')
                
            num_cores = 1#multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
        
        [dur_scl,n_th,sq_ampl,dis_ampl]=model_par
        rho_m, _=self.initialise_single_mode(n_th=n_th,sq_ampl=sq_ampl,dis_ampl=dis_ampl, verbose=False);
        times, Pdd, Puddu, Puu=self.get_fit_fct(data, rho_m, basis_fct, dur_scl=dur_scl)
        xVals, y_dd, y_du, y_uu, y_dd_e, y_du_e, y_uu_e = data
        #ax[0].plot(times,Pdd+Puddu+Puu, color='Black', lw=1, alpha=0.6, label='Tot. population:'+str(round(np.mean(Pdd+Puddu+Puu),2)))

        x = times
        y = Pdd
        new_x, new_y = zip(*sorted(zip(x, y)))
        ax[0].plot(new_x, new_y, color='Darkred', lw=2, alpha=0.68, marker ='', linestyle='-', label=r'P$(|\downarrow\downarrow\rangle)_{sim}$')

        x = times
        y = Puddu
        new_x, new_y = zip(*sorted(zip(x, y)))
        ax[1].plot(new_x, new_y, color='Navy', lw=2, alpha=0.68, marker ='', linestyle='-', label=r'P$(|\downarrow \uparrow\rangle$)+P($|\uparrow \downarrow\rangle)_{sim}$')

        x = times
        y = Puu
        new_x, new_y = zip(*sorted(zip(x, y)))
        ax[2].plot(new_x, new_y, color='Orange', lw=2, alpha=0.68, marker ='', linestyle='-', label=r'P$(|\uparrow \uparrow\rangle)_{sim}$')
        ax[0].errorbar(xVals, y_dd, y_dd_e, color='Darkred', marker ='o', markeredgewidth=1., markeredgecolor='black', linestyle='', label=r'P$(|\downarrow\downarrow\rangle)_{exp}$')
        ax[1].errorbar(xVals, y_du, y_du_e, color='Navy', marker ='o', markeredgewidth=1., markeredgecolor='black', linestyle='', label=r'P$(|\downarrow \uparrow\rangle$)+P($|\uparrow \downarrow\rangle)_{exp}$')
        ax[2].errorbar(xVals, y_uu, y_uu_e, color='Orange', marker ='o', markeredgewidth=1., markeredgecolor='black', linestyle='', label=r'P$(|\uparrow \uparrow\rangle)_{exp}$')
        for i in [0,1,2]:
            #ax[i].legend(loc=(1,0))
            
            ax[i].set_ylim(-0.02,1.02)
        ax[1].set_ylabel('State prop. ')    

        ax[2].set_xlabel(r'Flop dur. (µs)')
 
        plt.show()
        
        

    def single_spin_and_mode_ACpi2(self, spin_ini=[0,0], mode_para=[0.001, [0, 0], [0, 0], 1.3], theta_m = 0, drive_para=[0.125, 0], mod_on=False, mod_type=0, mod_fac=1, mod_amp=1, strobo_dur=.1, do_plot=False):
        from scipy.signal import butter, filtfilt
        from scipy import signal

        #Mode and drive parameters
        omega_1 = 2*np.pi*mode_para[3]
        omega_z = drive_para[1]*omega_1
        Omega = drive_para[0] * 2 * np.pi
        
        
        s_ini=spin_ini[0]
        s_phs=spin_ini[1]
        r_spin=[s_ini*2*np.pi,0.*2*np.pi,(0.677/2+s_phs)*2*np.pi]

        # Initial state

        n_th = mode_para[0]
        Fck = 0 
        dis_a = mode_para[1][0]
        dis_phs = mode_para[1][1]*2*np.pi 
        sq_a = mode_para[2][0]
        sq_phs = mode_para[2][1]*2*np.pi 

        Degree=np.pi/180

        eta_1 = self.lamb_dicke([0,-np.sqrt(2)/np.sqrt(2),np.sqrt(2)/np.sqrt(2)], [0, -np.sin(theta_m*Degree), np.cos(theta_m*Degree)], omega_1*cst.mega, 25 * cst.atomic_mass)
        #eta_1 = self.lamb_dicke([0,0,1], [0, -np.sin(theta_m*Degree), np.cos(theta_m*Degree)], omega_1*cst.mega, 25 * cst.atomic_mass)
        Omega_eff = np.exp(-eta_1**2/2) * Omega
        if do_plot==True:
            print('##########    Settings:      #########')
            print('Mode freq. (2pi MHz) = ', round(omega_1/(2*np.pi),3))
            print('Excitation:\n\tThermal <n> = ', n_th)
            print('\tDisplaced [amp, phs] = ', [dis_a, dis_phs])
            print('\tSqueezed [amp, phs] = ', [sq_a, sq_phs])
            print('Eff. eta: ', round(eta_1 * np.sqrt(2 * n_th + 1),3))
            print('Rabi rate (2pi MHz) = ', round(Omega/(2*np.pi),3))
            
            print('Eff. carrier Rabi rate (2pi MHz):', round(Omega_eff/(2*np.pi),3))
            print('Laser detunung: (2pi MHz) =', round(omega_z/(2*np.pi),3))
            print('\n')
            print('##########   Initial state   ##########')

        rho_spin_0, props_s=self.initialise_spins(no=1, angle=r_spin, verbose=do_plot)
        rho_motion_0, props_m=self.initialise_single_mode(n_th=n_th, Fck=Fck, 
                                                     sq_ampl=sq_a, sq_phi=sq_phs, 
                                                     dis_ampl=dis_a, dis_phi=dis_phs, 
                                                     Prec=10**(-10), Ncut=int(2*dis_a**2+10), verbose = do_plot)
        N=props_m[0]
        rho_0 = qutip.tensor(rho_motion_0, rho_spin_0)

        # operators
        a  = qutip.tensor(qutip.destroy(N), qutip.qeye(2))
        sm = qutip.tensor(qutip.qeye(N), qutip.destroy(2))
        sz = qutip.tensor(qutip.qeye(N), qutip.sigmaz())

        # Hamiltonian

        H0 = omega_z/2 * sz + omega_1 * a.dag() * a

        C = (1j * eta_1 * (a.dag() + a) + 1j).expm()
        #C = (1+1j * eta_1 * (a.dag() + a))
        HI=Omega/2 * (sm.dag() * C + sm * C.dag())

        def mode_osc(t):
            return -np.cos(omega_1*t+dis_phs)

        if mod_on==True:
            omega_mod=mod_fac*omega_1
            duty_cycle = strobo_dur/(2*np.pi/omega_mod)
            if mod_type==0:
                tend=0.925*2*np.pi/Omega_eff/4/duty_cycle
            if mod_type==1:
                tend=2*np.pi/Omega_eff/4*2 
            tlist = np.linspace(0, tend, int(tend * 200))
            if do_plot==True:
                print('##########   AC drive modulation   ##########')
                if mod_type==0:
                    print("Modulation: Stroboscopic")
                    print("Pulse duration approx. (µs): ",np.round(2*np.pi/omega_mod*duty_cycle,3))
                if mod_type==1:
                    print("Modulation: Sinusoidal")

            def mod_fct(t):
                t_off=2*np.pi/omega_mod/4-strobo_dur/2
                return (signal.square(omega_mod*(t-t_off), duty=duty_cycle)+1)/2

            def aom_filter(data, cutoff, fs, order):
                #define empiric filter fct of AOM, given by the finite speed of sound
                nyq=3
                normal_cutoff = cutoff / nyq
                # Get the filter coefficients 
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                y = filtfilt(b, a, data)
                return y

            def mod_fct_fil(t):
                f_sampl = 1/(tend/len(tlist)) # sample rate
                f_order = 1
                f_cutoff = 0.2
                return np.abs(aom_filter(mod_fct(t), f_cutoff, f_sampl, f_order))
                #return np.abs(mod_fct(t))

            def spline_mod_fct_fil(t):
                from scipy.interpolate import CubicSpline

                # Define some sample points
                x = tlist
                y = mod_fct_fil(tlist)

                # Create a cubic spline interpolation function
                cs = CubicSpline(x, y)

                return cs(t)

            def mod(t, args):
                if mod_type==0:
                    return spline_mod_fct_fil(t)
                if mod_type==1:
                    return (1-mod_amp*np.cos(omega_mod*t))/2

            if do_plot==True:
                fig, ax = plt.subplots(1, 1, sharex=True)
                fig.set_size_inches(4.5,2)
                if mod_type==0:
                    plt.plot(tlist, mod_fct(tlist), color='navy', ls='--')
                    #plt.plot(tlist, mod_fct_fil(tlist), color='navy')
                plt.plot(tlist, mod(tlist, []), color='navy', label='AC Drive')
                plt.plot(tlist, mode_osc(tlist), color='red', label='Mode')
                plt.xlabel('Evolution duration (us)')
                plt.ylabel('Ampl. (a.u.)')
                plt.legend(loc=(.95,.2))
                plt.grid(visible=True)
                plt.xlim(0,tend/7)
                plt.show()
                          
                # Generate a signal
                freq_ac=1776
                fs = 200000  # Sampling frequency
                t = np.arange(0, tend, 1/fs)  # Time vector
                f = 5  # Signal frequency
                x = mod(t, [])*np.sin(2*np.pi*freq_ac*t)  # Signal

                # Perform FFT
                X = np.fft.fft(x)

                # Compute frequency vector
                freq = np.fft.fftfreq(len(x), d=1/fs)

                # Plot signal and its FFT
                fig, (ax1, ax2) = plt.subplots(2, 1)

                ax1.plot(t, x)
                ax1.set_xlabel('Time (µs)')
                ax1.set_ylabel('Amplitude')
                ax1.set_title('AC Drive')

                
                ax2.set_xlabel('Frequency detuning from AC carrier (MHz)')
                ax2.set_ylabel('Magnitude (arb. units)')
                #ax2.vlines(freq_ac,0,1, lw=4., color='black', alpha=.6)
                for n in [1,2,3,4,5,6,7,8,9]:
                    ax2.vlines(n*mode_para[3],0,1, lw=4., color='navy', alpha=eta_1**n)
                    ax2.vlines(-n*mode_para[3],0,1, lw=4., color='red', alpha=eta_1**n)
                ax2.plot(freq-freq_ac, np.abs(X)/np.max(np.abs(X)), color='black', ls='', marker='o', ms=1.)
                ax2.set_xlim(-5, +5)
                ax2.set_title('FFT of AC drive')

                plt.tight_layout()
                plt.show()
            
            H=[H0,[HI,mod]]

        if mod_on==False:
            tend=2*np.pi/Omega_eff/4 #pi/2 duration
            tlist = np.linspace(0, tend, int(tend * 200))
            H=H0+HI
            def mod(t, args):
                return tlist*0+1

        c_op_list = []

        if do_plot==True:
            print('##########   Spin-mode dynamics   ##########')
        output = qutip.mesolve(H, rho_0, tlist, c_op_list, [], progress_bar = None)
        prop_l_S, m_values_S=self.trace_spin_props(output, ptrace_sel=[1], verbose = do_plot)
        prop_l_M, m_values_M=self.trace_motional_props(output, ptrace_sel=[0], verbose = do_plot)
        if do_plot==True:
            fig, ax = plt.subplots(3, 1, sharex=True)
            fig.set_size_inches(6,4)
            ax[1].plot(tlist, (prop_l_S[:,2]+1)/2, color='Black', ls='-',label='P_down')
            if mod_type==0 and mod_on==True:
                ax[0].plot(tlist, mod_fct(tlist), color='lightgrey', ls='-', alpha=1)
                ax[1].plot(tlist, mod_fct(tlist), color='lightgrey', ls='-', alpha=1)
                ax[2].plot(tlist, np.max(prop_l_M[:,-1])*mod_fct(tlist), color='lightgrey', ls='-', alpha=1)
                #plt.plot(tlist, mod_fct_fil(tlist), color='navy')
            ax[0].plot(tlist, mod(tlist, []), color='navy', label='AC Drive')
            ax[2].plot(tlist, prop_l_M[:,-2], color='red', label='Mode <X>')
            ax[2].plot(tlist, np.abs(prop_l_M[:,-1]), color='Blue', label='Mode |<P>|')
            ax[2].plot(tlist, prop_l_M[:,1]-prop_l_M[1,1], color='grey', label='Back action - Delta<n>')
            ax[2].set_xlabel('Evolution duration (us)')
            ax[1].set_ylabel('Ampl. (a.u.)')
            #ax[1].set_ylim(0.25,.55)
            ax[2].set_xlim([0,tend/5])
            ax[2].set_xlim([4*tend/5,tend])

            ax[0].legend(loc=(.95,.2))
            ax[1].legend(loc=(.95,.2))
            ax[2].legend(loc=(.95,.2))
            plt.show()


            print('##########   Final state   ##########')
            self.detect_motion_spin_state(output, near_dur=[tlist[-2]], ptrace_sel=[[0],[1]], w0=omega_1, verbose=do_plot)
            b_action=((prop_l_M[-1:,1]-prop_l_M[0,1]))[0]
            print('##########   Back action  ##########')
            print('Amount of back action <n>_back=<n>_fin-<n>_ini:',np.round(b_action,3))
            print('rel. back action (<n>_fin - <n>_ini)/<n>_ini:', np.round(b_action/prop_l_M[0,1],3))
        return output