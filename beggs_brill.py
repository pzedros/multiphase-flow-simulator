import numpy as np

#Balanço de massa em escoamentos:

"""Vazões superficie"""

def vazaosc(Vlsc, RGL, BSW):

    Vosc = Vlsc * (1 - BSW)

    Vwsc = Vlsc * BSW

    Vgsc = RGL * Vlsc

    return Vosc, Vwsc, Vgsc

"""Vazões insitu"""

def vazaoinsitu(Vosc, Vwsc, Vgsc, Bo, Bg, Bw, Rs, Rsw):

    Vl = Vosc*Bo + Vwsc*Bw

    Vg = (Vgsc - (Vosc*Rs) - (Vwsc*Rsw)) * Bg

    return Vl, Vg

"""Propriedades do líquido """

def prop_liquid(pho_o, pho_w, Vosc, Vwsc, Bo, Bw, Vl, mu_o, mu_w, tensao_og, tensao_wg ):

    Fo = (Vosc*Bo)/Vl

    Fwc = (Vwsc*Bw)/Vl

    pho_l = (pho_o * Fo) + (pho_w * Fwc)

    mu_l = (mu_o * Fo) + (mu_w * Fwc)

    tensao_lg = (tensao_og * Fo) + (tensao_wg * Fwc)

    return Fo, Fwc, pho_l, mu_l, tensao_lg


"MOdelo de Beggs & Brill"

#Holdup No Slip
def holdup(Vl, Vg, Ap):

    Vsl = Vl / Ap

    Vsg = Vg / Ap

    Vm = Vsl + Vsg

    holdup_L = Vsl / Vm

    holdup_G = Vsg / Vm

    return holdup_L, holdup_G, Vsl, Vsg, Vm

def Froude(Vm, Dh, g):

    #Frm² = (Vm**2) / (g*Dh)
    Frm = (Vm ** 2) / (g * Dh)
    return Frm

"""Determinação do padrão de escoamento"""

#Parâmetros 𝐿1 𝑎 𝐿4
def param_L(holdup_L):
    L1 = 316 * holdup_L**0.302
    L2 = 0.0009252 * holdup_L**-2.4684
    L3 = 0.10 * holdup_L**-1.4516
    L4 = 0.5 * holdup_L**-6.738

    return L1, L2, L3, L4

def padrao_escoamento(L1, L2, L3, L4, holdup_L, Frm):
    # Distribuido
    if (holdup_L < 0.4 and Frm >= L1) or (holdup_L >= 0.4 and Frm > L4):
        flag=1

    #Segragado
    if (holdup_L < 0.01 and Frm < L1) or (holdup_L >= 0.001 and Frm < L2):
        flag = 2

    #Transição
    if (holdup_L >= 0.001 and L2 <= Frm <= L3 ):
        flag = 3

    #Intermitente
    if (0.01 <= holdup_L < 0.4 and L3 <= Frm <= L1) or (holdup_L >= 0.4 and L3 <= Frm <= L4 ):
        flag = 4

    return flag

def HLO(flag, holdup_L, Frm, L2, L3):

    if flag == 1:
        HlO = (1.065 * holdup_L**0.5824) / (Frm**0.0609)

    elif flag == 2:
        HlO = (0.98 * holdup_L**0.4846) / (Frm**0.0868)

    elif flag == 3:
        # Interpolação linear entre segregado e intermitente (Beggs & Brill, 1973)
        A = (L3 - Frm) / (L3 - L2)
        HlO_seg = (0.98 * holdup_L**0.4846) / (Frm**0.0868)
        HlO_int = (0.845 * holdup_L**0.5351) / (Frm**0.0173)
        HlO = A * HlO_seg + (1 - A) * HlO_int

    elif flag == 4:
        HlO = (0.845 * holdup_L**0.5351) / (Frm**0.0173)

    return HlO

