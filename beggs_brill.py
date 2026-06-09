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


def numero_velocidade_liquido(Vsl, pho_l, g, tensao_lg):
    """N_LV = Vsl * (rho_L / (g * sigma_GL))^(1/4)"""
    NLV = Vsl * (pho_l / (g * tensao_lg)) ** 0.25
    return NLV


def psi_inclinacao(flag, holdup_L, Frm, NLV, theta, L2, L3):
    """
    Parâmetro de correção da inclinação psi.
    theta: ângulo em graus (positivo = ascendente, negativo = descendente)
    Frm: número de Froude modificado (Vm²/g·d)
    """
    if theta >= 0:  # ascendente
        if flag == 1:  # Distribuído: sem correção
            C = 0.0

        elif flag == 2:  # Segregado
            d, e, f, g = 0.011, -3.768, 3.539, -1.614
            C = (1 - holdup_L) * np.log(d * holdup_L**e * NLV**f * Frm**g)
            C = max(C, 0.0)

        elif flag == 3:  # Transição — interpola C entre segregado e intermitente
            d_s, e_s, f_s, g_s = 0.011, -3.768, 3.539, -1.614
            d_i, e_i, f_i, g_i = 2.960, 0.305, -0.4473, 0.0978
            C_seg = max((1 - holdup_L) * np.log(d_s * holdup_L**e_s * NLV**f_s * Frm**g_s), 0.0)
            C_int = max((1 - holdup_L) * np.log(d_i * holdup_L**e_i * NLV**f_i * Frm**g_i), 0.0)
            A = (L3 - Frm) / (L3 - L2)
            C = A * C_seg + (1 - A) * C_int

        elif flag == 4:  # Intermitente
            d, e, f, g = 2.960, 0.305, -0.4473, 0.0978
            C = (1 - holdup_L) * np.log(d * holdup_L**e * NLV**f * Frm**g)
            C = max(C, 0.0)

    else:  # descendente — mesmos coeficientes para todos os padrões
        d, e, f, g = 4.700, -0.3692, 0.1244, -0.5056
        C = (1 - holdup_L) * np.log(d * holdup_L**e * NLV**f * Frm**g)
        C = max(C, 0.0)

    theta_rad = np.radians(1.8 * theta)
    psi = 1 + C * (np.sin(theta_rad) - 0.333 * np.sin(theta_rad)**3)
    return C, psi


def holdup_liquido(HlO, psi, holdup_L):
    """H_L = H_LO * psi, limitado entre lambda_L e 1"""
    HL = HlO * psi
    HL = max(HL, holdup_L)  # HL nunca menor que o no-slip
    HL = min(HL, 1.0)
    return HL
