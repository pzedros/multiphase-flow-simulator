import numpy as np
from fontTools.misc.cython import returns
from fontTools.ttLib.tables.otTables import VarRegionList
from pandas.core.apply import Apply

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

def HLO(flag, holdup_L, Frm):
    """Holdup horizontal (H_LO) — apenas flags 1, 2, 4.
    Para transição (flag 3) use HL_transicao()."""
    if flag == 1:    # Distribuído
        HlO = (1.065 * holdup_L**0.5824) / (Frm**0.0609)
    elif flag == 2:  # Segregado
        HlO = (0.98 * holdup_L**0.4846) / (Frm**0.0868)
    elif flag == 4:  # Intermitente
        HlO = (0.845 * holdup_L**0.5351) / (Frm**0.0173)

    # Condição B&B: H_LO >= lambda_L
    return max(HlO, holdup_L)


def numero_velocidade_liquido(Vsl, pho_l, g, tensao_lg):
    """N_LV = Vsl * (rho_L / (g * sigma_GL))^(1/4)"""
    NLV = Vsl * (pho_l / (g * tensao_lg)) ** 0.25
    return NLV


def psi_inclinacao(flag, holdup_L, Frm, NLV, theta):
    """Parâmetro de correção da inclinação.
    ψ = 1 + C · {sin(1,8θ) - 0,333 · [sin(1,8θ)]³}
    C = (1 - λ_L) · ln(d' · λ_L^e · N_LV^f · (Fr_m²)^g)
    theta em graus (positivo = ascendente, negativo = descendente).
    Apenas para flags 1, 2, 4. Para transição use HL_transicao()."""
    if theta >= 0:  # ascendente
        if flag == 1:  # Distribuído: sem correção de inclinação
            C = 0.0
        elif flag == 2:  # Segregado
            d, e, f, g = 0.011, -3.768, 3.539, -1.614
            C = max((1 - holdup_L) * np.log(d * holdup_L**e * NLV**f * Frm**g), 0.0)
        elif flag == 4:  # Intermitente
            d, e, f, g = 2.960, 0.305, -0.4473, 0.0978
            C = max((1 - holdup_L) * np.log(d * holdup_L**e * NLV**f * Frm**g), 0.0)
    else:  # descendente — mesmos coeficientes para todos os padrões
        d, e, f, g = 4.700, -0.3692, 0.1244, -0.5056
        C = max((1 - holdup_L) * np.log(d * holdup_L**e * NLV**f * Frm**g), 0.0)

    sin_1_8_theta = np.sin(np.radians(1.8 * theta))
    psi = 1 + C * (sin_1_8_theta - 0.333 * sin_1_8_theta**3)
    return C, psi


def holdup_liquido(HlO, psi, holdup_L):
    """H_L = H_LO · ψ quando H_LO > λ_L, senão H_L = λ_L"""
    if HlO > holdup_L:
        HL = HlO * psi
    else:
        HL = holdup_L
    return min(HL, 1.0)


def HL_transicao(holdup_L, Frm, NLV, theta, L2, L3):
    """Holdup real para padrão de transição (B&B).
    H_L(trans) = A · H_L(Seg) + (1 - A) · H_L(Interm)
    A = (L3 - Fr_m²) / (L3- L2)
    A interpolação é feita sobre H_L (com ψ já aplicado), não sobre H_LO."""
    A = (L3 - Frm) / (L3 - L2)

    HlO_seg = HLO(2, holdup_L, Frm)
    _, psi_seg = psi_inclinacao(2, holdup_L, Frm, NLV, theta)
    HL_seg = holdup_liquido(HlO_seg, psi_seg, holdup_L)

    HlO_int = HLO(4, holdup_L, Frm)
    _, psi_int = psi_inclinacao(4, holdup_L, Frm, NLV, theta)
    HL_int = holdup_liquido(HlO_int, psi_int, holdup_L)

    return A * HL_seg + (1 - A) * HL_int


"""Gradiente de pressão — Beggs & Brill"""

def densidade_noslip(pho_l, pho_g, holdup_L):
    """Densidade No-Slip da mistura:
    rho_NS = rho_L * lambda_L + rho_g * (1 - lambda_L)"""
    pho_NS = pho_l * holdup_L + pho_g * (1 - holdup_L)
    return pho_NS


def densidade_slip(pho_l, pho_g, HL):
    """Densidade Slip da mistura (usa holdup real H_L):
    rho_SLIP = rho_L * H_L + rho_g * (1 - H_L)"""
    pho_slip = pho_l * HL + pho_g * (1 - HL)
    return pho_slip


def viscosidade_noslip(mu_l, mu_g, holdup_L):
    """Viscosidade No-Slip da mistura:
    mu_NS = mu_L * lambda_L + mu_g * (1 - lambda_L)"""
    mu_NS = mu_l * holdup_L + mu_g * (1 - holdup_L)
    return mu_NS


def reynolds_noslip(pho_NS, Vm, Dh, mu_NS):
    """Número de Reynolds No-Slip:
    Re_NS = rho_NS * Vm * Dh / mu_NS"""
    Re_NS = (pho_NS * Vm * Dh) / mu_NS
    return Re_NS


def fator_atrito_noslip(Re_NS, eps, Dh):
    """Fator de atrito No-Slip — Correlação de Hall:
    fN = 0.0055 * [1 + (2e4 * eps/Dh + 1e6/Re_NS)^(1/3)]"""
    fN = 0.0055 * (1 + (2e4 * (eps / Dh) + 1e6 / Re_NS) ** (1/3))
    return fN


def parametro_s(holdup_L, HL):
    """Parâmetro s para correção do fator de atrito bifásico.
    y = lambda_L / H_L²
    Se 1.0 < y < 1.2:  s = ln(2.2*y - 1.2)
    Fora desse intervalo: s = ln(y) / (-0.0523 + 3.182*ln(y) - 0.8725*ln(y)² + 0.01853*ln(y)^4)
    """
    y = holdup_L / HL**2

    if 1.0 < y <= 1.2:
        s = np.log(2.2 * y - 1.2)
    else:
        ln_y = np.log(y)
        s = ln_y / (-0.0523 + 3.182 * ln_y - 0.8725 * ln_y**2 + 0.01853 * ln_y**4)

    return s


def fator_atrito_bifasico(fN, s):
    """Fator de atrito bifásico:
    fTP = e^s * fN"""
    fTP = np.exp(s) * fN
    return fTP


def gradiente_friccao(fTP, pho_NS, Vm, Dh):
    """Gradiente de pressão por fricção:
    dP/dL|_F = fTP * rho_NS * Vm² / (2 * Dh)"""
    dPdL_F = fTP * pho_NS * Vm**2 / (2 * Dh)
    return dPdL_F


def gradiente_gravitacional(pho_slip, g, theta):
    """Gradiente de pressão gravitacional:
    dP/dL|_G = rho_slip * g * sin(theta)
    theta em graus (positivo = ascendente)"""
    dPdL_G = pho_slip * g * np.sin(np.radians(theta))
    return dPdL_G


def parametro_EK(pho_slip, Vm, Vsg, P):
    """Parâmetro de aceleração:
    EK = rho_slip * Vm * Vsg / P"""
    EK = (pho_slip * Vm * Vsg) / P
    return EK


def gradiente_total(dPdL_F, dPdL_G, EK):
    """Gradiente de pressão total Beggs & Brill:
    dP/dL|_T = (-(dP/dL|_F) - (dP/dL|_G)) / (1 - EK)

    O termo de aceleração é feito via EK, eliminando necessidade
    de calcular dP/dL|_A soziho """
    EK = min(EK, 0.99)
    dPdL_T = (-(dPdL_F) - (dPdL_G)) / (1 - EK)
    return dPdL_T



