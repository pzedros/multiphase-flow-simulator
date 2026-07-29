import numpy as np


# Balanço de massa em escoamentos:
def vazaosc(Vlsc, RGL, BSW):
    Vosc = Vlsc * (1.0 - BSW)
    Vwsc = Vlsc * BSW
    Vgsc = RGL * Vlsc
    return Vosc, Vwsc, Vgsc


def vazaoinsitu(Vosc, Vwsc, Vgsc, Bo, Bg, Bw, Rs, Rsw):
    Vl = Vosc * Bo + Vwsc * Bw
    # Evita gás livre negativo por oscilações numéricas do PVT
    gas_livre_sc = max(Vgsc - (Vosc * Rs) - (Vwsc * Rsw), 0.0)
    Vg = gas_livre_sc * Bg
    return Vl, Vg


def prop_liquid(pho_o, pho_w, Vosc, Vwsc, Bo, Bw, Vl, mu_o, mu_w, tensao_og, tensao_wg):
    Fo = (Vosc * Bo) / max(Vl, 1e-12)
    Fwc = 1.0 - Fo
    pho_l = (pho_o * Fo) + (pho_w * Fwc)
    mu_l = (mu_o * Fo) + (mu_w * Fwc)
    tensao_lg = (tensao_og * Fo) + (tensao_wg * Fwc)
    return Fo, Fwc, pho_l, mu_l, tensao_lg


# Modelo de Beggs & Brill (1973)
def holdup(Vl, Vg, Ap):
    Vsl = Vl / max(Ap, 1e-12)
    Vsg = Vg / max(Ap, 1e-12)
    Vm = Vsl + Vsg
    holdup_L = Vsl / max(Vm, 1e-12)
    holdup_G = Vsg / max(Vm, 1e-12)
    return holdup_L, holdup_G, Vsl, Vsg, Vm


def Froude(Vm, Dh, g):
    Frm = (Vm ** 2) / max(g * Dh, 1e-12)
    return Frm


def param_L(holdup_L):
    hl = max(holdup_L, 1e-6)
    L1 = 316.0 * (hl ** 0.302)
    L2 = 0.0009252 * (hl ** -2.4684)
    L3 = 0.10 * (hl ** -1.4516)
    L4 = 0.50 * (hl ** -6.738)
    return L1, L2, L3, L4


def padrao_escoamento(L1, L2, L3, L4, holdup_L, Frm):

    if (holdup_L < 0.4 and Frm >= L1) or (holdup_L >= 0.4 and Frm > L4):
        flag = 1  # Distribuído
    elif (holdup_L < 0.01 and Frm < L1) or (holdup_L >= 0.001 and Frm < L2):
        flag = 2  # Segregado
    elif holdup_L >= 0.01 and L2 <= Frm <= L3:
        flag = 3  # Transição
    elif (0.01 <= holdup_L < 0.4 and L3 <= Frm <= L1) or (holdup_L >= 0.4 and L3 <= Frm <= L4):
        flag = 4  # Intermitente
    else:
        flag = 404
        print("Nenhum padrão escolhido")
    return flag


def HLO(flag, holdup_L, Frm):
    """Holdup horizontal (H_LO) bruto — APENAS flags 1, 2, 4."""
    hl = max(holdup_L, 1e-6)
    fr = max(Frm, 1e-6)
    if flag == 1:  # Distribuído
        HlO = (1.065 * hl ** 0.5824) / (fr ** 0.0609)
    elif flag == 2:  # Segregado
        HlO = (0.98 * hl ** 0.4846) / (fr ** 0.0868)
    else:  # Intermitente (ou fallback)
        HlO = (0.845 * hl ** 0.5351) / (fr ** 0.0173)
    return HlO


def numero_velocidade_liquido(Vsl, pho_l, g, tensao_lg):
    NLV = Vsl * (pho_l / max(g * tensao_lg, 1e-12)) ** 0.25
    return NLV


def psi_inclinacao(flag, holdup_L, Frm, NLV, theta):
    C = 0.0  # Inicialização de segurança
    hl = max(holdup_L, 1e-6)
    fr = max(Frm, 1e-6)
    nlv = max(NLV, 1e-6)

    if theta >= 0:  # Ascendente
        if flag == 1:
            C = 0.0
        elif flag == 2:
            d, e, f, g_exp = 0.011, -3.768, 3.539, -1.614
            termo = d * hl ** e * nlv ** f * fr ** g_exp
            C = max((1.0 - hl) * np.log(max(termo, 1e-12)), 0.0)
        elif flag == 4:
            d, e, f, g_exp = 2.960, 0.305, -0.4473, 0.0978
            termo = d * hl ** e * nlv ** f * fr ** g_exp
            C = max((1.0 - hl) * np.log(max(termo, 1e-12)), 0.0)
    else:  # Descendente
        d, e, f, g_exp = 4.700, -0.3692, 0.1244, -0.5056
        termo = d * hl ** e * nlv ** f * fr ** g_exp
        C = max((1.0 - hl) * np.log(max(termo, 1e-12)), 0.0)

    sin_1_8_theta = np.sin(np.radians(1.8 * theta))
    psi = 1.0 + C * (sin_1_8_theta - 0.333 * sin_1_8_theta ** 3)
    return C, psi


def holdup_liquido(HlO_bruto, psi, holdup_L):
    HL_corrigido = HlO_bruto * psi
    HL_final = max(HL_corrigido, holdup_L)
    return min(HL_final, 1.0)


def HL_transicao(holdup_L, Frm, NLV, theta, L2, L3):
    A = (L3 - Frm) / max(L3 - L2, 1e-12)
    A = max(min(A, 1.0), 0.0)

    HlO_seg = HLO(2, holdup_L, Frm)
    _, psi_seg = psi_inclinacao(2, holdup_L, Frm, NLV, theta)
    HL_seg = holdup_liquido(HlO_seg, psi_seg, holdup_L)

    HlO_int = HLO(4, holdup_L, Frm)
    _, psi_int = psi_inclinacao(4, holdup_L, Frm, NLV, theta)
    HL_int = holdup_liquido(HlO_int, psi_int, holdup_L)

    return A * HL_seg + (1.0 - A) * HL_int


def densidade_noslip(pho_l, pho_g, holdup_L):
    return pho_l * holdup_L + pho_g * (1.0 - holdup_L)


def densidade_slip(pho_l, pho_g, HL):
    return pho_l * HL + pho_g * (1.0 - HL)


def viscosidade_noslip(mu_l, mu_g, holdup_L):
    return mu_l * holdup_L + mu_g * (1.0 - holdup_L)


def reynolds_noslip(pho_NS, Vm, Dh, mu_NS):
    return (pho_NS * Vm * Dh) / max(mu_NS, 1e-12)


def fator_atrito_noslip(Re_NS, eps, Dh):
    fN = 0.0055 * (1.0 + (2e4 * (eps / Dh) + 1e6 / max(Re_NS, 1e-12)) ** (1.0 / 3.0))
    return fN


def parametro_s(holdup_L, HL):
    y = holdup_L / max(HL ** 2, 1e-12)
    if 1.0 < y <= 1.2:
        s = np.log(max(2.2 * y - 1.2, 1e-12))
    else:
        ln_y = np.log(max(y, 1e-12))
        den = -0.0523 + 3.182 * ln_y - 0.8725 * ln_y ** 2 + 0.01853 * ln_y ** 4
        s = ln_y / den if abs(den) > 1e-12 else 0.0
    return s


def fator_atrito_bifasico(fN, s):
    return fN * np.exp(s)


def gradiente_friccao(fTP, pho_NS, Vm, Dh):
    return (fTP * pho_NS * Vm ** 2) / max(2.0 * Dh, 1e-12)


def gradiente_gravitacional(pho_slip, g, theta):
    return pho_slip * g * np.sin(np.radians(theta))


def parametro_EK(pho_slip, Vm, Vsg, P):
    return (pho_slip * Vm * Vsg) / max(P, 1e-12)


def gradiente_total(dPdL_F, dPdL_G, EK):
    EK = min(EK, 0.99)
    dPdL_T = (-(dPdL_F) - (dPdL_G)) / max(1.0 - EK, 1e-12)
    return dPdL_T