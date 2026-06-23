import numpy as np

#  MODELO DE DRIFT FLUX # 

def froude(vm, g, Dh):
    """
    Numero de Froude para escolha do regime em Bendiksen:
    Fr = |vm| / sqrt(g * Dh)
    """
    return abs(vm) / np.sqrt(g * Dh)


def bendiksen(vm, g, Dh, theta):
    """
    Correlacao de Bendiksen (1984):
    Calcula o parametro de distribuicao C0 e a velocidade drift vd
    em funcao do numero de Froude e da inclinacao.

    Entradas:
        vm    : velocidade da mistura [m/s]
        g     : aceleracao da gravidade [m/s²]
        Dh    : diametro hidraulico [m]
        theta : angulo de inclinacao [graus]  (+ ascendente, - descendente)

    Saidas:
        C0    : parametro de distribuicao  [-]
        vd    : velocidade drift           [m/s]
    """
    t = np.radians(theta)
    Fr = froude(vm, g, Dh)

    if Fr < 3.5:
        C0 = 1.05 + 0.15 * np.sin(t)
        vd = np.sqrt(g * Dh) * (0.35 * np.sin(t) + 0.54 * np.cos(t))
    else:
        C0 = 1.2
        vd = 0.35 * np.sqrt(g * Dh) * np.sin(t)

    return C0, vd


def holdup_gas(vsg, C0, vm, vd):
    """
    Fracao de vazio (holdup de gas) — Zuber & Findlay (1965):
    H_G = v_SG / (C0 * vm + vd)

    H_G limitado entre 0 e 1.
    """
    den = C0 * vm + vd
    if abs(den) < 1e-12:
        return 0.0
    HG = vsg / den
    return float(np.clip(HG, 0.0, 1.0))


def prop_mistura(rho_l, rho_g, mu_l, mu_g, HG):
    """
    Propriedades da mistura a partir de H_G:
        H_L   = 1 - H_G
        rho_m = rho_L * H_L + rho_G * H_G   [kg/m³]
        mu_m  = mu_L  * H_L + mu_G  * H_G   [Pa.s]

    Retorna: rho_m, mu_m, H_L
    """
    HL    = 1.0 - HG
    rho_m = rho_l * HL + rho_g * HG
    mu_m  = mu_l  * HL + mu_g  * HG
    return rho_m, mu_m, HL


def reynolds_mistura(rho_m, vm, Dh, mu_m):
    """Re = rho_m * |vm| * Dh / mu_m"""
    return rho_m * abs(vm) * Dh / mu_m


def fator_atrito_darcy(Re, eps, Dh):
    """
    Fator de atrito de Darcy — correlacao de Hall (1957):
    fD = 0.0055 * [1 + (2e4 * eps/Dh + 1e6/Re)^(1/3)]

    eps : rugosidade absoluta [m]
    Dh  : diametro hidraulico [m]
    """
    return 0.0055 * (1.0 + (2e4 * (eps / Dh) + 1e6 / Re) ** (1.0 / 3.0))


def gradiente_pressao(fD, rho_m, vm, Dh, g, theta, dvm_dL=0.0):
    """
    Gradiente de pressao total do modelo de Drift Flux:

        dP/dL = - fD * rho_m * vm² / (2*Dh)   <- friccao
                - rho_m * g * sen(theta)         <- gravitacional
                - rho_m * vm * dvm/dL            <- aceleracao

    Entradas:
        fD      : fator de atrito de Darcy [-]
        rho_m   : densidade da mistura [kg/m³]
        vm      : velocidade da mistura [m/s]
        Dh      : diametro hidraulico [m]
        g       : aceleracao da gravidade [m/s²]
        theta   : angulo de inclinacao [graus]
        dvm_dL  : derivada da velocidade da mistura ao longo de L [1/s]
                  (pode ser ignorado em escoamento incompressivel, default=0)

    Retorna:
        dPdL        : gradiente total   [Pa/m]
        dPdL_f      : parcela friccao   [Pa/m]
        dPdL_g      : parcela grav.     [Pa/m]
        dPdL_a      : parcela acel.     [Pa/m]
    """
    t      = np.radians(theta)
    dPdL_f = -fD * rho_m * vm**2 / (2.0 * Dh)
    dPdL_g = -rho_m * g * np.sin(t)
    dPdL_a = -rho_m * vm * dvm_dL
    dPdL   = dPdL_f + dPdL_g + dPdL_a
    return dPdL, dPdL_f, dPdL_g, dPdL_a


# FUNCAO PRINCIPAL — calcula tudo de uma vez dado o estado local


def calcular(vsl, vsg, rho_l, rho_g, mu_l, mu_g,
             Dh, g, theta, eps, dvm_dL=0.0):
    """
    Calcula o gradiente de pressao pelo modelo de Drift Flux (Bendiksen, 1984).

    Entradas:
        vsl, vsg   : velocidades superficiais de liquido e gas [m/s]
        rho_l/g    : densidades de liquido e gas [kg/m³]
        mu_l/g     : viscosidades de liquido e gas [Pa.s]
        Dh         : diametro hidraulico [m]
        g          : aceleracao gravitacional [m/s²]
        theta      : angulo de inclinacao [graus]
        eps        : rugosidade da tubulacao [m]
        dvm_dL     : derivada da velocidade da mistura [1/s]  (default 0)

    """
    vm = vsl + vsg

    # Froude e coeficientes de Bendiksen
    Fr = froude(vm, g, Dh)
    C0, vd= bendiksen(vm, g, Dh, theta)

    # Holdup de gas e de liquido
    HG = holdup_gas(vsg, C0, vm, vd)
    HL = 1.0 - HG

    # Propriedades da mistura
    rho_m, mu_m, _= prop_mistura(rho_l, rho_g, mu_l, mu_g, HG)

    # Reynolds e fator de atrito
    Re = reynolds_mistura(rho_m, vm, Dh, mu_m)
    fD = fator_atrito_darcy(Re, eps, Dh)

    # Gradiente de pressao
    dPdL, dPdL_f, dPdL_g, dPdL_a = gradiente_pressao(
        fD, rho_m, vm, Dh, g, theta, dvm_dL
    )

    return {
        "vm":     vm,
        "Fr":     Fr,
        "C0":     C0,
        "vd":     vd,
        "HG":     HG,
        "HL":     HL,
        "rho_m":  rho_m,
        "mu_m":   mu_m,
        "Re":     Re,
        "fD":     fD,
        "dPdL":   dPdL,
        "dPdL_f": dPdL_f,
        "dPdL_g": dPdL_g,
        "dPdL_a": dPdL_a,
    }
