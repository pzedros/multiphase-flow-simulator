import numpy as np
import pandas as pd

#======================================================================================================================#
"""Propriedades PVT da fase Óleo"""


"""Correlações: Usaremos as de Standing """

def do(pho_o):
    """É a razão entre a massa específica do óleo e a massa específica da água,
    ambas medidas na mesma condição de Pressão ( ) e Temperatura ( ).
    onde,
    do é a  densidade relativa do óleo
    pho_o é a massa específica do óleo [Kg/m³]
    pho_w é a massa específica da água [Kg/m³]"""
    pho_w = 1000 #[Kg/m³]
    do = pho_o/pho_w
    return do

def grauAPI(do):
    API = (141.5 / do ) - 131.5
    return API

def  calcular_api_do(API= None, do=None):
    if API is not None and do is None:
        do = 141.5/(API+131.5)
    elif API is None and do is not None:
        API = (141.5 / do) - 131.5

    return API, do

# def pressao_bolha(Rs, dg, Tf, API): #
#     a = 0.00091*Tf - 0.0125 * API
#     Pb = 18.2 * (10**a * (Rs/dg)**0.83 - 1.4) #𝑃𝑏 [𝑝𝑠𝑖𝑎]
#
#     return Pb

def pressao_bolha(RGL, dg, T, API):
    # T entra em °C e temos que converter em F
    Tf = T * 9/5 + 32
    a = 0.00091 * Tf - 0.0125 * API
    Pb = 18.2 * (((RGL / dg) ** 0.83) * 10 ** a - 1.4) #psia
    #Pb = 18.2 * (((Rs / dg) ** 0.83) * 10 ** a - 1.4)
    Pb_bar = Pb / 14.5038 #bar
    return Pb_bar

def rs_gas_oil(dg, Pbar, Pb, API, T):
    P = Pbar * 14.5038
    Tf = T * 9 / 5 + 32
    if P > Pb:
        P = Pb

    #P<=Pb
    Rs = dg * (((P/18.2)+1.4) * 10**(0.0125*API-0.00091*Tf))**(1/0.83)
    return Rs

def bo(Rs, dg, do, T, Pbar, Pb, Co, RGL):
    P = Pbar * 14.5038
    Tf = T * 9 / 5 + 32

    # Fator Volume-Formação do Óleo
    #Bo  [𝑏𝑏𝑙/𝑆𝑇𝐵]
    Bob = 0.9759 + 0.00012 * (RGL * (dg/do)**0.5 + 1.25*Tf)**1.2
    if P > Pb:
        Bo = Bob * np.exp(-Co*(P-Pb))

    elif P <= Pb:
        Bo = 0.9759 + 0.00012 * (Rs * (dg / do) ** 0.5 + 1.25 * Tf) ** 1.2
    return Bo

def pho_o_insitu(do, Rs, dg, Bo, Bob, Pbar, Pb, Co, RGL, Tf):

    #pho_o_insitu Massa específica do óleo [lb/ft³]
    #do Gravidade  específica do óleo
    #Rs Razão solubilidade Gás-óleo SCF/STB
    #dg Gravidade específica do gás

    P = Pbar * 14.5038

    pho_ob = (62.4 * do + 0.0136 * Rs * dg) / Bob

    if P > Pb:
        pho_o = pho_ob * np.exp(Co*(P-Pb))

    else:
        #P <= Pb
        pho_o = (62.4*do + 0.0136*Rs*dg)/Bo
    return pho_o

def compressibilidade_oleo(Pbar, T, dg, do, Rs, Bob, API, Bg, Pb, pho_ob):
    P = Pbar * 14.5038
    Tf = T * 9 / 5 + 32

    if P >= Pb:
        Co = (1e-6)* np.exp((pho_ob + 0.004347 * (P - Pb) - 79.1) / (0.0007141 * (P - Pb) - 12.938))
    else:
        #P<Pb
        dRs_dP = dg * (1/0.83) * (((P/18.2)+1.4) * (10 ** (0.0125 * API - 0.00091 * Tf))) ** (1 / 0.83 - 1) * (1 / 18.2) * (10 ** (0.0125 * API - 0.00091 * Tf))
        dBo_dP = 0.00012 * 1.2 * (Rs * ((dg/do)**0.5) + 1.25 * Tf) ** 0.2 * ((dg / do) ** 0.5) * dRs_dP
        Co = -(1/Bob) * dBo_dP + (Bg/Bob)*dRs_dP
    return Co

# pho_o = (pho_osc + (Rs * pho_gsc)) / Bo ????????
#===================================================

"""Correlação de Beggs e Robinson (1975):"""
def viscosidade_oleomorto(T, API):

    Tf = T * 9 / 5 + 32
    A = 10**(3.0324 - (0.02023 * API))

    mu_od = 10**(A * Tf**(-1.163)) - 1
    return mu_od

def viscosidade_oleosaturado(Rs, mu_od, Pbar, Pb):
    P = Pbar * 14.5038
    # P <= Pb:
    # Viscosidade de óleo saturado
    a = 10.715 * (Rs + 100) ** (-0.515)

    b = 5.44 * (Rs + 150) ** (-0.338)

    mu_obs = a * mu_od ** b

    if P > Pb:
    ## Viscosidade de óleo subsaturado
        m = 2.6 * P**(1.187) * np.exp(-11.513 - (8.98e-5 * P))

        mu_obss = mu_obs * (P/Pb)**m
        mu_obs = mu_obss

    return mu_obs

#======================================================================================================================#
"""Propriedades PVT da fase Gás"""

def dg(pho_g):
    """É definida como a relação entre a massa específica de um gás e a massa específica do ar seco,
    na mesma condição de Pressão ( ) e de Temperatura( ).
    dg -> densidade relativa ou gravidade específica (adimensional)
    pho_g -> massa específica de um gás [Kg/m/3]
    pho_ar -> massa específica do ar [Kg/m/3]
    Psc -> pressão na condição padrão  [Pa]
    Tsc -> Temperatura na condição padrão [K]
    """
    pho_ar = 1.225 #[Kg/m³]
    dg = pho_g/pho_ar
    return dg

def propseudo(dg,Pbar,T):
    #Prop. Pseudocríticas e Pseudoreduzidas
    """
    Quando é conhecido apenas a gravidade específica da mistura gasosautiliza-se as seguintes correlações:
    onde,
    P é a pressão absoluta
    Ppc é a pressão pseudocrítica
    T é a temperatura (em ºR Rankine)
    Tpc é a temperatura pseudocrítica
    """
    P = Pbar * 14.5038
    TR = T * 9/5 + 491.67

    Ppc = 0
    Tpc = 0
    if dg < 0.75: #Gás seco
        Ppc = 677 + 15 * dg - 37.5 * dg**2
        Tpc = 168 + 325 * dg - 12.5 * dg**2

    elif dg >= 0.75: #Gás umido
        Ppc = 706 - 51.7 * dg - 11.1 * dg**2
        Tpc = 187 + 330 * dg - 71.5 * dg**2

    Ppr = P/Ppc

    Tpr = TR/Tpc

    return Ppc, Tpc, Ppr, Tpr

def pho_g(P, Mg, Z, T):
    #Cálculo de massa específica do gás
    """É definida como a massa do gás ocupando um certo volume em pressão etemperatura especificada."""
    R = 10.73 #cte universal dos gases
    #8.314 Pa.m³ / mol . K
    pho_g = (P * Mg)/(Z * R * T)
    return pho_g

def factor_Z(Ppc, Tpc, Ppr, Tpr):
    #Correlação do Papay (1985)
    z = 1 - ((3.53*Ppr)/(10**(0.9813*Tpr))) + ((0.274*Ppr**2)/(10**(0.8157*Tpr)))
    return z

def compress_gas(Z, Ppc, Ppr, Tpr):
    """É definida como à variação relativa de volume do gás por variação unitáriade
    pressão em temperatura constante (Bahadori, 2017).
    Cg é a compressibilidade de gás [Pa^-¹]"""

    #Cpr = (1/Ppr) - (1/Z) * (dZ_dPpr)_Tpr

    dZ_dPpr = - (3.53/(10**(0.9813*Tpr))) + 2 * (0.274/10**(0.8157*Tpr)) * Ppr

    Cpr = (1/Ppr) - (1/Z) * dZ_dPpr

    Cg = Cpr/Ppc
    return Cg

def bg(T,Pbar,Z):
    #Fator Volume-Formação do Gás
    """Em unidades de campo, Psc = 14.7 psia (pressão na condição de superfície) e Tsc = 60ºF (temperatura na condição de superfície),
    ou seja, nas condiçõespadrão."""
    P = Pbar * 14.5038
    Tf = T * 9 / 5 + 32

    Psc =14.7 #[psia]
    Tsc = 60 #[°F]
    Bg = Psc/Tsc * Z * Tf/P
    return Bg

def Eg(Bg):
    """Eg é o fator de expansão do gás, m³ std/m³"""
    Eg = 1/Bg
    return Eg

def viscosidade_fgas(pho_g, Mg, TR):
    """Correlação de Lee et al. (1966)
    onde,
    mu_g é a viscosidade de gás, cP
    pho_g é a densidade de gás, lb/ ft³
    Temperatura (ºR)
    Mg é o peso molecular do gás (lbm/lb mol)"""

    #mu_g = 10**(-4) * Kv * np.exp(Xv*(pho_g/62.4)**Yv)

    Xv = 3.448 + (986.4/TR) + 0.01009 * Mg
    Yv = 2.4 - 0.2 * Xv

    Kv = ((9.379 + 0.0160 * Mg) * TR**1.5)/(209.2 + 19.26 * Mg + TR)

    mu_g = 10 ** (-4) * Kv * np.exp(Xv * (pho_g / 62.4) ** Yv) #[cP]
    return mu_g


#======================================================================================================================#
"""Propriedades PVT da fase Água"""

def pho_w(S):
    """O efeito da salinidade sobre a densidade da água produzida em
    campos deóleo é estimado através da seguinte correlação:
    pho_w é a massa específica da água, lb/SCF
    S representa a salinidade, % de peso de sólidos
    Comportamento da massa específica da água sobre o efeito
    da salinidade: Quanto maior a salinidade, maior será a massa específica"""

    #S ~= 0,125 = 12.5%
    # Plotar grafico 
    pho_w = 62.368 + 0.438603 * S + 1.60074e-3 * S**2

    return pho_w

def rs_w(Pbar, T):

    P = Pbar * 14.5038

    Tf = T * 9 / 5 + 32

    #Razão de solubilidade gás-água
    #P é a pressão, psia
    #T é a temperatura, ºF
    #Rs_w = a + (b * P) + (c * P**2)
    #Os parâmetros , e são correlacionados com a temperatura, como mostra:

    #Coeficientes da correlação de Rs_w
    A = np.array([8.15839, -6.12265e-2, 1.91663e-4, -2.1654e-7])
    B = np.array([1.01021e-2, -7.44241e-5, 3.05553e-7, -2.94883e-10])
    C = np.array([-9.02505, 0.130237, -8.53425e-4, 2.34122e-6, -2.37049e-9])

    a = A[0] + A[1] * Tf + A[2] * Tf ** 2 + A[3] * Tf ** 3

    b = B[0] + B[1] * Tf + B[2] * Tf ** 2 + B[3] * Tf ** 3

    c = (C[0] + (C[1] * Tf) + (C[2] * Tf ** 2) + (C[3] * Tf ** 3) + (C[4] * Tf ** 4)) * (10 ** -7)

    Rs_w = a + (b * P) + (c * P ** 2)
    return Rs_w

def compress_agua(T, Pbar, S):
    """Compressibilidade Isotérmica da água:
    Cw [psia-¹]
    Pressão P , psia
    Temperatura ºF
    Salinidade (S, mg/L

    obs: A correlação é válida somente para pressões entre 1000 e 20000 psia,temperaturas entre 200 e 270 ºF ,
    e salinidade de 0 a 200 mg/L."""
    P = Pbar * 14.5038
    Tf = T * 9 / 5 + 32

    A_1 = 7.033
    A_2 = 0.5415
    A_3 = -537.0
    A_4 = 403.300

    Cw = 1 /((A_1*P) + (A_2*S) + (A_3 * Tf) + A_4)
    return Cw

def bw(T, Pbar):
    """Fator Volume_Formação da água:
    Bw é o FVF da água, bbl/STB
    P é a pressão, psia
    T é a temperatura, ºF"""

    P = Pbar * 14.5038
    Tf = T * 9 / 5 + 32

    #Mudança em volume devido à redução em temperatura
    DVwt = (-1.0001e-2) + (1.33391e-4 * Tf) + (5.50654e-7 * Tf ** 2)
    #Mudança em volume durante a redução de pressão
    DVwp = (-1.95301e-9 * P * Tf) - (1.72834e-13 * P ** 2 * Tf) - (3.58922e-7 * P) - (2.25341e-10 * P ** 2)

    Bw = (1 + DVwt)*(1+DVwp)
    return Bw

def viscosidade_agua(T, Pbar, S):
    P = Pbar * 14.5038
    Tf = T * 9 / 5 + 32

    #Coeficientes da correlação de viscosidade da água


    A = np.array([109.527, -8.40564, 0.313314, 8.72213e-3])
    B = np.array([-1.12166, 2.63951e-2, -6.79461e-4, -5.47119e-5, -1.55586e-6])

    a = A[0] + (A[1] * S) + (A[2] * S**2) + (A[3] * S**3)
    b = B[0] + (B[1] * S) + (B[2] * S**2) + (B[3] * S**3) + (B[4] * S**4)

    #A viscosidade da água em pressão atmosférica é dada por:
    mu_w1 = a * Tf ** b

    #O efeito da pressão sobre a viscosidade da água é calculado pela seguinte equação:
    mu_w = (0.9994 + (4.0295 * 10**(-5) * P ) + (3.1062*10**(-9) * P**2)) * mu_w1
    #P em psia

    return mu_w1, mu_w



#======================================================================================================================#












