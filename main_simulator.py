import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions_flow as fc
import beggs_brill as bb

# ============================================================
# DADOS DO PROBLEMA
# ============================================================
API = 20
dg  = 0.75
RGL = 130
BSW = 0.20
S   = 12.5
g_accel = 9.81

Q_oleo_sm3d = 3500.0
Q_liq_sm3d  = Q_oleo_sm3d / (1.0 - BSW)   # 4375 sm³/d

P_res_bar = 250.0
T_res_C   = 90.0

eps = 1.5e-5

tensao_og = 0.00841
tensao_wg = 0.03

TEC_poco    = 2.0
TEC_marinho = 1.0

do_val   = 141.5 / (API + 131.5)
Mg_lbmol = dg * 28.96

Q_m3s   = Q_liq_sm3d / 86400.0
qm_kg_s = Q_m3s * 880.0

DIAMETROS_POL = [5, 6, 7]   # polegadas
CORES         = ["#1f77b4", "#d62728", "#2ca02c"]

CONV   = 5.6146
Rs_sup = RGL * CONV         # GOR total em scf/STB (730)

# ============================================================
# TOPOLOGIA (igual para todos os diâmetros)
# ============================================================
dL_step = 10.0

def montar_secoes():
    secoes = []
    L_poco_total = 1000.0 / math.sin(math.radians(75.0))
    n_poco = int(L_poco_total / dL_step)
    for i in range(n_poco):
        frac  = i / n_poco
        T_amb = 90.0 + (5.0 - 90.0) * frac
        secoes.append({"theta": 75.0, "T_amb_C": T_amb, "TEC": TEC_poco, "dL": dL_step})

    theta_flow = math.degrees(math.asin(500.0 / 1500.0))
    n_flow = int(1500.0 / dL_step)
    for i in range(n_flow):
        secoes.append({"theta": theta_flow, "T_amb_C": 5.0, "TEC": TEC_marinho, "dL": dL_step})

    n_riser = int(1500.0 / dL_step)
    for i in range(n_riser):
        frac  = i / n_riser
        T_amb = 5.0 + (12.0 - 5.0) * frac
        secoes.append({"theta": 90.0, "T_amb_C": T_amb, "TEC": TEC_marinho, "dL": dL_step})

    return secoes

sections_base = montar_secoes()

# Posições para linhas verticais nos gráficos
L_poco_total = 1000.0 / math.sin(math.radians(75.0))
L_anm      = int(L_poco_total / dL_step) * dL_step
L_manifold = L_anm + int(1500.0 / dL_step) * dL_step

# ============================================================
# FUNÇÃO DE SIMULAÇÃO
# ============================================================
def simular(D_m, sections):
    Area = math.pi * (D_m / 2.0) ** 2

    P_atual_bar = P_res_bar
    T_atual_C   = T_res_C
    L_acumulado = 0.0

    Pb_ini_bar = fc.pressao_bolha(Rs_sup, dg, T_res_C, API) * 0.0689476

    dados = {
        "L_m": [0.0], "P_bar": [P_res_bar], "T_C": [T_res_C],
        "Pb_bar": [Pb_ini_bar],
        "Holdup": [0.0], "Regime": ["N/A"],
        "Vsl": [0.0], "Vsg": [0.0], "Vm": [0.0],
        "Bo": [0.0], "Bg": [0.0],
        "dp_total": [0.0], "dp_fric": [0.0], "dp_grav": [0.0], "dp_acc": [0.0],
        "rho_o": [0.0], "rho_g": [0.0], "rho_w": [0.0],
        "rho_liq": [0.0], "rho_mix": [0.0],
        "mu_o": [0.0], "mu_g": [0.0], "mu_mix": [0.0],
    }

    for step in sections:
        dL      = step["dL"]
        theta   = step["theta"]
        T_amb_C = step["T_amb_C"]
        TEC     = step["TEC"]

        if P_atual_bar < 1.0:
            break

        T_C    = T_atual_C
        P_bar  = P_atual_bar
        Tf     = T_C * 9.0 / 5.0 + 32.0
        TR     = T_C * 9.0 / 5.0 + 491.67
        Tk     = T_C + 273.15
        P_psia = P_bar * 14.5038
        P_Pa   = P_bar * 1.0e5

        Ppc, Tpc, Ppr, Tpr = fc.propseudo(dg, P_bar, T_C)
        Z = fc.factor_Z(Ppc, Tpc, Ppr, Tpr)

        Bg_ft3scf = (14.7 / 519.67) * Z * (TR / P_psia)
        Bg_SI     = Bg_ft3scf

        Pb_psia = fc.pressao_bolha(Rs_sup, dg, T_C, API)
        Rs      = fc.rs_gas_oil(dg, P_bar, Pb_psia, API, T_C)

        Bob    = 0.9759 + 0.00012 * (Rs_sup * (dg / do_val) ** 0.5 + 1.25 * Tf) ** 1.2
        rho_ob = (62.4 * do_val + 0.0136 * Rs * dg) / Bob

        Co       = fc.compressibilidade_oleo(P_bar, T_C, dg, do_val, Rs, Bob, API, Bg_ft3scf, Pb_psia, rho_ob)
        Bo       = fc.bo(Rs, dg, do_val, T_C, P_bar, Pb_psia, Co, Rs_sup)
        Bo_SI    = Bo

        rho_o_si = fc.pho_o_insitu(do_val, Rs, dg, Bo, Bob, P_bar, Pb_psia, Co, Rs_sup, T_C) * 16.018463
        mu_od    = fc.viscosidade_oleomorto(T_C, API)
        mu_o_cP  = fc.viscosidade_oleosaturado(Rs, mu_od, P_bar, Pb_psia)
        mu_o_Pas = mu_o_cP * 1e-3

        rho_w_si   = fc.pho_w(S) * 16.018463
        bw_val     = fc.bw(T_C, P_bar)
        rs_w_val   = fc.rs_w(P_bar, T_C)
        _, mu_w_cP = fc.viscosidade_agua(T_C, P_bar, S)
        mu_w_Pas   = mu_w_cP * 1e-3

        Mg_kg    = dg * 0.02896
        rho_g_si = (P_Pa * Mg_kg) / (Z * 8.314 * Tk)
        mu_g_cP  = fc.viscosidade_fgas(rho_g_si / 16.018463, Mg_lbmol, TR)
        mu_g_Pas = mu_g_cP * 1e-3

        q_oil_s      = Q_m3s * (1.0 - BSW)
        q_wat_s      = Q_m3s * BSW
        q_gas_livre  = max((RGL - Rs / CONV) * q_oil_s - (rs_w_val / CONV) * q_wat_s, 0.0)
        q_gas_insitu = q_gas_livre * Bg_SI
        q_liq_insitu = q_oil_s * Bo_SI + q_wat_s * bw_val

        holdup_L, _, Vsl, Vsg, Vm = bb.holdup(q_liq_insitu, q_gas_insitu, Area)

        Fo      = (q_oil_s * Bo_SI) / max(q_liq_insitu, 1e-12)
        Fwc     = 1.0 - Fo
        rho_liq = rho_o_si * Fo + rho_w_si * Fwc
        mu_liq  = mu_o_Pas * Fo + mu_w_Pas * Fwc
        tensao_lg = tensao_og * Fo + tensao_wg * Fwc

        Frm = bb.Froude(Vm, D_m, g_accel)
        L1, L2, L3, L4 = bb.param_L(holdup_L)

        try:
            flag = bb.padrao_escoamento(L1, L2, L3, L4, holdup_L, Frm)
        except Exception:
            flag = 4

        padroes    = {1: "Distribuído", 2: "Segregado", 3: "Transição", 4: "Intermitente"}
        regime_str = padroes.get(flag, "Intermitente")

        NLV = bb.numero_velocidade_liquido(Vsl, rho_liq, g_accel, tensao_lg)

        if flag == 3:
            HL = bb.HL_transicao(holdup_L, Frm, NLV, theta, L2, L3)
        else:
            HlO     = bb.HLO(flag, holdup_L, Frm)
            _, psi  = bb.psi_inclinacao(flag, holdup_L, Frm, NLV, theta)
            HL      = bb.holdup_liquido(HlO, psi, holdup_L)

        rho_NS   = bb.densidade_noslip(rho_liq, rho_g_si, holdup_L)
        rho_slip = bb.densidade_slip(rho_liq, rho_g_si, HL)
        mu_NS    = bb.viscosidade_noslip(mu_liq, mu_g_Pas, holdup_L)

        Re_NS   = bb.reynolds_noslip(rho_NS, Vm, D_m, mu_NS)
        fN      = bb.fator_atrito_noslip(Re_NS, eps, D_m)
        s_param = bb.parametro_s(holdup_L, HL)
        fTP     = bb.fator_atrito_bifasico(fN, s_param)

        dPdL_F = bb.gradiente_friccao(fTP, rho_NS, Vm, D_m)
        dPdL_G = bb.gradiente_gravitacional(rho_slip, g_accel, theta)
        EK     = bb.parametro_EK(rho_slip, Vm, Vsg, P_Pa)
        dPdL_T = bb.gradiente_total(dPdL_F, dPdL_G, EK)

        dp_total_Pa_m = -dPdL_T
        P_new_bar = P_atual_bar - dp_total_Pa_m * 1e-5 * dL

        if P_new_bar < 1.0:
            break

        T_amb_K   = T_amb_C + 273.15
        T_old_K   = T_atual_C + 273.15
        theta_rad = math.radians(theta)
        Cp_J      = (((2e-3) * T_amb_C - 1.429) * do_val + (2.67e-3) * T_amb_C + 3.049) * 1000.0
        term_A    = (qm_kg_s * g_accel * math.sin(theta_rad)) / TEC
        fator_exp = math.exp(-(TEC * dL) / (qm_kg_s * Cp_J))
        T_new_C   = ((T_amb_K - term_A) - fator_exp * (T_amb_K - term_A - T_old_K)) - 273.15

        L_acumulado += dL
        P_atual_bar  = P_new_bar
        T_atual_C    = T_new_C

        dp_total_bm = dp_total_Pa_m * 1e-5
        dp_fric_bm  = dPdL_F * 1e-5
        dp_grav_bm  = dPdL_G * 1e-5

        dados["L_m"].append(L_acumulado)
        dados["P_bar"].append(P_atual_bar)
        dados["T_C"].append(T_atual_C)
        dados["Pb_bar"].append(fc.pressao_bolha(Rs_sup, dg, T_new_C, API) * 0.0689476)
        dados["Holdup"].append(HL)
        dados["Regime"].append(regime_str)
        dados["Vsl"].append(Vsl)
        dados["Vsg"].append(Vsg)
        dados["Vm"].append(Vm)
        dados["Bo"].append(Bo)
        dados["Bg"].append(Bg_SI)
        dados["dp_total"].append(dp_total_bm)
        dados["dp_fric"].append(dp_fric_bm)
        dados["dp_grav"].append(dp_grav_bm)
        dados["dp_acc"].append(dp_total_bm - dp_fric_bm - dp_grav_bm)
        dados["rho_o"].append(rho_o_si)
        dados["rho_g"].append(rho_g_si)
        dados["rho_w"].append(rho_w_si)
        dados["rho_liq"].append(rho_liq)
        dados["rho_mix"].append(rho_slip)
        dados["mu_o"].append(mu_o_Pas)
        dados["mu_g"].append(mu_g_Pas)
        dados["mu_mix"].append(mu_liq * holdup_L + mu_g_Pas * (1.0 - holdup_L))

    return pd.DataFrame(dados), L_acumulado

# ============================================================
# RODAR PARA CADA DIÂMETRO
# ============================================================
resultados = {}
for D_pol in DIAMETROS_POL:
    D_m = D_pol * 0.0254
    print(f"Simulando D = {D_pol} pol ({D_m:.4f} m)...")
    df, L_total = simular(D_m, sections_base)
    P_chegada = df["P_bar"].iloc[-1]
    print(f"  Pressão de chegada: {P_chegada:.2f} bar | {'SUCESSO' if P_chegada >= 1.013 else 'FALHA'}")
    resultados[D_pol] = df
    df.to_excel(f"Relatorio_BB_{D_pol}pol.xlsx", index=False)

print("\nArquivos Excel gerados.")

# ============================================================
# GRÁFICOS COMPARATIVOS
# ============================================================
def estilo_padrao():
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.axvline(L_anm,      color="gray", linestyle="--", lw=1.2)
    plt.axvline(L_manifold, color="gray", linestyle=":",  lw=1.2)
    plt.tight_layout()

def plotar(var, titulo, ylabel, logy=False):
    fig, ax = plt.subplots(figsize=(11, 6))
    for D_pol, cor in zip(DIAMETROS_POL, CORES):
        df = resultados[D_pol]
        ax.plot(df["L_m"], df[var], color=cor, lw=2, label=f"{D_pol} pol")
    ax.set_title(titulo + " — Beggs & Brill", fontsize=14)
    ax.set_xlabel("Comprimento (m)"); ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend(title="Diâmetro", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axvline(L_anm,      color="gray", linestyle="--", lw=1.2)
    ax.axvline(L_manifold, color="gray", linestyle=":",  lw=1.2)
    fig.tight_layout()
    plt.show()

# Pressão (com Pb do primeiro diâmetro como referência)
plt.figure(figsize=(11, 6))
for D_pol, cor in zip(DIAMETROS_POL, CORES):
    df = resultados[D_pol]
    plt.plot(df["L_m"], df["P_bar"],  color=cor, lw=2, label=f"{D_pol} pol")
plt.plot(resultados[DIAMETROS_POL[0]]["L_m"],
         resultados[DIAMETROS_POL[0]]["Pb_bar"],
         color="black", lw=1.5, linestyle="--", label="Pressão de Bolha")
plt.axhline(1.01325, color="black", linestyle=":", lw=1.5, label="Pressão Atm.")
plt.title("Pressão ao longo do sistema — Beggs & Brill", fontsize=14)
plt.xlabel("Comprimento (m)"); plt.ylabel("Pressão (bar)")
plt.legend(title="Diâmetro", fontsize=11)
estilo_padrao()
plt.show()

plotar("T_C",     "Temperatura",                  "T (°C)")
plotar("Holdup",  "Holdup Líquido",               "H_L (-)")
plotar("Vsg",     "Velocidade Superficial do Gás","Vsg (m/s)")
plotar("Vsl",     "Velocidade Superficial do Líquido", "Vsl (m/s)")
plotar("Vm",      "Velocidade da Mistura",         "Vm (m/s)")
plotar("Bo",      "Fator de Volume do Óleo",       "Bo (m³/m³)")
plotar("Bg",      "Fator de Volume do Gás",        "Bg (m³/sm³)")
plotar("dp_fric", "Gradiente — Fricção",           "dP/dL (bar/m)")
plotar("dp_grav", "Gradiente — Gravidade",         "dP/dL (bar/m)")
plotar("dp_acc",  "Gradiente — Aceleração",        "dP/dL (bar/m)")
plotar("dp_total","Gradiente Total de Pressão",    "dP/dL (bar/m)")
plotar("rho_o",   "Densidade do Óleo",             "kg/m³")
plotar("rho_g",   "Densidade do Gás",              "kg/m³")
plotar("rho_w",   "Densidade da Água",             "kg/m³")
plotar("rho_mix", "Densidade da Mistura (slip)",   "kg/m³")
plotar("mu_o",    "Viscosidade do Óleo",           "Pa.s")
plotar("mu_g",    "Viscosidade do Gás",            "Pa.s")
plotar("mu_mix",  "Viscosidade da Mistura",        "Pa.s")
