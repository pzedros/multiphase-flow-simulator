# ============================================================
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions_pvt as fc
import beggs_brill as bb
# ============================================================




# DADOS DO PROBLEMA
API = 20 #[°API]
dg  = 0.75 # gravidade específica do gás
RGL = 130 # razão gás-óleo na superfície [sm³/sm³]
BSW = 0.20 # 20%
S   = 12.5 # Salinidade da água em
g_accel = 9.81 # Aceleração da gravidade


Q_liq_sm3d = 3500.0 #[sm³/d]
Q_oleo_sm3d  = Q_liq_sm3d / (1.0 - BSW)

P_res_bar = 250.0
T_res_C   = 90.0

eps = 4.5e-5  # Rugosidade aço comercial em metros

tensao_og = 0.00841
tensao_wg = 0.03

TEC_poco    = 2.0
TEC_marinho = 1.0


# Temperaturas Ambiente
T_fundo_mar_C = 5.0     # °C
T_superficie_C = 12.0   # °C

do_val   = 141.5 / (API + 131.5)
Mg_lbmol = dg * 28.96

Q_m3s   = Q_liq_sm3d / 86400.0 #[sm³/s]
qm_kg_s = Q_m3s * 880.0

DIAMETROS_POL = [6]   # polegadas
CORES         = ["#1f77b4"]

# TOPOLOGIA (igual para todos os diâmetros)
dL_step = 10.0
sections = []

# Poço (0->1035m)
for i in range(int((1035 - 0) / dL_step)):
    frac = i / int((1035 - 0) / dL_step)
    T_amb = 90.0 + (5.0 - 90.0) * frac
    sections.append({"theta": 75.0, "T_amb_C": T_amb, "TEC": TEC_poco, "dL": dL_step})

# Flowline (1581m))
L_flow = 1581.0
for i in range(int(L_flow / dL_step)):
    sections.append({"theta": 18.43, "T_amb_C": T_fundo_mar_C, "TEC": TEC_marinho, "dL": dL_step})

# Riser (1500m))
for i in range(int(1500.0 / dL_step)):
    frac = i / int(1500.0 / dL_step)
    T_amb = 5.0 + (12.0 - 5.0) * frac
    sections.append({"theta": 90.0, "T_amb_C": T_amb, "TEC": TEC_marinho, "dL": dL_step})

L_total_sistema = sum([s['dL'] for s in sections])


# Posições para linhas verticais nos gráficos
#(Para identificar os trechos no plot)
L_anm      = 1035.0
L_manifold = 2616.0


# FUNÇÃO DE SIMULAÇÃO
def simular(D_m, sections):
    Area = math.pi * (D_m / 2.0) ** 2

    P_atual_bar = P_res_bar
    T_atual_C   = T_res_C
    L_acumulado = 0.0

    Pb_ini_bar = fc.pressao_bolha(RGL, dg, T_res_C, API) * 0.0689476

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
        Bg_SI     = Bg_ft3scf * 0.0283168

        Pb_psia = fc.pressao_bolha(RGL, dg, T_C, API)
        Pb_bar = Pb_psia * 0.0689476

        Rs      = fc.rs_gas_oil(dg, P_bar, Pb_psia, API, T_C)

        Bob    = 0.9759 + 0.00012 * (RGL * (dg / do_val) ** 0.5 + 1.25 * Tf) ** 1.2
        rho_ob = (62.4 * do_val + 0.0136 * Rs * dg) / Bob

        Co       = fc.compressibilidade_oleo(P_bar, T_C, dg, do_val, Rs, Bob, API, Bg_ft3scf, Pb_psia, rho_ob)
        Bo       = fc.bo(Rs, dg, do_val, T_C, P_bar, Pb_psia, Co, RGL)
        Bo_SI    = Bo * 0.158987

        rho_o_si = fc.pho_o_insitu(do_val, Rs, dg, Bo, Bob, P_bar, Pb_psia, Co, RGL, T_C) * 16.018463
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
        q_gas_livre  = max((RGL - Rs) * q_oil_s - rs_w_val * q_wat_s, 0.0)
        q_gas_insitu = q_gas_livre * Bg_SI
        q_liq_insitu = q_oil_s * Bo_SI + q_wat_s * bw_val

        holdup_L, _, Vsl, Vsg, Vm = bb.holdup(q_liq_insitu, q_gas_insitu, Area)

        Fo      = (q_oil_s * Bo_SI) / max(q_liq_insitu, 1e-12)
        Fwc     = 1.0 - Fo
        rho_liq = rho_o_si * Fo + rho_w_si * Fwc
        mu_liq  = mu_o_Pas * Fo + mu_w_Pas * Fwc
        tensao_lg = tensao_og * Fo + tensao_wg * Fwc

        if P_bar > Pb_bar:
            HL = holdup_L
            regime_str = "Monofásico"
            flag = 0
            print(flag)

        elif P_bar <= Pb_bar:
             # BB
            Frm = bb.Froude(Vm, D_m, g_accel)
            L1, L2, L3, L4 = bb.param_L(holdup_L)

            # try:
            #     flag = bb.padrao_escoamento(L1, L2, L3, L4, holdup_L, Frm)
            # except Exception:
            #     flag = 4

            flag = bb.padrao_escoamento(L1, L2, L3, L4, holdup_L, Frm)
            print(flag)

            padroes = {1: "Distribuído", 2: "Segregado", 3: "Transição", 4: "Intermitente"}
            regime_str = padroes.get(flag)


            NLV = bb.numero_velocidade_liquido(Vsl, rho_liq, g_accel, tensao_lg)

            if flag == 3:
                HL = bb.HL_transicao(holdup_L, Frm, NLV, theta, L2, L3)
            else:
                HlO = bb.HLO(flag, holdup_L, Frm)
                _, psi = bb.psi_inclinacao(flag, holdup_L, Frm, NLV, theta)
                HL = bb.holdup_liquido(HlO, psi, holdup_L)


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

        dp_total_bm = dp_total_Pa_m*1e-5
        dp_fric_bm  = dPdL_F*1e-5
        dp_grav_bm  = dPdL_G*1e-5

        dados["L_m"].append(L_acumulado)
        dados["P_bar"].append(P_atual_bar)
        dados["T_C"].append(T_atual_C)
        dados["Pb_bar"].append(fc.pressao_bolha(RGL, dg, T_new_C, API) * 0.0689476)
        dados["Holdup"].append(HL)
        dados["Regime"].append(regime_str)
        dados["Vsl"].append(Vsl)
        dados["Vsg"].append(Vsg)
        dados["Vm"].append(Vm)
        dados["Bo"].append(Bo)
        dados["Bg"].append(Bg_ft3scf)
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


resultados = {}
for D_pol in DIAMETROS_POL:
    D_m = D_pol * 0.0254
    print(f"Simulando D = {D_pol} pol ({D_m:.4f} m)...")
    df, L_total = simular(D_m, sections)
    P_chegada = df["P_bar"].iloc[-1]
    print(f"  Pressão de chegada: {P_chegada:.2f} bar | {'SUCESSO' if P_chegada >= 1.013 else 'FALHA'}")
    resultados[D_pol] = df
    df.to_excel(f"Relatorio_no Excel_{D_pol}pol.xlsx", index=False)

print("\nArquivos Excel gerados.")


# GRÁFICOS
def estilo_padrao():
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.axvline(L_anm,      color="gray", linestyle="--", lw=1.2)
    plt.axvline(L_manifold, color="gray", linestyle="--",  lw=1.2)
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
    ax.axvline(L_manifold, color="gray", linestyle="--",  lw=1.2)
    fig.tight_layout()
    nome = var.replace("/", "_")
    fig.savefig(f"BB_{nome}.png", dpi=150)
    plt.close(fig)

print("Gerando gráfico comparativo de Pressão...")

fig_p, ax_p = plt.subplots(figsize=(11, 6))

# Plota a pressão para cada diâmetro
for D_pol, cor in zip(DIAMETROS_POL, CORES):
    df = resultados[D_pol]
    ax_p.plot(df["L_m"], df["P_bar"], color=cor, lw=2, label=f"Pressão ({D_pol} pol)")

# Plota a Pressão de Bolha (usando o 1º diâmetro como referência)
ax_p.plot(resultados[DIAMETROS_POL[0]]["L_m"],
         resultados[DIAMETROS_POL[0]]["Pb_bar"],
         color="green", lw=1.5, linestyle="-", label="Pressão de Bolha")

# Plota a Pressão Atmosférica
ax_p.axhline(1.01325, color="black", linestyle="--", lw=1.5, label="Pressão Atm.")
ax_p.set_title("Pressão ao longo do sistema — Beggs & Brill", fontsize=14)
ax_p.set_xlabel("Comprimento (m)")
ax_p.set_ylabel("Pressão (bar)")
ax_p.legend(title="Legenda", fontsize=11)
estilo_padrao()
fig_p.savefig("BB_Pressao_Comparativa_Pb.png", dpi=150)
plt.close(fig_p)

# BLOCO DE PLOTAGEM INDIVIDUAL
print("\nGerando e salvando gráficos individuais...")

# DataFrame de referência (6 pol) para os gráficos que usam apenas uma curva
df = resultados[DIAMETROS_POL[0]]


# PRESSÃO COMPARATIVA (Diâmetros vs Pressão de Bolha)
fig, ax = plt.subplots(figsize=(11, 6))
for D_pol, cor in zip(DIAMETROS_POL, CORES):
    ax.plot(resultados[D_pol]["L_m"], resultados[D_pol]["P_bar"], color="blue", lw=2, label=f"Pressão ({D_pol} pol)")
ax.plot(df["L_m"], df["Pb_bar"], color="green", lw=1.5, linestyle="-", label="Pressão de Bolha")
ax.axhline(1.01325, color="black", linestyle="--", lw=1.5, label="Pressão Atm.")
ax.set_title("Pressão ao longo do sistema — Beggs & Brill", fontsize=14)
ax.set_xlabel("Comprimento (m)")
ax.set_ylabel("Pressão (bar)")
ax.legend(title="Legenda", fontsize=11)
estilo_padrao()
fig.savefig("BB_Pressao_Comparativa_Pb.png", dpi=150)
plt.close(fig)


# # PRESSÃO DE BOLHA
# fig, ax = plt.subplots(figsize=(11, 6))
# ax.plot(df["L_m"], df["Pb_bar"], lw=2, color="blue")
# ax.set_title("Pressão de Bolha ao Longo do Sistema", fontsize=14)
# ax.set_xlabel("Comprimento (m)")
# ax.set_ylabel("Pb (bar)")
# estilo_padrao()
# fig.savefig("BB_Pb_bar.png", dpi=150)
# plt.close(fig)


# TEMPERATURA
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(df["L_m"], df["T_C"], lw=2, color="blue")
ax.set_title("Temperatura do Fluido ao Longo do Sistema", fontsize=14)
ax.set_xlabel("Comprimento (m)")
ax.set_ylabel("T (°C)")
estilo_padrao()
fig.savefig("BB_T_C.png", dpi=150)
plt.close(fig)


# HOLDUP LÍQUIDO
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(df["L_m"][1:], df["Holdup"][1:], lw=2, color="blue")
ax.set_title("Holdup Líquido", fontsize=14)
ax.set_xlabel("Comprimento (m)")
ax.set_ylabel("HL")
estilo_padrao()
fig.savefig("BB_Holdup.png", dpi=150)
plt.close(fig)


# FATOR DE VOLUME DO ÓLEO (Bo)
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(df["L_m"][1:], df["Bo"][1:], lw=2, color="blue")
ax.set_title("Fator de Volume de Formação do Óleo (Bo)", fontsize=14)
ax.set_xlabel("Comprimento (m)")
ax.set_ylabel("Bo (m³/sm³)")
estilo_padrao()
fig.savefig("BB_Bo.png", dpi=150)
plt.close(fig)


# FATOR DE VOLUME DO GÁS (Bg)
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(df["L_m"][1:], df["Bg"][1:], lw=2, color="blue")
ax.set_title("Fator de Volume de Formação do Gás (Bg)", fontsize=14)
ax.set_xlabel("Comprimento (m)")
ax.set_ylabel("Bg (m³/sm³)")
estilo_padrao()
fig.savefig("BB_Bg.png", dpi=150)
plt.close(fig)


# FATORES DE VOLUME DE FORMAÇÃO (Bo e Bg) JUNTOS
fig, ax1 = plt.subplots(figsize=(11, 6))
linha1 = ax1.plot(df["L_m"][1:], df["Bo"][1:], lw=2, color="#8c564b", label="Bo (Óleo)")
ax1.set_xlabel("Comprimento (m)")
ax1.set_ylabel("Bo (m³/sm³)", color="#8c564b", fontweight='bold')
ax1.tick_params(axis='y', labelcolor="#8c564b")
ax2 = ax1.twinx()
linha2 = ax2.plot(df["L_m"][1:], df["Bg"][1:], lw=2, color="#1f77b4", label="Bg (Gás)")
ax2.set_ylabel("Bg (m³/sm³)", color="#1f77b4", fontweight='bold')
ax2.tick_params(axis='y', labelcolor="#1f77b4")
linhas = linha1 + linha2
labels = [l.get_label() for l in linhas]
ax1.legend(linhas, labels, loc="upper center", fontsize=11)
plt.title("Fatores de Volume de Formação (Bo e Bg)", fontsize=14)
estilo_padrao()
fig.savefig("BB_Bo_Bg_Composto.png", dpi=150)
plt.close(fig)


# VELOCIDADE SUPERFICIAL DO GÁS (Vsg)
# fig, ax = plt.subplots(figsize=(11, 6))
# ax.plot(df["L_m"], df["Vsg"], lw=2, color="blue")
# ax.set_title("Velocidade Superficial do Gás (Vsg)", fontsize=14)
# ax.set_xlabel("Comprimento (m)")
# ax.set_ylabel("Vsg (m/s)")
# estilo_padrao()
# fig.savefig("BB_Vsg.png", dpi=150)
# plt.close(fig)


# VELOCIDADE SUPERFICIAL DO LÍQUIDO (Vsl)

# fig, ax = plt.subplots(figsize=(11, 6))
# ax.plot(df["L_m"], df["Vsl"], lw=2, color="blue")
# ax.set_title("Velocidade Superficial do Líquido (Vsl)", fontsize=14)
# ax.set_xlabel("Comprimento (m)")
# ax.set_ylabel("Vsl (m/s)")
# estilo_padrao()
# fig.savefig("BB_Vsl.png", dpi=150)
# plt.close(fig)


# VELOCIDADE DA MISTURA (Vm)

# fig, ax = plt.subplots(figsize=(11, 6))
# ax.plot(df["L_m"], df["Vm"], lw=2, color="blue")
# ax.set_title("Velocidade Espacial da Mistura (Vm)", fontsize=14)
# ax.set_xlabel("Comprimento (m)")
# ax.set_ylabel("Vm (m/s)")
# estilo_padrao()
# fig.savefig("BB_Vm.png", dpi=150)
# plt.close(fig)


# GRADIENTES DE PRESSÃO INDIVIDUAIS

gradientes = [
    ("dp_fric", "Gradiente de Fricção (Atrito)", "blue"),
    ("dp_grav", "Gradiente Gravitacional (Elevação)", "green"),
    ("dp_acc", "Gradiente de Aceleração (Cinético)", "red"),
    ("dp_total", "Gradiente Total de Pressão", "black")
]
for var, titulo, cor in gradientes:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["L_m"][1:], df[var][1:], lw=2, color=cor)
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel("Comprimento (m)")
    ax.set_ylabel("dP/dL (bar/m)")
    estilo_padrao()
    fig.savefig(f"BB_{var}.png", dpi=150)
    plt.close(fig)


# DENSIDADES INDIVIDUAIS
densidades = [
    ("rho_o", "Densidade In-Situ do Óleo", "#8c564b"),
    ("rho_w", "Densidade In-Situ da Água", "#1f77b4"),
    ("rho_g", "Densidade In-Situ do Gás", "#ff7f0e"),
    ("rho_liq", "Densidade da Fase Líquida Combinada", "#2ca02c"),
    ("rho_mix", "Densidade Hidrodinâmica da Mistura (Slip)", "#7f7f7f")
]
for var, titulo, cor in densidades:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["L_m"][1:], df[var][1:], lw=2, color=cor)
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel("Comprimento (m)")
    ax.set_ylabel("Densidade (kg/m³)")
    estilo_padrao()
    fig.savefig(f"BB_{var}.png", dpi=150)
    plt.close(fig)


# VISCOSIDADES INDIVIDUAIS

viscosidades = [
    ("mu_o", "Viscosidade do Óleo", "#8c564b"),
    ("mu_g", "Viscosidade do Gás", "#ff7f0e"),
    ("mu_mix", "Viscosidade da Mistura Multifásica", "#7f7f7f")
]
for var, titulo, cor in viscosidades:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["L_m"][1:], df[var][1:], lw=2, color=cor)
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel("Comprimento (m)")
    ax.set_ylabel("Viscosidade (Pa.s)")
    estilo_padrao()
    fig.savefig(f"BB_{var}.png", dpi=150)
    plt.close(fig)



# COMPOSTO: VISCOSIDADES JUNTAS
fig, ax = plt.subplots(figsize=(11, 6))
# Plota cada viscosidade manualmente com a cor que você escolher
ax.plot(df["L_m"][1:], df["mu_o"][1:], lw=2, color="#8c564b", label="Óleo")
ax.plot(df["L_m"][1:], df["mu_g"][1:], lw=2, color="#ff7f0e", label="Gás")
ax.plot(df["L_m"][1:], df["mu_mix"][1:], lw=2, color="#7f7f7f", label="Mistura")
ax.set_title("Perfil de Viscosidade das Fases", fontsize=14)
ax.set_xlabel("Comprimento (m)")
ax.set_ylabel("Viscosidade (Pa.s)")
ax.legend(fontsize=11)
estilo_padrao()
fig.savefig("BB_Viscosidades_Composto.png", dpi=150)
plt.close(fig)


# COMPOSTO: TERMOS DA PERDA DE CARGA
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(df["L_m"][1:], df["dp_fric"][1:], lw=2, color="blue", label='Fricção')
ax.plot(df["L_m"][1:], df["dp_grav"][1:], lw=2, color="green", label='Gravidade')
ax.plot(df["L_m"][1:], df["dp_acc"][1:], '--', lw=2, color="red", label='Aceleração')
ax.set_title(f"Análise dos Componentes de Perda de Carga ({DIAMETROS_POL[0]} pol)", fontsize=14)
ax.set_xlabel("Comprimento (m)")
ax.set_ylabel("dP/dL (bar/m)")
ax.legend(fontsize=11)
estilo_padrao()
fig.savefig("BB_Termos_Perda_Carga_Composto.png", dpi=150)
plt.close(fig)


# COMPOSTO: COMPONENTES DAS DENSIDADES
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(df["L_m"][1:], df["rho_o"][1:], lw=2, color="green", label='Óleo')
ax.plot(df["L_m"][1:], df["rho_w"][1:], lw=2, color="blue", label='Água')
ax.plot(df["L_m"][1:], df["rho_g"][1:], lw=2, color="purple", label='Gás')
ax.plot(df["L_m"][1:], df["rho_liq"][1:], lw=2, color="red", label='Líquido')
ax.plot(df["L_m"][1:], df["rho_mix"][1:],"--", lw=2, color="black", label='Mistura')
ax.set_title(f"Perfil de Massa Específica das Fases ({DIAMETROS_POL[0]} pol)", fontsize=14)
ax.set_xlabel("Comprimento (m)")
ax.set_ylabel("Massa Específica (kg/m³)")
ax.legend(fontsize=11)
estilo_padrao()
fig.savefig("BB_Componentes_Densidade_Composto.png", dpi=150)
plt.close(fig)

print("Todos os gráficos foram customizados de forma independente e salvos com sucesso!")


# MAPA HORIZONTAL DE PADRÕES DE ESCOAMENTO
print("Gerando gráfico do perfil de padrões de escoamento...")

# Mapeia os regimes para cores específicas para manter consistência visual
cores_regime = {
    "Monofásico": "#2ca02c",   # Verde
    "Segregado": "#1f77b4",    # Azul
    "Transição": "#ff7f0e",    # Laranja
    "Intermitente": "#d62728", # Vermelho
    "Distribuído": "#9467bd"   # Roxo
}

fig, ax = plt.subplots(figsize=(11, 2.5)) # Gráfico mais "achatado" estilo barra

L = df["L_m"].values
regimes = df["Regime"].values

for i in range(len(L) - 1):
    regime_atual = regimes[i]
    cor = cores_regime.get(regime_atual, "#7f7f7f")
    # Desenha um retângulo vertical cobrindo o passo atual dL
    ax.axvspan(L[i], L[i+1], color=cor, alpha=0.8)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cor, label=regime) for regime, cor in cores_regime.items() if regime in regimes]
ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5), title="Regimes")

# Configurações de eixos
ax.set_title("Perfil de Padrões de Escoamento ao Longo do Sistema", fontsize=14)
ax.set_xlabel("Comprimento (m)")
ax.set_yticks([]) # Remove o eixo Y já que a informação está nas cores
ax.set_xlim(L[0], L[-1])
estilo_padrao()
fig.savefig("BB_Perfil_Regimes_Escoamento.png", dpi=150)
plt.close(fig)
print("Gerando gráfico comparativo de Pressão com fundo de Regimes...")

fig_p, ax_p = plt.subplots(figsize=(11, 6))

cores_regime = {"Monofásico": "#2ca02c", "Segregado": "#1f77b4", "Transição": "#ff7f0e", "Intermitente": "#d62728", "Distribuído": "#9467bd"}
L = df["L_m"].values
regimes = df["Regime"].values

for i in range(len(L) - 1):
    ax_p.axvspan(L[i], L[i+1], color=cores_regime.get(regimes[i], "#7f7f7f"), alpha=0.15) # alpha baixo para ficar suave ao fundo

for D_pol, cor in zip(DIAMETROS_POL, CORES):
    df_res = resultados[D_pol]
    ax_p.plot(df_res["L_m"], df_res["P_bar"], color=cor, lw=2, label=f"Pressão ({D_pol} pol)")

ax_p.plot(df["L_m"], df["Pb_bar"], color="green", lw=1.5, linestyle="-", label="Pressão de Bolha")
ax_p.axhline(1.01325, color="black", linestyle="--", lw=1.5, label="Pressão Atm.")

from matplotlib.patches import Patch
legenda_linhas = ax_p.legend(loc="upper right", title="Variáveis")
legenda_regimes = [Patch(facecolor=cor, alpha=0.3, label=regime) for regime, cor in cores_regime.items() if regime in regimes]
ax_p.legend(handles=legenda_regimes, loc="lower left", title="Regimes de Fundo")
ax_p.add_artist(legenda_linhas)

ax_p.set_title("Pressão ao longo do sistema — Beggs & Brill", fontsize=14)
ax_p.set_xlabel("Comprimento (m)")
ax_p.set_ylabel("Pressão (bar)")
ax_p.set_xlim(L[0], L[-1])

estilo_padrao()

fig_p.savefig("BB_Pressao_Padroesesc_Pb.png", dpi=150)
plt.close(fig_p)


# GRÁFICO DE DEGRAUS DOS PADRÕES DE ESCOAMENTO (STEP CHART)
print("Gerando gráfico de degraus dos regimes (Step Chart)...")

fig, ax = plt.subplots(figsize=(11, 4))

df_step = resultados[DIAMETROS_POL[0]]

niveis_regime = {
    "Monofásico": 1,
    "Segregado": 2,
    "Transição": 3,
    "Intermitente": 4,
    "Distribuído": 5
}

y_numerico = [niveis_regime.get(regime, 0) for regime in df_step["Regime"]]

# Plotamos o gráfico de degraus
# O parâmetro where='post' é o que garante que a linha ande reto para a direita
# e só mude de altura quando chegar na próxima coordenada X (exatamente como no seu desenho)
ax.step(df_step["L_m"], y_numerico, where='post', color='black', linewidth=1.5)

# Trocamos os números do eixo Y pelos nomes dos regimes
ax.set_yticks(list(niveis_regime.values()))
ax.set_yticklabels(list(niveis_regime.keys()), fontsize=11)

ax.set_ylim(0.5, 5.5)  # Dá um "respiro" acima e abaixo da linha
ax.set_xlim(df_step["L_m"].min(), df_step["L_m"].max())

ax.set_title(f"Evolução do Padrão de Escoamento ({DIAMETROS_POL[0]} pol)", fontsize=14)
ax.set_xlabel("Comprimento (m)", fontsize=12)
ax.set_ylabel("Padrão de Escoamento", fontsize=12)

ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.grid(axis='x', linestyle=':', alpha=0.4)

ax.axvline(L_anm, color="gray", linestyle="--", lw=1.2)
ax.axvline(L_manifold, color="gray", linestyle="--", lw=1.2)

fig.tight_layout()
fig.savefig("BB_Regimes_StepChart.png", dpi=150)
plt.close(fig)