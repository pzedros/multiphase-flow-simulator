import numpy as np
import matplotlib.pyplot as plt
import functions_flow as fc

# -----------------------------------------------------------------
# DADOS DE ENTRADA PRINCIPAIS
# -----------------------------------------------------------------
API = 20          # grau API do óleo
dg = 0.75         # gravidade específica do gás
RGL = 130         # razão gás-óleo na superfície [sm³/sm³]
BSW = 0.20        # corte de água
T = 90.0          # temperatura do reservatório [°C]
P = 250 #bar

Tf = T * 9 / 5 + 32

pho_w = 965.3 #kg/m³
# Cálculo inicial mantendo sua lógica exata


Pb = fc.pressao_bolha(RGL, dg, T, API)
Pbar = Pb * 0.0689476
print(Pbar)



do = 141.5/(API+131.5)

# -----------------------------------------------------------------
# GERAÇÃO DOS DADOS PARA O GRÁFICO
# -----------------------------------------------------------------
# Vetor de pressões em bar
pressures = np.linspace(1, 250, 100)
Rs_list = []
Co_list = []
Bo_list = []
Pho_list = []
Mu_obs_list = []
Rs_w_list = []
Bw_list = []
Bg_list = []
Z_list = []

for P in pressures:
    Rs = fc.rs_gas_oil(dg, P, Pb, API, T)

    Bob = 0.9759 + 0.00012 * (RGL * (dg / do) ** 0.5 + 1.25 * Tf) ** 1.2
    pho_ob = (62.4 * do + 0.0136 * Rs * dg) / Bob

    Ppc, Tpc, Ppr, Tpr = fc.propseudo(dg,P,T)
    Z = fc.factor_Z(Ppc, Tpc, Ppr, Tpr)
    Bg = fc.bg(T,P,Z)

    Co = fc.compressibilidade_oleo(P, Tf, dg, do, Rs, Bob, API, Bg, Pb, pho_ob)

    Bo = fc.bo(Rs, dg, do, T, P, Pb, Co, RGL)

    pho_o = fc.pho_o_insitu(do, Rs, dg, Bo, Bob, P, Pb, Co, RGL, T)


    mu_od = fc.viscosidade_oleomorto(Tf, API)
    mu_obs = fc.viscosidade_oleosaturado(Rs, mu_od, P, Pb)

    # Agua
    rs_w = fc.rs_w(P, Tf)
    bw = fc.bw(T, P)

    Bg = fc.bg(T, P, Z)
          
    Rs_list.append(Rs)
    Co_list.append(Co)
    Bo_list.append(Bo)
    Pho_list.append(pho_o)
    Mu_obs_list.append(mu_obs)

    Rs_w_list.append(rs_w)
    Bw_list.append(bw)

    Bg_list.append(Bg)
    Z_list.append(Z)    

# -----------------------------------------------------------------
# PLOTAGEM DO GRÁFICO Òleo
# -----------------------------------------------------------------
#Gráfico do Rs
plt.figure(figsize=(8, 5))
plt.plot(pressures, Rs_list, color='#5A7D9A', lw=2, label='Rs')
plt.title('Rs por Pressão', fontsize=12)
plt.xlabel('Pressão [bar]', fontsize=11)
plt.ylabel('Rs', fontsize=11)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()

#Gráfico do Co
plt.figure(figsize=(8, 5))
plt.plot(pressures, Co_list, color='#5A7D9A', lw=2, label='Co')
plt.title('Co por Pressão', fontsize=12)
plt.xlabel('Pressão [bar]', fontsize=11)
plt.ylabel('Co', fontsize=11)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()

#Gráfico do Bo
plt.figure(figsize=(8, 5))
plt.plot(pressures, Bo_list, color='#5A7D9A', lw=2, label='Bo')
plt.title('Bo por Pressão', fontsize=12)
plt.xlabel('Pressão [bar]', fontsize=11)
plt.ylabel('Bo', fontsize=11)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()

#Gráfico do pho_o
plt.figure(figsize=(8, 5))
plt.plot(pressures, Pho_list, color='#5A7D9A', lw=2, label='pho_o')
plt.title('pho_o por Pressão', fontsize=12)
plt.xlabel('Pressão [bar]', fontsize=11)
plt.ylabel('pho_o', fontsize=11)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()

#Gráfico do Viscosidade do oleo
plt.figure(figsize=(8, 5))
plt.plot(pressures, Mu_obs_list, color='#5A7D9A', lw=2, label='mu_o')
plt.title('mu_o por Pressão', fontsize=12)
plt.xlabel('Pressão [bar]', fontsize=11)
plt.ylabel('mu_o', fontsize=11)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()



# -----------------------------------------------------------------
# PLOTAGEM DO GRÁFICO Agua
# -----------------------------------------------------------------

#Gráfico do Rs_w
plt.figure(figsize=(8, 5))
plt.plot(pressures, Rs_w_list, color='#5A7D9A', lw=2, label='Rs_w')
plt.title('Rs_w por Pressão', fontsize=12)
plt.xlabel('Pressão [bar]', fontsize=11)
plt.ylabel('Rs_w', fontsize=11)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()

#Gráfico do Bw
plt.figure(figsize=(8, 5))
plt.plot(pressures, Bw_list, color='#5A7D9A', lw=2, label='Bw')
plt.title('Bw por Pressão', fontsize=12)
plt.xlabel('Pressão [bar]', fontsize=11)
plt.ylabel('Bw', fontsize=11)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()


#Plot Gas

#Gráfico do Bg
plt.figure(figsize=(8, 5))
plt.plot(pressures, Bg_list, color='#5A7D9A', lw=2, label='Bg')
plt.title('Bg por Pressão', fontsize=12)
plt.xlabel('Pressão [bar]', fontsize=11)
plt.ylabel('Bg', fontsize=11)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()

#Gráfico do Z
plt.figure(figsize=(8, 5))
plt.plot(pressures, Z_list, color='#5A7D9A', lw=2, label='Z')
plt.title('Z por Pressão', fontsize=12)
plt.xlabel('Pressão [bar]', fontsize=11)
plt.ylabel('Z', fontsize=11)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()

plt.show()

# # -----------------------------------------------------------------
# # PLOTAGEM DOS GRÁFICOS (EM UMA SÓ TELA)
# # -----------------------------------------------------------------
# # Cria uma grade de 3 linhas por 2 colunas para os 5 gráficos
# fig, axs = plt.subplots(3, 2, figsize=(14, 12))
# fig.subplots_adjust(hspace=0.4, wspace=0.3)
# cor = '#5A7D9A'
#
# # 1. Gráfico do Rs
# axs[0, 0].plot(pressures, Rs_list, color=cor, lw=2, label='Rs')
# axs[0, 0].set_title('Rs por Pressão', fontsize=12)
# axs[0, 0].set_xlabel('Pressão [bar]', fontsize=11)
# axs[0, 0].set_ylabel('Rs', fontsize=11)
# axs[0, 0].grid(True, linestyle='-', alpha=0.7)
# axs[0, 0].legend()
#
# # 2. Gráfico do Co
# axs[0, 1].plot(pressures, Co_list, color=cor, lw=2, label='Co')
# axs[0, 1].set_title('Co por Pressão', fontsize=12)
# axs[0, 1].set_xlabel('Pressão [bar]', fontsize=11)
# axs[0, 1].set_ylabel('Co', fontsize=11)
# axs[0, 1].grid(True, linestyle='-', alpha=0.7)
# axs[0, 1].legend()
#
# # 3. Gráfico do Bo
# axs[1, 0].plot(pressures, Bo_list, color=cor, lw=2, label='Bo')
# axs[1, 0].set_title('Bo por Pressão', fontsize=12)
# axs[1, 0].set_xlabel('Pressão [bar]', fontsize=11)
# axs[1, 0].set_ylabel('Bo', fontsize=11)
# axs[1, 0].grid(True, linestyle='-', alpha=0.7)
# axs[1, 0].legend()
#
# # 4. Gráfico do pho_o
# axs[1, 1].plot(pressures, Pho_list, color=cor, lw=2, label='pho_o')
# axs[1, 1].set_title('pho_o por Pressão', fontsize=12)
# axs[1, 1].set_xlabel('Pressão [bar]', fontsize=11)
# axs[1, 1].set_ylabel('pho_o', fontsize=11)
# axs[1, 1].grid(True, linestyle='-', alpha=0.7)
# axs[1, 1].legend()
#
# # 5. Gráfico do mu_o (Viscosidade do óleo)
# axs[2, 0].plot(pressures, Mu_obs_list, color=cor, lw=2, label='mu_o')
# axs[2, 0].set_title('mu_o por Pressão', fontsize=12)
# axs[2, 0].set_xlabel('Pressão [bar]', fontsize=11)
# axs[2, 0].set_ylabel('mu_o', fontsize=11)
# axs[2, 0].grid(True, linestyle='-', alpha=0.7)
# axs[2, 0].legend()
#
# # Oculta o 6º espaço da grade que ficou vazio
# axs[2, 1].axis('off')
#
# plt.tight_layout()
# plt.show()