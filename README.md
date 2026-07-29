 # Trabalho Final – Escoamento Multifásico

Este repositório contém o código desenvolvido para o trabalho final da disciplina de Escoamento Multifásico, elaborado em conjunto por Luara Peterle e Paulo Navegante

O projeto consiste na modelagem e simulação do escoamento multifásico em um sistema de produção offshore assistido por bombas centrífugas em série (ESP), avaliando o comportamento da pressão, temperatura e propriedades dos fluidos ao longo de todo o sistema de produção.

## Descrição do Problema

Foi considerado um poço offshore com 3000 m de profundidade, equipado com bombas centrífugas em série, cuja produção mínima requerida é de 3500 Sm³/d de petróleo. A produção é direcionada a um tanque de superfície operando à pressão atmosférica.

A geometria do sistema é composta por:

* Poço inclinado em **75°**;
* Árvore de Natal Molhada (ANM) localizada a **2000 m** de profundidade;
* Linha submarina de **1500 m** entre a ANM e o manifold;
* Manifold situado a **1500 m** de profundidade;
* Riser conectando o manifold à plataforma de produção.

## Condições Térmicas

O perfil de temperatura adotado no modelo considera:

* Temperatura do reservatório de **90 °C**;
* Variação linear da temperatura até **5 °C** no leito marinho (sucção da bomba);
* Temperatura constante de **5 °C** ao longo da linha submarina até o manifold;
* No riser, variação linear entre **5 °C** no leito marinho e **12 °C** na superfície.

## Características do Fluido

As propriedades do fluido produzidas consideradas na simulação são:

* **BSW:** 20%;
* **Razão gás-líquido (RGL):** 130 Sm³/Sm³;
* **Grau API:** 20°API;
* **Gravidade específica do gás:** 0,75;
* **Pressão do reservatório:** 250 bar;
* Fluido inicialmente em equilíbrio termodinâmico com o reservatório.

Quando requerido pelo modelo de escoamento, foram utilizadas as seguintes tensões interfaciais:

* σₒg = **0,00841 N/m**
* σwg = **0,03 N/m**

## Transferência de Calor

Os coeficientes globais de transferência de calor (TEC) adotados são:

* **Poço:** 2 W/(m·K)
* **Ambiente marinho:** 1 W/(m·K)

## Objetivo

O objetivo deste trabalho é implementar computacionalmente um modelo capaz de calcular o comportamento hidráulico e térmico do sistema de produção, determinando perfis de pressão, temperatura, propriedades PVT e demais parâmetros necessários para a análise do escoamento multifásico em um sistema de produção offshore.
