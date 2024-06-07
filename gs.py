import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# Função para salvar dados no CSV
def salvar_dados_csv(houve_vazamento, dia, tipo_vazamento, data):
    df = pd.DataFrame(
        {
            "houve_vazamento": [houve_vazamento],
            "dia": [dia],
            "tipo_vazamento": [tipo_vazamento],
            "data": [data],
        }
    )

    # Verifica se o arquivo CSV já existe
    if os.path.exists("dados_incidentes.csv"):
        existing_data = pd.read_csv("dados_incidentes.csv", encoding="utf-8")
        df = pd.concat([existing_data, df], ignore_index=True)

    df.to_csv("dados_incidentes.csv", index=False)
    print("Dados salvos no arquivo 'dados_incidentes.csv'.")


# Função para carregar dados do CSV
def carregar_dados():
    try:
        data = pd.read_csv("dados_incidentes.csv", encoding="utf-8")
        print("Dados carregados com sucesso.")
        return data
    except FileNotFoundError:
        print("Erro: Arquivo 'dados_incidentes.csv' não encontrado.")
        return pd.DataFrame()


# Função para filtrar dados
def filtrar_dados(data):
    data["data"] = pd.to_datetime(data["data"])
    data["dia_semana"] = data["data"].dt.dayofweek  # 0=Segunda, 1=Terça, etc.
    return data


# Função para treinar o modelo
def treinar_modelo(features, target):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Ajuste do tamanho do conjunto de teste
    test_size = min(0.2, len(features) - 1)  # Evita um conjunto de teste vazio

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, target, test_size=test_size, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Salvar modelo e scaler
    joblib.dump((model, scaler), "modelo_ajustado_e_scaler.pkl")

    print("Modelo e scaler salvos como 'modelo_ajustado_e_scaler.pkl'.")
    print(f"Acurácia do modelo: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    return model, scaler


# Função para fazer previsões
def prever_probabilidades(model, scaler, dados):
    X_scaled = scaler.transform(dados)
    probabilidades = model.predict_proba(X_scaled)
    return probabilidades


# Função para gerar previsões para o próximo mês
def previsao_proximo_mes(model, scaler):
    now = datetime.now()
    next_month_start = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
    next_month_end = (next_month_start + timedelta(days=32)).replace(day=1) - timedelta(
        days=1
    )

    # Seleciona todos os dias do próximo mês
    all_days = pd.date_range(start=next_month_start, end=next_month_end, freq="D")

    # Simulação de dados para a previsão
    n_samples = len(all_days)
    features = pd.DataFrame(
        {
            "houve_vazamento": np.random.randint(
                0, 2, n_samples
            ),  # 0 ou 1 para simulação
            "tipo_vazamento": np.random.randint(
                1, 4, n_samples
            ),  # Tipo de vazamento, por exemplo, 1, 2, 3
        }
    )
    X_scaled = scaler.transform(features)
    probabilidades = model.predict_proba(X_scaled)
    previsoes = pd.DataFrame(
        {"data": all_days, "probabilidade_vazamento": probabilidades[:, 1]}
    )
    previsoes["dia_semana"] = previsoes[
        "data"
    ].dt.dayofweek  # Adiciona coluna do dia da semana
    return previsoes


# Função para exibir previsões formatadas
def exibir_previsoes_formatadas(previsoes):
    dias_semana = [
        "Segunda-feira",
        "Terça-feira",
        "Quarta-feira",
        "Quinta-feira",
        "Sexta-feira",
        "Sábado",
        "Domingo",
    ]

    # Agrupar por dia da semana e calcular média das probabilidades
    previsoes_agrupadas = previsoes.groupby("dia_semana").mean()

    for dia_semana, grupo in previsoes_agrupadas.iterrows():
        print(
            f"{dias_semana[dia_semana]} - Probabilidade: {grupo['probabilidade_vazamento'] * 100:.2f}%"
        )


# Função para menu
def menu():
    print("\nMenu:")
    print("1. Inserir informações de vazamento")
    print("2. Fazer previsão de vazamentos para o próximo mês")
    print("0. Sair")
    return input("Escolha uma opção: ")


def main():
    model, scaler = None, None
    while True:
        escolha = menu()

        if escolha == "1":
            while True:
                try:
                    houve_vazamento = int(input("Houve vazamento (0 = Sim, 1 = Não): "))
                    if houve_vazamento in [0, 1]:
                        break
                    else:
                        print("Erro: O valor deve ser 0 (Sim) ou 1 (Não).")
                except ValueError:
                    print("Erro: Por favor, insira um valor válido.")

            while True:
                try:
                    dia = int(input("Dia do mês (01 a 31): "))
                    if 1 <= dia <= 31:
                        break
                    else:
                        print("Erro: O dia deve ser um número entre 01 e 31.")
                except ValueError:
                    print("Erro: Por favor, insira um número válido.")

            while True:
                try:
                    tipo_vazamento = int(input("Tipo de vazamento (1, 2 ou 3): "))
                    if tipo_vazamento in [1, 2, 3]:
                        break
                    else:
                        print("Erro: O tipo de vazamento deve ser 1, 2 ou 3.")
                except ValueError:
                    print("Erro: Por favor, insira um valor válido.")

            data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            salvar_dados_csv(houve_vazamento, dia, tipo_vazamento, data)

        elif escolha == "2":
            data = carregar_dados()
            if not data.empty:
                data_filtrada = filtrar_dados(data)
                features = data_filtrada[["houve_vazamento", "tipo_vazamento"]]
                target = data_filtrada["houve_vazamento"]
                if model is None or scaler is None:
                    model, scaler = treinar_modelo(features, target)
                previsoes = previsao_proximo_mes(model, scaler)
                exibir_previsoes_formatadas(previsoes)

        elif escolha == "0":
            print("Saindo do programa...")
            break

        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()
