import pandas as panda
from datetime import datetime, timedelta
import os
import numpy as numpy
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib


# função que salva os dados no arquivo pra eu poder usar depois
def salvar_dados_csv(houve_vazamento, dia, tipo_vazamento, data):

    data = datetime.now().strftime("%Y-%m-%d")
    dataframe = panda.DataFrame(
        {
            "houve_vazamento": [houve_vazamento],
            "dia": [dia],
            "tipo_vazamento": [tipo_vazamento],
            "data": [data],
        }
    )

    if os.path.exists("dados_incidentes.csv"):
        existing_data = panda.read_csv("dados_incidentes.csv", encoding="utf-8")
        dataframe = panda.concat([existing_data, dataframe], ignore_index=True)

    dataframe.to_csv("dados_incidentes.csv", index=False)
    print("os dados foram salvos corretamente no arquivo .csv")


# a Função que carrega os dados do .csv
def carregar_dados():
    try:
        dataframe = panda.read_csv("dados_incidentes.csv", parse_dates=["data"])
        dataframe["diasemana"] = dataframe["data"].dt.dayofweek
        print("Os dados foram carregados")
        return dataframe
    except FileNotFoundError:
        print("Erro: o arquivo dados_incidentes não foi encontrado!")
        return panda.DataFrame()


# função que eu usei pra filtrar os dados (tava dando erro de filtragem na acuracia e eu tive que filtrar)
def filtrar_dados(data):
    data["data"] = panda.to_datetime(data["data"])
    data["diasemana"] = data["data"].dt.dayofweek
    return data


# aqui eu fiz o treinamento do modelo da minha IA
def treinamento(features, target):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = (
        X_scaled,
        X_scaled,
        target,
        target,
    )
    # basicamente instalando o modelo. o Random state tá definido pra todas as vezes que eu executar o código com os mesmos dados
    # ele me dê uma resposta UNICA e precisa.
    model = DecisionTreeClassifier(random_state=42)
    model.fit(
        X_train, y_train
    )  # aqui, ele minimiza a perca de 20% dos dados para treino e aprende a relação (a diferença entre as caracteristicas (X) e as labels (y))

    joblib.dump(
        (model, scaler), "modelo_ajustado_e_scaler.pkl"
    )  # aqui é um código pra eu salvar o meu modelo (se não, não dá pra usar KKK)

    print(
        f"Acurácia do modelo: {accuracy_score(y_test, model.predict(X_test)):.2f}"
    )  # aqui é o calculo de acuracia do modelo! no caso 1.00 significa 100% de acuracia :D
    return model, scaler


# Função de previsão que foi alterada 500 mil vezes
def previsao_prox_mes(model, scaler, dados_reais):
    now = datetime.now()
    next_month_start = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
    next_month_end = (next_month_start + timedelta(days=32)).replace(day=1) - timedelta(
        days=1
    )

    proximomes = panda.date_range(start=next_month_start, end=next_month_end, freq="D")

    n_samples = len(proximomes)

    # aqui eu extraio os dados do DataFrame (que foi gerado após ler o arquivo .csv através do panda)
    dias_novos = numpy.linspace(
        dados_reais["dia"].min(), dados_reais["dia"].max(), n_samples
    )
    tipos_novos = numpy.linspace(
        dados_reais["tipo_vazamento"].min(),
        dados_reais["tipo_vazamento"].max(),
        n_samples,
    )

    # Aqui eu crio um dataframe caso não haja com os novos valores.
    features = panda.DataFrame({"dia": dias_novos, "tipo_vazamento": tipos_novos})

    # aqui eu dou uma checada pra ver se tá tudo certo antes de fazer a escala da média (depuração)
    if (
        features["dia"].min() < dados_reais["dia"].min()
        or features["tipo_vazamento"].min() < dados_reais["tipo_vazamento"].min()
    ):
        print(
            "Aviso!!! Alguma característica está pra fora do intervalo esperado após a geração."
        )

    # basicamente essa pequena função mas super importante faz com que eu escale os dados do jeito que eu treinei la em cima no scaler!
    #
    X_scaled = scaler.transform(features)

    # Fazer previsões
    probabilidades = model.predict_proba(X_scaled)

    previsoes = panda.DataFrame(
        {"data": proximomes, "probabilidadevazamento": probabilidades[:, 1]}
    )
    previsoes["diasemana"] = previsoes["data"].dt.dayofweek

    return previsoes


# Formatação básica das previsões
def mostrar_previsoes_formatadas(previsoes):
    dias_semana = [
        "Segunda feira",
        "Terça feira",
        "Quarta feira",
        "Quinta feira",
        "Sexta feira",
        "Sábado",
        "Domingo",
    ]

    previsoesjuntas = previsoes.groupby("diasemana").mean()

    for diasemana, grupo in previsoesjuntas.iterrows():
        print(
            f"{dias_semana[diasemana]} - Probabilidade: {grupo['probabilidadevazamento'] * 100:.2f}%"
        )


# Menu básico com opções para navegar
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
                    houve_vazamento = int(input("Houve vazamento (0 = Não, 1 = Sim): "))
                    if houve_vazamento in [0, 1]:
                        break
                    else:
                        print("O valor deve ser 0 para Sim e 1 para não!")
                except ValueError:
                    print("Por favor, coloque um valor valido.")

            while True:
                try:
                    dia = int(input("Dia do mês (01 a 31): "))
                    if 1 <= dia <= 31:
                        break
                    else:
                        print("Erro: O dia deve ser um número entre 01 e 31.")
                except ValueError:
                    print("Erro, por favor insira um valor valido")

            while True:
                try:
                    tipo_vazamento = int(input("Tipo de vazamento (1, 2 ou 3): "))
                    if tipo_vazamento in [1, 2, 3]:
                        break
                    else:
                        print("O Tipo de vazamento é entre 1, 2 ou 3 !")
                except ValueError:
                    print("Erro: Valor invalido, por favor insira um valor correto.")

            data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            salvar_dados_csv(houve_vazamento, dia, tipo_vazamento, data)

        elif escolha == "2":
            data = carregar_dados()
            if not data.empty:
                data_filtrada = filtrar_dados(data)
                features = data_filtrada[["dia", "tipo_vazamento"]]
                target = data_filtrada["houve_vazamento"]
                if model is None or scaler is None:
                    model, scaler = treinamento(features, target)
                previsoes = previsao_prox_mes(model, scaler, data_filtrada)
                mostrar_previsoes_formatadas(previsoes)

        elif escolha == "0":
            print("Saindo.")
            break

        else:
            print("Opção não é valida. Tente novamente!")


if __name__ == "__main__":
    main()
