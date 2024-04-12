import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

def segundos_para_hh_mm_ss(segundos_float):
    segundos = int(segundos_float)
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segundos = segundos % 60

    return f'{horas:02d}:{minutos:02d}:{segundos:02d}'

def correlacao(filtered_df):
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    # List of numerical features to consider for scatter plots
    numerical_features = ['Area', 
                          'Quartos',
                          'Banheiros',
                          'Vagas',
                          'Preco',
                          'Preco_por_m2',
                          'Log_Preco']

    # Compute the correlation matrix
    # You can extend this to more features if needed
    corr = filtered_df[numerical_features].corr()

    # Initialize a Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Generate a heatmap using Seaborn
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)

    # Set title
    ax.set_title('Mapa de Calor - Correlação Linear')

    # Display the plot using Streamlit
    st.pyplot(fig)


def dispersao(df):

    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    col1, col2 = st.columns(2)

    with col1:
        x = st.selectbox('Argumento X:', ['Area', 'Quartos','Banheiros', 'Vagas','Preco', 'Preco_por_m2', 'Log_Preco'])

    with col2:
        y = st.selectbox('Argumento Y:', ['Preco', 'Preco_por_m2', 'Log_Preco', 'Area', 'Quartos', 'Banheiros'])

    cor = st.selectbox('Coloração por:', ['Preco Abaixo do Mercado', 'Regiao', 'Bairro', 'Quartos','Banheiros', 'Vagas', 'Erro do Modelo'])

    st.markdown('---')

    # Correlação
    corr = df[x].corr(df[y])

    # Exibir o resultado
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f'Correlação linear entre {x} e {y}:',value = round(corr,2))  
        st.metric(f'Total de anúncios:',value = len(df))  

    with col2:
        # Exibir o indicador de acordo com o valor da correlação
        if corr > 0.7:
            st.success('Correlação Forte')
        elif 0.5 <= corr <= 0.7:
            st.warning('Correlação Moderada')
        else:
            st.error('Correlação Fraca')

    # Modelo de Regressão Linear

    # Criando o modelo de regressão linear
    X = df[[x]]
    z = df[y]
    model = LinearRegression()
    model.fit(X, z)

    # Fazendo previsões
    predictions = model.predict(X)

    # Adicionando as previsões ao DataFrame
    df['Previsões'] = predictions.round(0)

    # Calculando os erros (resíduos)
    errors = z - predictions
    df['Erro do Modelo'] = errors.round(0)
    df['ID'] = df.index

    # Criando o gráfico de dispersão com Plotly
    fig = px.scatter(df, x=x, y=y, color=cor,
                        title=f'Gráfico de Dispersão: {x} vs. {y}',
                        labels={
                            f'{x}': f'{x}', f'{y}': f'{y}'},
                        hover_data={'ID': True,
                                    'Descricao' : True},
                        width=900, height=600)

    # Adicionando texto de ID a cada ponto
    #fig.update_traces(text='ID', hoverinfo='text')

    # Adicionando texto de descrição a cada ponto
    #fig.update_traces(text='Descricao', hoverinfo='text')

    # Definindo a ação ao clicar no ponto
    fig.update_traces(marker=dict(size=7))

    # Aumentando o tamanho da fonte nos eixos x e y
    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=17)  # Define o tamanho da fonte no eixo x
        ),
        yaxis=dict(
            tickfont=dict(size=17)  # Define o tamanho da fonte no eixo y
        ),
        title=dict(
            font=dict(size=20)  # Define o tamanho da fonte no título
        )
    )

    # Adicionando a linha de regressão
    fig.add_scatter(x=df[x], y=df['Previsões'],
                    mode='lines', name='Linha de Regressão',
                    line=dict(color='black', width=3))

    # Mostrando o gráfico no Streamlit
    st.plotly_chart(fig)

    id = st.number_input('ID:', step = 1)
    busca_id = st.button('Buscar por ID')

    # Tabela com os dados
    if busca_id:
        st.write(df[['Link', 'Area', 'Preco',
            'Bairro', 'Previsões', 'Erro do Modelo']].loc[df.ID == id])
    else:
        st.write(df[['Link', 'Area', 'Preco',
            'Bairro', 'Previsões', 'Erro do Modelo']])


def medias(filtered_df):

    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    col1, col2 = st.columns(2)

    with col1:
        x = st.selectbox('Argumento X:', ['Regiao','Bairro', 'Classe', 'Area', 'Quartos','Banheiros', 'Vagas'])

    with col2:
        y = st.selectbox('Argumento Y:', ['Preco', 'Preco_por_m2','Log_Preco', 'Area', 'Quartos','Banheiros', 'Vagas'])    

    # Calcular a média de Preço por metro quadrado e os desvios padrão por bairro
    media_x_por_y = filtered_df.groupby(
        x)[y].mean()
    std_x_por_y = filtered_df.groupby(
        x)[y].std()

    # Criar DataFrame com média e desvio padrão
    data = pd.DataFrame({x: media_x_por_y.index,
                            'Media_y': media_x_por_y.values,
                            'Std_y': std_x_por_y.values})

    # Ordenar do maior ao menor
    data = data.sort_values(by='Media_y', ascending=False)

    # Criar o gráfico de barras com barras de erro usando Plotly Express
    fig = px.scatter(data, x=x, y='Media_y',
                        error_y='Std_y',
                        title=f'Média de {y} por {x}',
                        labels={
                            'Media_y': f'Média de {y}', x: x},
                        color=x,
                        width=800, height=600)

    # Definir rotação dos rótulos do eixo x
    fig.update_layout(xaxis_tickangle=-45)

    # Aumentando o tamanho da fonte nos eixos x e y
    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=17)  # Define o tamanho da fonte no eixo x
        ),
        yaxis=dict(
            tickfont=dict(size=17)  # Define o tamanho da fonte no eixo y
        ),
        title=dict(
            font=dict(size=20)  # Define o tamanho da fonte no título
        )
    )

    # Mostrar o gráfico no Streamlit
    st.plotly_chart(fig)

 

def histogramas(filtered_df):
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import skew, kurtosis
    from sklearn.linear_model import LinearRegression

    x = st.selectbox('Argumento X:', ['Preco','Preco_por_m2', 'Log_Preco', 'Area', 'Quartos','Banheiros', 'Vagas'])

    df_hist = filtered_df[x].dropna()

    # Criando o histograma para 'x'
    histogram_preco_por_m2 = go.Histogram(
        x=df_hist,
        histnorm='probability',
        name=f'{x} Histogram',
        marker=dict(color='#800000')
    )

    # Skewness
    skew_df_hist = skew(df_hist)

    # Curtosis
    curtosis_df_hist = kurtosis(df_hist)

    # Média e Mediana para 'x'

    st.markdown('---')

    col1, col2, col3 = st.columns(3)


    mean_x = df_hist.mean()
    median_x = df_hist.median()
    std_x = df_hist.std()

    with col1:
        st.metric('Média',round(mean_x,0))

    with col2:
        st.metric('Mediana',round(median_x,0))

    with col3:
        st.metric('Desvio Padrão', round(std_x,0))

    col1, col2,col3,col4,col5 = st.columns(5)

    with col2:
        st.metric('Skewness', round(skew_df_hist,2), help = 'Medida da falta de simetria de uma determinada distribuição de frequência.')

    with col4:
        st.metric('Curtose', round(curtosis_df_hist,2), help = 'Medida de forma que caracteriza o achatamento da curva da função de distribuição normal de probabilidade.')

    # Criando a figura para o primeiro histograma (Preco_por_m2)
    fig1 = go.Figure(data=[histogram_preco_por_m2])

    # Adicionando linhas verticais para a média e mediana com legendas
    fig1.add_vline(x=mean_x, line=dict(
        color="black", width=2, dash='dash'), name="Média")
    fig1.add_vline(x=median_x, line=dict(
        color="darkgrey", width=2, dash='dash'), name="Mediana")
    fig1.add_vline(x=mean_x + std_x, line=dict(
        color="blue", width=2, dash='dash'), name="Mediana")
    fig1.add_vline(x=mean_x - std_x, line=dict(
        color="blue", width=2, dash='dash'), name="Mediana")

    fig1.update_layout(
        title=f'Histograma de {x}',
        xaxis=dict(title=f'{x}'),
        yaxis=dict(title='Probabilidade')
    )

    # Organizando os gráficos na mesma linha
    st.plotly_chart(fig1, use_container_width=True)

def oportunidades(df):

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.model_selection import cross_val_score

    if len(df) >= 5:

        # Total de imóveis

        # Imóveis abaixo da média

        # Imóveis acima da média

        col1, col2, col3 = st.columns(3)

        with col2:
            st.metric('Total de Anúncios:', value = len(df))

        with col1:
            abaixo_media = len(df.loc[df.Preco<df["Preco"].mean()])

            st.metric('Abaixo da Média', value = abaixo_media, 
                    delta = str(round(-abaixo_media*100/len(df),0)) + '%',
                    delta_color='normal',)

        with col3:    
            acima_media = len(df.loc[df.Preco>df["Preco"].mean()])

            st.metric('Acima da Média', value = acima_media, 
                    delta = str(round(acima_media*100/len(df),0)) + '%',
                    delta_color='normal')
                  
        # Importar LabelEncoder do scikit-learn
        from sklearn.preprocessing import LabelEncoder

        # Inicializar o codificador
        le1 = LabelEncoder()
        le2 = LabelEncoder()

        # Ajustar o codificador aos bairros no DataFrame
        df['Bairro_Numerico'] = le1.fit_transform(df['Bairro'])
        df['Classe_Numerica'] = le2.fit_transform(df['Classe'])

        # Supondo que você tenha carregado sua base de dados em um DataFrame chamado 'dados'
        # Exemplo:
        # dados = pd.read_csv('seu_arquivo.csv')

        # Selecionando as características relevantes para a precificação
        X = df[['Area', 'Quartos', 'Banheiros', 'Vagas', 'Bairro_Numerico', 'Classe_Numerica']]

        # Selecionando os preços correspondentes
        y = df['Preco']

        # Dividir os dados em treino (70%) e o restante (30%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Criando e treinando o modelo de árvore de decisão
        modelo = DecisionTreeRegressor(min_samples_leaf=5)
        modelo.fit(X_train, y_train)

        # Realizando a validação cruzada com 5 folds
        scores = cross_val_score(modelo, X_test, y_test, cv=5, scoring='neg_mean_squared_error')

        # Convertendo os scores de erro para RMSE
        rmse_scores = [sqrt(abs(score)) for score in scores]

        df['Previsão'] = modelo.predict(X).round(0)

        df['Erro'] = df.Preco - df['Previsão']

        df['Pessimista'] = round(((df['Previsão'] - df.Preco - np.mean(rmse_scores))/df.Preco)*100 , 0)

        df['Provável'] = round(((df['Previsão'] - df.Preco)/df.Preco)*100 , 0)

        df['Otimista'] = round(((df['Previsão'] - df.Preco + np.mean(rmse_scores))/df.Preco)*100 , 0)

        # Exibir o resultado
        col1, col2 = st.columns(2)

        mean_price = df['Preco'].mean()

        with col1:
            st.metric('Média de Preço:', value=fr'R${mean_price:.2f}')
        with col2:
            rmse_mean = np.mean(rmse_scores)
            st.metric('Raiz do Erro Quadrático Médio (RMSE):', 
                      value=f'R${rmse_mean:.2f}',
                      help='A Raiz do Erro Quadrático Médio (RMSE) é o desvio padrão dos valores residuais (erros de previsão).')

        df = df.sort_values(by='Provável', ascending=False)

        with st.expander('Dispersão'):
            import plotly.express as px
            import plotly.graph_objects as go
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.linear_model import LinearRegression

            col1, col2 = st.columns(2)

            with col1:
                x = st.selectbox('Argumento X:', ['Area', 'Quartos','Banheiros', 'Vagas','Preco', 'Preco_por_m2', 'Log_Preco'])

            with col2:
                y = st.selectbox('Argumento Y:', ['Preco', 'Preco_por_m2', 'Log_Preco', 'Area', 'Quartos', 'Banheiros'])

            st.markdown('---')

            # Correlação
            corr = df[x].corr(df[y])

            # Exibir o resultado
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f'Correlação linear entre {x} e {y}:',value = round(corr,2))   

            with col2:
                # Exibir o indicador de acordo com o valor da correlação
                if corr > 0.7:
                    st.success('Correlação Forte')
                elif 0.5 <= corr <= 0.7:
                    st.warning('Correlação Moderada')
                else:
                    st.error('Correlação Fraca')

            df['ID'] = df.index

            # Criando o gráfico de dispersão com Plotly
            fig = px.scatter(df, x=x, y=y, color='Erro',
                                title=f'Gráfico de Dispersão: {x} vs. {y}',
                                labels={
                                    f'{x}': f'{x}', f'{y}': f'{y}'},
                                hover_data={'ID': True,
                                            'Descricao' : True},
                                width=900, height=600)

            # Adicionando texto de ID a cada ponto
            #fig.update_traces(text='ID', hoverinfo='text')

            # Adicionando texto de descrição a cada ponto
            #fig.update_traces(text='Descricao', hoverinfo='text')

            # Definindo a ação ao clicar no ponto
            fig.update_traces(marker=dict(size=7))

            # Aumentando o tamanho da fonte nos eixos x e y
            fig.update_layout(
                xaxis=dict(
                    tickfont=dict(size=17)  # Define o tamanho da fonte no eixo x
                ),
                yaxis=dict(
                    tickfont=dict(size=17)  # Define o tamanho da fonte no eixo y
                ),
                title=dict(
                    font=dict(size=20)  # Define o tamanho da fonte no título
                )
            )

            # Mostrando o gráfico no Streamlit
            st.plotly_chart(fig)

            id = st.number_input('ID:', step = 1)
            busca_id = st.button('Buscar por ID')

            # Tabela com os dados
            if busca_id:
                st.write(df[['Link','Ativo','Dias Anunciado','Area','Preco','Quartos','Banheiros',
                     'Vagas','Previsão','Erro','Pessimista', 'Provável','Otimista']].loc[df.ID == id])
            else:
                st.write(df[['Link','Ativo','Dias Anunciado','Area','Preco','Quartos','Banheiros',
                     'Vagas','Previsão','Erro','Pessimista', 'Provável','Otimista']])

        st.write(df[['Link','Ativo','Dias Anunciado','Area','Preco','Quartos','Banheiros',
                     'Vagas','Previsão','Erro','Pessimista', 'Provável','Otimista']])

        # Avaliando a importância de cada variável
        importancias = modelo.feature_importances_

        # Criando os rótulos e valores para o gráfico de pizza
        labels = [col.split('_')[0] for col in X.columns]
        values = importancias*100

        import plotly.graph_objects as go

        # Criando o gráfico de pizza
        fig = go.Figure(data=[go.Bar(x=labels, y=values)])

        # Personalizando o layout do gráfico
        fig.update_layout(title='Importância das Variáveis',
                        yaxis_range=[0, 100],
                        width=700,
                        height=500)

        # Mostrando o gráfico no Streamlit
        st.plotly_chart(fig)

        cols = st.columns(len(X.columns))

        for i in range(len(X.columns)):
            with cols[i]:
                st.metric(f'{labels[i]}:', value=str(round(importancias[i]*100, 1))+"%")
    else:
        st.error('**Base de dados muito pequena para análise!**')
        st.write('Por favor, aplique filtros que apresentem pelo menos 5 anúncios.')


def main():
    col1, col2, col3 = st.columns(3)

    with col2:
        st.title('Macondo')

    # Widget para upload de arquivo
    st.sidebar.title('Macondo')
    st.sidebar.markdown('---')
    uploaded_file = st.sidebar.file_uploader(
        "Escolha um arquivo CSV ou Excel", type=["csv", "xlsx"])

    # Verifica se o arquivo foi carregado
    if uploaded_file:
        # Verifica o tipo de arquivo e lê
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=';', index_col=0)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl', index_col=0)

        if len(df.columns) == 0:
            df = pd.read_csv(uploaded_file, index_col=0)


        func = ['Início',
                'Correlação',
                'Dispersões',
                'Médias',
                'Histogramas',
                'Oportunidades']
        
        funcao = st.sidebar.radio('Seção', func)

        # Filtro de área na sidebar
        st.sidebar.markdown('---')
        st.sidebar.title('Filtros:')

        tipo_anuncio = st.sidebar.selectbox('Selecione o tipo de anúncio:', ['Venda', 'Aluguel'])

        if tipo_anuncio == 'Aluguel':
            periodo = df['Periodo Aluguel'].unique()
            periodos_selecionadas = st.sidebar.multiselect(
                'Selecione o(s) Período(s)', periodo)
            # Filtrando os dados
            df = df[df['Periodo Aluguel'].isin(
                periodos_selecionadas)] if periodos_selecionadas else df    
        else:
            pass

        # Filtrando por Região
        regioes = df['Regiao'].unique()
        regioes_selecionadas = st.sidebar.multiselect(
            'Selecione a(s) Região(ões)', regioes)

        # Filtrando os dados
        filtered_df = df[df['Regiao'].isin(
            regioes_selecionadas)] if regioes_selecionadas else df

        # Filtrando por bairro
        bairros = filtered_df['Bairro'].unique()
        bairros_selecionados = st.sidebar.multiselect(
            'Selecione o(s) Bairro(s)', bairros)

        # Filtrando os dados
        filtered_df = filtered_df[filtered_df['Bairro'].isin(
            bairros_selecionados)] if bairros_selecionados else filtered_df

        # Filtrando por classe
        classes = df['Classe'].unique()
        classes_selecionadas = st.sidebar.multiselect(
            'Selecione a(s) Classe(s)', classes)

        # Filtrando os dados
        filtered_df = filtered_df[filtered_df['Classe'].isin(
            classes_selecionadas)] if classes_selecionadas else filtered_df

        # Organizando os inputs na mesma linha
        col1, col2 = st.sidebar.columns(2)

        with col1:
            min_area = st.number_input(
                'Área mínima [m²]', min_value=0, value=0)

        with col2:
            max_area = st.number_input(
                'Área máxima [m²]', min_value=min_area, value=1000)

        # Organizando os inputs na mesma linha
        col1, col2 = st.sidebar.columns(2)

        with col1:
            min_pm2 = st.number_input(
                'Limite inferior de Preço/m² [R$/m²]', min_value=0, value=0)

        with col2:
            max_pm2 = st.number_input(
                'Limite superior de Preço/m² [R$/m²]', min_value=min_pm2, value=100000)

        # Organizando os inputs na mesma linha
        col1, col2 = st.sidebar.columns(2)

        with col1:
            min_p = st.number_input(
                'Limite inferior de Preço [R$]', min_value=0, value=0)

        with col2:
            max_p = st.number_input(
                'Limite superior de Preço [R$]', min_value=min_p, value=100000000)

        # Defina o valor mínimo e máximo de quartos no DataFrame
        min_quartos = int(filtered_df['Quartos'].min())
        max_quartos = int(filtered_df['Quartos'].max())

        # Crie um range slider para selecionar o intervalo de número de quartos
        min_quartos_selected, max_quartos_selected = st.sidebar.slider('Número de quartos',
                                                                       min_value=min_quartos,
                                                                       max_value=max_quartos,
                                                                       value=(min_quartos, max_quartos))
        
        # Defina o valor mínimo e máximo de banheiros no DataFrame
        min_banheiros = int(filtered_df['Banheiros'].min())
        max_banheiros = int(filtered_df['Banheiros'].max())

        # Crie um range slider para selecionar o intervalo de número de quartos
        min_banheiros_selected, max_banheiros_selected = st.sidebar.slider('Número de banheiros',
                                                                       min_value=min_banheiros,
                                                                       max_value=max_banheiros,
                                                                       value=(min_banheiros, max_banheiros))

        # Defina o valor mínimo e máximo de quartos no DataFrame
        min_vagas = int(filtered_df['Vagas'].min())
        max_vagas = int(filtered_df['Vagas'].max())

        # Crie um range slider para selecionar o intervalo de número de quartos
        min_vagas_selected, max_vagas_selected = st.sidebar.slider('Número de vagas',
                                                                   min_value=min_vagas,
                                                                   max_value=max_vagas,
                                                                   value=(min_vagas, max_vagas))

        # Filtrando os dados
        filtered_df = filtered_df[(filtered_df['Area'] >= min_area) & (filtered_df['Area'] <= max_area)
                                  & (filtered_df['Preco_por_m2'] >= min_pm2) & (filtered_df['Preco_por_m2'] <= max_pm2)
                                  & (filtered_df['Preco'] >= min_p) & (filtered_df['Preco'] <= max_p)
                                  & (filtered_df['Quartos'] >= min_quartos_selected) & (filtered_df['Quartos'] <= max_quartos_selected)
                                  & (filtered_df['Banheiros'] >= min_banheiros_selected) & (filtered_df['Banheiros'] <= max_banheiros_selected)
                                  & (filtered_df['Vagas'] >= min_vagas_selected) & (filtered_df['Vagas'] <= max_vagas_selected)]

        filtered_df['Log_Preco'] = np.log(filtered_df.Preco)

        if funcao == 'Correlação':
            st.markdown('---')
            st.header('Correlação')
            st.markdown('---')
            correlacao(filtered_df)
        elif funcao == 'Dispersões':
            st.markdown('---')
            st.header('Dispersões')
            st.markdown('---')
            dispersao(filtered_df)            
        elif funcao == 'Médias':
            st.markdown('---')
            st.header('Médias')
            st.markdown('---')
            medias(filtered_df)
        elif funcao == 'Histogramas':
            st.markdown('---')
            st.header('Histogramas')
            st.markdown('---')
            histogramas(filtered_df)
        elif funcao == 'Oportunidades':
            st.markdown('---')
            st.header('Oportunidades')
            st.markdown('---')
            oportunidades(filtered_df)            
        else:
            st.markdown('---')
            st.write(r'"A ideia de um povo alegre, miscigenado, ordeiro, sem grandes problemas políticos, capaz de lidar com as desigualdades sociais e de gênero foi amplamente reproduzida no pensamento social"')
            st.image('macondo.jpg',width=700)
            st.write('---')
main()