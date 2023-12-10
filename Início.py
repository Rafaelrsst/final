import  streamlit as st
import streamlit as st 
import pandas as pd 
# Import libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
    
# Importing libraries
import pandas as pd
import numpy as np
import nltk
import spacy
import sklearn

import re
import contractions
nltk.download('punkt')
nltk.download('all')
from nltk.corpus import stopwords
from nltk import word_tokenize
# Used in Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
# Wordcloud to check the most used words and add appropriate ones to the stopwords list
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import warnings
warnings.filterwarnings("ignore")

from wordcloud import WordCloud







# Tempo
with st.container():
    
    import time
    import streamlit as st

    with st.spinner('Carregando dados. Por favor aguarde.'):
        time.sleep(1)
   

    





# IMPORTAR ARQUIVO
df = pd.read_csv('scopus1.csv')






# Cabeçalho
with st.container():
    st.subheader("Trabalho de Introdução à Ciência de Dados")
    st.write("Mestrado em Ciência de Dados para Ciências Sociais da Universidade de Aveiro")
    st.title("Análise de E-Commerce")
    st.write("---")
    
    
    
    
    
    
       
    
# Imagem do e-commerce e texto de contextualização e problema
with st.container():
    import streamlit as st
    from PIL import Image
    
    #Abrir a imagem com a PIL
    image = Image.open('ecommerce_image.png')

    # Adicionar a imagem usando st.image()
    st.image(image, width=690, caption='')
    
    st.write("---")
    
    st.subheader("Contextualização do trabalho")
    
    tab1, tab2, tab3 = st.tabs(["Geral", "Específica", "Motivação"])
    
    with tab1:
    
        st.write("Trabalho académico cujo objetivo principal assenta na perceção de transformações aa abordagem ao E-Commerce e Marketing ao longo do tempo (2014-2024). Para a sua execução e compreensão é mostrado, ao longo da presente app, vários meios interativos que permitem extrair e analisar esses insights, nomeadamente TreeMaps, gráficos de barras e tabelas.")
    
    with tab2:
        
        st.write("O presente aplicativo tem o propósito de perceber as modificações temporais ocorridas no E-Commerce e no Marketing ao longo do tempo. Para isso, foram feitos três principais momentos de análise. A primeira foi uma análise de bibliometria, onde se pretende mostrar dados mais quantitativos dos resultados, num segundo momento, análise de conteúdo, pode-se observar a perceção e enfoque do conteúdo escrito dos artigos analisados, e, por fim, uma análise de insights que cruza os dois indicadores anteriormente referidos. Para obter estes dados, posteriormente usados nos gráficos aqui disponivéis, realizamos diversas etapas de limpeza, tratamento e pré-processamento de dados, demos uso a ferramentas de Natural Language Processed e com isso criámos inúmeros mecanismos de estração de conhecimento.")
    
    with tab3:
        
        st.write("Num mundo cada vez mais inerente a tecnologias digitais, o E-Commerce apresenta um impacto complexo, mas significativo na vida social da grande generelidade d população. Com isto pretendemos medir como é tratado o seu impacto ao longo do tempo, através de:")
        st.write("(1) - Compreender as transformações de conteúdo/temáticas ao longo do tempo;")
        st.write("(2) - Analisar o olhar/abordagem que é dado ao tema por cada um dos artigos;")
        st.write("(3) - Entender os pricipais contribuidores, bem como a sua abordagem, em relação ao tema em questão;")
        st.write("(4) - Perspetivar o impacto da Covid-19 no tema;")
        
    st.write("---")
    
    
    
    


    
    
# processamento do clean_ abstract
with st.container():
    
    # Remove duplicate rows
    df = df.drop_duplicates(subset=['clean_Abstract'])
    
    df = df[df['clean_Abstract'] != 'abstract avail']
  
  





# nuvem de palavras, frequencia 
with st.container():
    st.subheader("Palavras mais relevantes para o contexto do trabalho")
    tab1, tab2 = st.tabs(["Wordcloud", "Frequência"])

    # Gerar a wordcloud com o abstract
    with tab1:
        
        # Criar a nuvem de palavras
        wordcloud = WordCloud(
            background_color='white', max_words=200, width=800, height=600, stopwords=STOPWORDS
        ).generate(" ".join(df["clean_Abstract"]))

        # Criar o gráfico de nuvem de palavras usando Matplotlib e exibi-lo no Streamlit
        fig, ax = plt.subplots(figsize=(16, 13))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title("", fontsize=20)
        ax.axis('off')

        # Exibir o gráfico no Streamlit
        st.pyplot(fig)
        
    




    # gerar uma st.progress para as 10 palavras mais frequentes
    with tab2:

        from collections import Counter
        import time
       
        
        # Obter as palavras mais frequentes
        word_freq = Counter(" ".join(df["clean_Abstract"]).split())
        top_words = word_freq.most_common(10)

        # Encontrar a frequência máxima para normalizar o preenchimento da barra de progresso
        max_freq = top_words[0][1]

        # Layout com texto acima e barra de progresso abaixo para cada palavra
        for word, freq in top_words:
            st.write(f"**{word}**: {freq} frequência")  # Exibir palavra e frequência acima da barra de progresso
            progress_bar = st.progress(0)  # Criar a barra de progresso
            
            for progress_count in range(freq + 1):
                progress_percentage = progress_count / max_freq  # Calcular a porcentagem de progresso
                progress_bar.progress(progress_percentage)  # Atualizar a barra de acordo com a porcentagem de progresso
                time.sleep(0.00)  # Pequeno intervalo para controlar a velocidade da barra







# treemap interativo com os insigths

st.write("---")
with st.container():
    
    import plotly.express as px
    
    st.caption("Problema em questão")
    st.subheader("Como é representado o E-Commerce ao longo do tempo? Quais as alterações e enfoques de abordagem?")
    
    # Selecionar o parametro para ver e caixas       
    modificador = st.selectbox("Selecione o indicador que deseja visualizar", ["Tipologia dos Documentos por Ano", "Frequência de Artigos por Ano", "Mutações do Conteúdo", "Mutações do Título", "Mutações das Palavras-Chave do Autor"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:   
        tabela = st.checkbox('Adicionar tabela')
    
    with col2:   
        grafico = st.checkbox('Adicionar gráfico')
     
       
    if modificador == "Tipologia dos Documentos por Ano":
        
        # Obter contagem dos tipos de documento para cada ano
        contagem_tipos_documento_por_ano = df.groupby(['Year', 'Document Type']).size().reset_index(name='Count')

        #Criar o treemap com Plotly Express
        fig = px.treemap(contagem_tipos_documento_por_ano, path=['Year', 'Document Type'], values='Count',
                        hover_data=['Count'], branchvalues='total', width=700, height=800)
        fig.update_traces(root_color="white", selector=dict(type='treemap'))
        st.plotly_chart(fig)
        
            
        # dados da tabela caso seja selecionada, so para a tipologia
        if tabela:

            # Criar uma tabela para mostrar a contagem de tipos de documento por ano
            st.write('Frequência de Tipos de Documento por Ano')
            for year in df['Year'].unique():
                st.write(f"Ano: {year}")
                df_year = contagem_tipos_documento_por_ano[contagem_tipos_documento_por_ano['Year'] == year]
                st.write(df_year)

            # Criar uma tabela com a contagem total de cada tipo de documento
            st.write('Total de Tipos de Documento')
            df_total = contagem_tipos_documento_por_ano.groupby('Document Type')['Count'].sum().reset_index(name='Total')
            st.write(df_total)


        # dados em grafico de bolhas
        if grafico:

            # Criar o gráfico de bolhas
            fig = px.scatter(contagem_tipos_documento_por_ano, x='Year', y='Document Type', size='Count', color='Document Type',
                            hover_name='Document Type', hover_data={'Year': True, 'Document Type': False, 'Count': True},
                            width=900, height=600)
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
            st.plotly_chart(fig)

            










    if modificador == "Frequência de Artigos por Ano":
        
        # Obter contagem total de documentos para cada ano
        contagem_documentos_por_ano = df['Year'].value_counts().reset_index()
        contagem_documentos_por_ano.columns = ['Year', 'Count']

        # Criar o treemap com Plotly Express
        fig = px.treemap(contagem_documentos_por_ano, path=['Year'], values='Count',
                        hover_data=['Count'], branchvalues='total', width=700, height=800)
        fig.update_traces(root_color="white", selector=dict(type='treemap'))
        st.plotly_chart(fig)
        
        
        if tabela:

            # Ordenar os dados pelo ano
            contagem_documentos_por_ano = contagem_documentos_por_ano.sort_values(by='Ano')

            # Exibir a tabela
            st.write("Contagem de Documentos por Ano:")
            st.write(contagem_documentos_por_ano)
            
              
        if grafico:

            # Criar o gráfico de bolhas
            fig = px.scatter(contagem_documentos_por_ano, x='Year', y='Count', size='Count', color='Count',
                            hover_name='Year', hover_data={'Year': True, 'Count': True},
                            width=900, height=600)
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
            st.plotly_chart(fig)
    
    
    
    
    
    
    
    
    

    # Mutações do conteudo
    
    if modificador == "Mutações do Conteúdo":
        
        tab1, tab2, tab3, tab4 = st.tabs(["N-Grams mais Frequentes", "Bigrams mais Frequentes", "Análise de Sentimento", "Classificação Gramatical"])
        
        with tab1:
            
            def obter_dados_para_treemap_e_tabela(df):
                data = []

                for ano in df['Year'].unique():
                    dados_ano = df[df['Year'] == ano]
                    texto_ano = ' '.join(dados_ano['clean_Abstract'].values)
                    palavras_tokenizadas = word_tokenize(texto_ano)
                    contagem_palavras = Counter(palavras_tokenizadas)
                    palavras_comuns = contagem_palavras.most_common(10)

                    for palavra, contagem in palavras_comuns:
                        data.append({'Year': ano, 'Palavra': palavra, 'Contagem': contagem})

                return pd.DataFrame(data)

            

            df_treemap = obter_dados_para_treemap_e_tabela(df)

            # Exibir o treemap
            fig = px.treemap(df_treemap, path=['Year', 'Palavra'], values='Contagem', width=700, height=800)
            st.plotly_chart(fig)

            # Botão para exibir a tabela
            if tabela:
                st.write('Tabela de Palavras Mais Frequentes')
                for ano in df['Year'].unique():
                    df_ano = df_treemap[df_treemap['Year'] == ano]
                    st.write(f"Ano: {ano}")
                    st.write(df_ano[['Palavra', 'Contagem']].head(10))

            # Botão para exibir o gráfico de bolhas
            if grafico:
                dados_bolha = []
                for ano in df['Year'].unique():
                    dados_ano = df[df['Year'] == ano]
                    texto_ano = ' '.join(dados_ano['clean_Abstract'].values)
                    palavras_tokenizadas = word_tokenize(texto_ano)
                    contagem_palavras = Counter(palavras_tokenizadas)
                    palavras_comuns = contagem_palavras.most_common(10)

                    for palavra, contagem in palavras_comuns:
                        dados_bolha.append({'Year': ano, 'Palavra': palavra, 'Contagem': contagem})

                df_bolha = pd.DataFrame(dados_bolha)
                fig = px.scatter(df_bolha, x='Year', y='Palavra', size='Contagem', color='Contagem',
                                hover_name='Palavra', hover_data={'Year': True, 'Palavra': False, 'Contagem': True},
                                width=900, height=600)
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
                st.plotly_chart(fig)
        
        
        
        
        
        
        
        
        
        
        
        # para os bigrams mais comuns 
        
        with tab2:
            
            from nltk import bigrams

            def obter_dados_bigrams(df):
                data_bigrams = []

                for ano in df['Year'].unique():
                    dados_ano = df[df['Year'] == ano]
                    texto_ano = ' '.join(dados_ano['clean_Abstract'].values)
                    palavras_tokenizadas = word_tokenize(texto_ano)
                    lista_bigrams = list(bigrams(palavras_tokenizadas))
                    contagem_bigrams = Counter(lista_bigrams)
                    mais_comuns_bigrams = contagem_bigrams.most_common(10)

                    for bigrama, contagem in mais_comuns_bigrams:
                        data_bigrams.append({'Year': ano, 'Bigrama': ' '.join(bigrama), 'Contagem': contagem})

                return pd.DataFrame(data_bigrams)

            
            df_bigrams = obter_dados_bigrams(df)

            #Exibindo o treemap
            fig = px.treemap(df_bigrams, path=['Year', 'Bigrama'], values='Contagem', width=700, height=800)
            st.plotly_chart(fig)

            # Botão para exibir a tabela
            if tabela:
                for ano in df['Year'].unique():
                    tabela_ano = df_bigrams[df_bigrams['Year'] == ano]
                    st.write(f"Bigrams para o ano {ano}")
                    st.write(tabela_ano)

                st.write("Bigrams mais relevantes")
                tabela_geral = df_bigrams
                st.write(tabela_geral)

            # Botão para exibir o gráfico de bolhas
            if grafico:
                fig = px.scatter(df_bigrams, x='Year', y='Bigrama', size='Contagem', color='Contagem',
                                hover_name='Bigrama', hover_data={'Year': True, 'Bigrama': False, 'Contagem': True},
                                width=900, height=600)
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
                st.plotly_chart(fig)
                
                
                
                
                
                
                
                
                
                
        
        # Análise de sentimentos 
        
        with tab3:
    
            from textblob import TextBlob

            def analyze_sentiment(text):
                analysis = TextBlob(text)
                polarity = analysis.sentiment.polarity
                
                if polarity > 0:
                    return 'Positivo'
                elif polarity == 0:
                    return 'Neutro'
                else:
                    return 'Negativo'

            
            # Aplicar a função de análise de sentimento ao campo "clean_Abstract"
            df['Sentiment'] = df['clean_Abstract'].apply(analyze_sentiment)
                
            # Contagem dos resultados da análise de sentimento por ano
            sentiment_counts = df.groupby(['Year', 'Sentiment']).size().reset_index(name='Count')
                
            # Criar o treemap com Plotly Express
            fig = px.treemap(sentiment_counts, path=['Year', 'Sentiment'], values='Count', width=700, height=800)
            fig.update_traces(root_color="white", selector=dict(type='treemap'))
            st.plotly_chart(fig)

            # Botões para exibir a tabela e o gráfico
            if tabela:
                anos_unicos = df['Year'].unique()
                for ano in anos_unicos:
                    st.write(f"Tabela de análise de sentimento para o ano {ano}")
                    tabela_ano = sentiment_counts[sentiment_counts['Year'] == ano]
                    st.write(tabela_ano)
                    
                st.write("Total de sentimento")
                tabela_geral = sentiment_counts.groupby('Sentiment')['Count'].sum().reset_index()
                st.write(tabela_geral)

            if grafico:
                fig = px.scatter(sentiment_counts, x='Year', y='Sentiment', size='Count', color='Sentiment',
                                hover_name='Sentiment', hover_data={'Year': True, 'Sentiment': False, 'Count': True},
                                width=900, height=600)
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
                st.plotly_chart(fig)
                    
                    
                
                
                
                
                
                
                
                
                
        with tab4:
            
            import pandas as pd
            import plotly.express as px
            from nltk import pos_tag, word_tokenize

            # Estrutura de dados fictícios para ilustrar o processo
            data = df

            df = pd.DataFrame(data)

            # Obtendo as POS tags de cada ano
            pos_tags_by_year = {}
            anos_unicos = df['Year'].unique()

            for ano in anos_unicos:
                data_year = df[df['Year'] == ano]
                all_pos_tags = [tag for sublist in data_year['clean_Abstract'].apply(lambda x: pos_tag(word_tokenize(x))) for _, tag in sublist]
                pos_counts = pd.Series(all_pos_tags).value_counts().reset_index()
                pos_counts.columns = ['POS', 'Count']
                pos_tags_by_year[ano] = pos_counts

            # Criar um dicionário contendo todos os dados de POS tags
            all_pos_tags = pd.concat(pos_tags_by_year.values(), keys=pos_tags_by_year.keys())

            # Normalizar os valores da contagem para facilitar a visualização no treemap
            max_count = all_pos_tags['Count'].max()
            all_pos_tags['Normalized_Count'] = all_pos_tags['Count'] / max_count

            # Criar o treemap com Plotly Express
            fig = px.treemap(all_pos_tags.reset_index(), path=['level_0', 'POS'], values='Normalized_Count' , width=700, height=800)
            fig.update_traces(root_color="white", selector=dict(type='treemap'))
            st.plotly_chart(fig)


            if tabela:
                
                # Obtendo as POS tags de cada ano
                pos_tags_by_year = {}
                anos_unicos = df['Year'].unique()

                for ano in anos_unicos:
                    data_year = df[df['Year'] == ano]
                    all_pos_tags = [tag for sublist in data_year['clean_Abstract'].apply(lambda x: pos_tag(word_tokenize(x))) for _, tag in sublist]
                    pos_counts = pd.Series(all_pos_tags).value_counts().reset_index()
                    pos_counts.columns = ['POS', 'Count']
                    pos_tags_by_year[ano] = pos_counts

                # Criar tabelas separadas para cada ano
                for ano, pos_data in pos_tags_by_year.items():
                    st.write(f"Tabela para o ano {ano}")
                    st.write(pos_data)

                # Criar uma tabela final com a contagem total das POS tags
                all_pos_tags = pd.concat(pos_tags_by_year.values())
                total_pos_counts = all_pos_tags.groupby('POS')['Count'].sum().reset_index()
                st.write("Total das POS tags")
                st.write(total_pos_counts)
                
                
            if grafico:
                
                # Obtendo as POS tags de cada ano
                pos_tags_by_year = {}
                anos_unicos = df['Year'].unique()

                for ano in anos_unicos:
                    data_year = df[df['Year'] == ano]
                    all_pos_tags = [tag for sublist in data_year['clean_Abstract'].apply(lambda x: pos_tag(word_tokenize(x))) for _, tag in sublist]
                    pos_counts = pd.Series(all_pos_tags).value_counts().reset_index()
                    pos_counts.columns = ['POS', 'Count']
                    pos_tags_by_year[ano] = pos_counts

                # Criar um DataFrame para o gráfico de bolhas
                df_bolhas = pd.concat(pos_tags_by_year.values())

                # Criar o gráfico de bolhas com Plotly Express
                fig = px.scatter(df_bolhas, x='POS', y='Count', size='Count', color='POS',
                                hover_name='POS', hover_data={'POS': False, 'Count': True},
                                width=900, height=600)

                # Atualizando a aparência do gráfico de bolhas
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

                # Exibindo o gráfico de bolhas no Streamlit
                st.plotly_chart(fig)
                
    
    
    
    
    
    
    
    # Mutações do título
    
    # Remove duplicate rows
    df = df.drop_duplicates(subset=['clean_Title'])
    
    # Remover linhas onde a coluna 'Title' contém 'Compilation of references'
    df = df[~df['clean_Title'].str.contains('Compil refer', case=False)]
    
    if modificador == "Mutações do Título":
        
        tab1, tab2 = st.tabs(["N-Grams mais Frequentes", "Análise de Sentimento"])

        with tab1:
            
            # Obtendo as 10 palavras mais frequentes por ano na coluna 'clean_Title'
            top_words_by_year = df.groupby('Year')['clean_Title'].apply(lambda x: pd.Series(' '.join(x).split()).value_counts().head(10))

            # Redefinindo o índice e renomeando colunas para facilitar a visualização no treemap
            top_words_by_year = top_words_by_year.reset_index()
            top_words_by_year.columns = ['Year', 'Word', 'Count']

            # Criando o treemap com Plotly Express
            fig = px.treemap(top_words_by_year, path=['Year', 'Word'], values='Count', width=700, height=800)

            # Exibindo o treemap no Streamlit
            st.plotly_chart(fig)
                    
                
                    
            if tabela:
                
                # Exibindo uma tabela para as 10 palavras mais frequentes por ano
                st.subheader("Top 10 Palavras Mais Frequentes por Ano")
                for year in df['Year'].unique():
                    st.write(f"Ano: {year}")
                    top_words_year = top_words_by_year[top_words_by_year['Year'] == year]
                    st.write(top_words_year)

                # Calculando e exibindo o total das 10 palavras mais frequentes
                st.subheader("Total de Palavras Mais Frequentes")
                total_top_words = top_words_by_year.groupby('Word')['Count'].sum().nlargest(10).reset_index()
                st.write(total_top_words)
                
                
            
            if grafico:
                
                # Criando o gráfico de bolhas para as 10 palavras mais frequentes por ano
                fig = px.scatter(top_words_by_year, x='Year', y='Word', size='Count', color='Word',
                                hover_name='Word', hover_data={'Year': True, 'Word': False, 'Count': True}, width=900, height=600)
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

                # Exibindo o gráfico de bolhas no Streamlit
                st.plotly_chart(fig)
                
                
        
        
        
        with tab2:
            
            from textblob import TextBlob
            
            # Função para análise de sentimento
            def analyze_sentiment(title):
                analysis = TextBlob(title)
                polarity = analysis.sentiment.polarity

                if polarity > 0:
                    return 'Positivo'
                elif polarity == 0:
                    return 'Neutro'
                else:
                    return 'Negativo'

            # Aplicando a análise de sentimento ao título e criando uma nova coluna com os resultados
            df['Sentiment'] = df['clean_Title'].apply(analyze_sentiment)

            # Contando os resultados da análise de sentimento por ano
            sentiment_counts = df.groupby(['Year', 'Sentiment']).size().reset_index(name='Count')

            # Criando o treemap com Plotly Express
            fig = px.treemap(sentiment_counts, path=['Year', 'Sentiment'], values='Count', width=700, height=800)

            # Exibindo o treemap no Streamlit
            st.plotly_chart(fig)
                    
            
            
            if tabela:
                
                # Criando tabelas para os sentimentos por ano e o total
                for year in df['Year'].unique():
                    st.write(f"Ano: {year}")
                    sentiment_year = sentiment_counts[sentiment_counts['Year'] == year]
                    st.write(sentiment_year)

                st.subheader("Total de Sentimentos")
                total_sentiments = sentiment_counts.groupby('Sentiment')['Count'].sum().reset_index()
                st.write(total_sentiments)
        



            if grafico:
                
                # Criando o gráfico de bolhas com Plotly Express
                fig = px.scatter(sentiment_counts, x='Year', y='Sentiment', size='Count', color='Sentiment',
                                hover_name='Sentiment', hover_data={'Year': True, 'Sentiment': False, 'Count': True},
                                width=900, height=600)

                # Atualizando a aparência do gráfico de bolhas
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

                # Exibindo o gráfico de bolhas no Streamlit
                st.plotly_chart(fig)
        
        
        
    
    
    
    
    # Mutações do Author Keywords
    
    # Remove duplicate rows
    df = df.drop_duplicates(subset=['clean_Author_Keywords'])
        
    # Substituir 'nan' por NaN (valores nulos)
    df['clean_Author_Keywords'].replace('nan', np.nan, inplace=True)

    # Remover as linhas em que a coluna 'clean_Author_Keywords' contém valores NaN
    df = df.dropna(subset=['clean_Author_Keywords'])
    
    
    if modificador == "Mutações das Palavras-Chave do Autor":
        
        tab1, tab2 = st.tabs(["N-Grams mais Frequentes", "Análise de Sentimento"])

        with tab1:
            
            # Obtendo as 10 palavras mais frequentes por ano na coluna 'clean_Title'
            top_words_by_year = df.groupby('Year')['clean_Author_Keywords'].apply(lambda x: pd.Series(' '.join(x).split()).value_counts().head(10))

            # Redefinindo o índice e renomeando colunas para facilitar a visualização no treemap
            top_words_by_year = top_words_by_year.reset_index()
            top_words_by_year.columns = ['Year', 'Word', 'Count']

            # Criando o treemap com Plotly Express
            fig = px.treemap(top_words_by_year, path=['Year', 'Word'], values='Count', width=700, height=800)

            # Exibindo o treemap no Streamlit
            st.plotly_chart(fig)
                    
                
                    
            if tabela:
                
                # Exibindo uma tabela para as 10 palavras mais frequentes por ano
                st.subheader("Top 10 Palavras Mais Frequentes por Ano")
                for year in df['Year'].unique():
                    st.write(f"Ano: {year}")
                    top_words_year = top_words_by_year[top_words_by_year['Year'] == year]
                    st.write(top_words_year)

                # Calculando e exibindo o total das 10 palavras mais frequentes
                st.subheader("Total de Palavras Mais Frequentes")
                total_top_words = top_words_by_year.groupby('Word')['Count'].sum().nlargest(10).reset_index()
                st.write(total_top_words)
                
                
            
            if grafico:
                
                # Criando o gráfico de bolhas para as 10 palavras mais frequentes por ano
                fig = px.scatter(top_words_by_year, x='Year', y='Word', size='Count', color='Word',
                                hover_name='Word', hover_data={'Year': True, 'Word': False, 'Count': True}, width=900, height=600)
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

                # Exibindo o gráfico de bolhas no Streamlit
                st.plotly_chart(fig)
                
                
        
        
        
        with tab2:
            
            from textblob import TextBlob
            
            # Função para análise de sentimento
            def analyze_sentiment(title):
                analysis = TextBlob(title)
                polarity = analysis.sentiment.polarity

                if polarity > 0:
                    return 'Positivo'
                elif polarity == 0:
                    return 'Neutro'
                else:
                    return 'Negativo'

            # Aplicando a análise de sentimento ao título e criando uma nova coluna com os resultados
            df['Sentiment'] = df['clean_Author_Keywords'].apply(analyze_sentiment)

            # Contando os resultados da análise de sentimento por ano
            sentiment_counts = df.groupby(['Year', 'Sentiment']).size().reset_index(name='Count')

            # Criando o treemap com Plotly Express
            fig = px.treemap(sentiment_counts, path=['Year', 'Sentiment'], values='Count', width=700, height=800)

            # Exibindo o treemap no Streamlit
            st.plotly_chart(fig)
                    
            
            
            if tabela:
                
                # Criando tabelas para os sentimentos por ano e o total
                for year in df['Year'].unique():
                    st.write(f"Ano: {year}")
                    sentiment_year = sentiment_counts[sentiment_counts['Year'] == year]
                    st.write(sentiment_year)

                st.subheader("Total de Sentimentos")
                total_sentiments = sentiment_counts.groupby('Sentiment')['Count'].sum().reset_index()
                st.write(total_sentiments)
        



            if grafico:
                
                # Criando o gráfico de bolhas com Plotly Express
                fig = px.scatter(sentiment_counts, x='Year', y='Sentiment', size='Count', color='Sentiment',
                                hover_name='Sentiment', hover_data={'Year': True, 'Sentiment': False, 'Count': True},
                                width=900, height=600)

                # Atualizando a aparência do gráfico de bolhas
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

                # Exibindo o gráfico de bolhas no Streamlit
                st.plotly_chart(fig)  
        
        
        
        
        
        
        
        
        
        


# botões em baixo
with st.container():
    import webbrowser

    # Fazer por colunas
    st.write("---")
    st.subheader("Consulte os dados da pesquisa")

    col1, col2, col3 = st.columns(3)


    with col1:  # Botão do consulte dados da pesquisa
        btn = st.button("Ver dados através da SCOPUS")
        if btn:
            webbrowser.open_new_tab("https://www.scopus.com/")




    with col2:  # Botão para download do excel
        
        data = pd.DataFrame({
        'scopus.csv'
        })

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(data)

        
        st.download_button(
            label='Descarregar ficheiro CSV',
            data=csv,
            file_name='scopus1.csv',
            mime='text/csv',
            )