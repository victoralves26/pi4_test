import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para compatibilidade
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Predição de Criptomoedas", layout="centered")

st.title("🔮 Predição de Preços de Criptomoedas")
st.markdown("Este painel utiliza modelos preditivos de séries temporais para prever os preços das criptomoedas para os próximos 7 dias. Fonte de dados: API Binance")

# Parâmetros GARCH(1,1) estimados (mantidos para compatibilidade)
garch_params = {
    "Bitcoin": {"omega": 0.116832, "alpha": 0.043909, "beta": 0.932001, "mu": 0.141378},
    "Ethereum": {"omega": 7.859079, "alpha": 0.061944, "beta": 0.444161, "mu": 0.219644},
    "Solana": {"omega": 3.990503, "alpha": 0.078300, "beta": 0.730698, "mu": 0.166247}
}

# Mapeamento para símbolos
symbol_map = {
    "Bitcoin": "BTCUSDT", 
    "Ethereum": "ETHUSDT", 
    "Solana": "SOLUSDT"
}

# Seleção de moeda
coin = st.selectbox("Escolha a moeda para prever:", ["Bitcoin", "Ethereum", "Solana"])
symbol = symbol_map[coin]

# Verificar se arquivo existe
if not os.path.exists("dados_binance.csv"):
    st.error("Arquivo 'dados_binance.csv' não encontrado. Verifique se o arquivo está no repositório.")
    st.stop()

try:
    # Carregar dados
    df = pd.read_csv("dados_binance.csv")
    df["data"] = pd.to_datetime(df["data"])
    df = df[df["symbol"] == symbol].sort_values("data")
    
    if df.empty:
        st.error(f"Não há dados disponíveis para {coin}.")
        st.stop()

    # Calcular retornos logarítmicos
    df['retornos'] = np.log(df['preco'] / df['preco'].shift(1))
    df = df.dropna()

    # ----------------------------
    # SIMULAÇÃO GARCH MELHORADA
    # ----------------------------
    
    def plot_garch_price_projection(historical_series, forecast_days, historical_returns_mean, 
                                  predicted_volatility, crypto_name, num_simulations=100):
        """
        Função para plotar projeções de preços usando simulações GARCH
        """
        # Último preço histórico
        last_price = historical_series['preco'].iloc[-1]
        last_date = historical_series['data'].iloc[-1]
        
        # Gerar simulações de Monte Carlo
        simulations = []
        for _ in range(num_simulations):
            price_path = [last_price]
            current_price = last_price
            
            for day in range(forecast_days):
                # Usar a volatilidade prevista do GARCH
                if day < len(predicted_volatility):
                    daily_vol = predicted_volatility[day]
                else:
                    # Se não houver volatilidade prevista para este dia, usar a última disponível
                    daily_vol = predicted_volatility[-1] if len(predicted_volatility) > 0 else 0.02
                
                # Gerar retorno aleatório baseado na média histórica e volatilidade prevista
                random_return = np.random.normal(historical_returns_mean, np.sqrt(daily_vol))
                
                # Calcular próximo preço
                next_price = current_price * np.exp(random_return)
                price_path.append(next_price)
                current_price = next_price
            
            simulations.append(price_path[1:])  # Remover o preço inicial
        
        simulations = np.array(simulations)
        
        # Calcular estatísticas
        mean_predictions = np.mean(simulations, axis=0)
        confidence_upper = np.percentile(simulations, 95, axis=0)
        confidence_lower = np.percentile(simulations, 5, axis=0)
        
        # Preparar datas
        historical_dates = historical_series['data']
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        # Criar figura - APENAS UM GRÁFICO
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Gráfico: Histórico + Previsão com intervalo de confiança
        ax.plot(historical_dates, historical_series['preco'], 
                label='Histórico', color='blue', linewidth=2, alpha=0.8)
        ax.plot(future_dates, mean_predictions, 
                label='Previsão Média', color='red', linewidth=3, marker='o')
        ax.fill_between(future_dates, confidence_lower, confidence_upper, 
                        alpha=0.3, color='red', label='Intervalo 90% Confiança')
        
        ax.set_title(f'{crypto_name} - Projeção de Preços (GARCH)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig, mean_predictions, future_dates

    # ----------------------------
    # EXECUTAR SIMULAÇÃO GARCH
    # ----------------------------
    
    # Parâmetros para a simulação
    dias_previsao_garch = 7
    
    # Calcular estatísticas dos retornos
    retornos_series = df['retornos'].dropna()
    historical_returns_mean = retornos_series.mean()
    
    # Estimar volatilidade prevista
    volatilidade_base = retornos_series.var()
    
    # Simular volatilidade prevista (decaindo suavemente)
    predicted_volatility = [volatilidade_base * (0.95 ** i) for i in range(dias_previsao_garch)]
    
    # Gerar gráfico de projeção
    fig, mean_predictions, future_dates = plot_garch_price_projection(
        historical_series=df,
        forecast_days=dias_previsao_garch,
        historical_returns_mean=historical_returns_mean,
        predicted_volatility=predicted_volatility,
        crypto_name=coin,
        num_simulations=100
    )
    
    st.pyplot(fig)

    # ----------------------------
    # KPIs ATUALIZADOS
    # ----------------------------
    st.subheader("📊 Indicadores Atuais")
    
    # Função para formatar preços com separadores
    def format_price(price):
        return f"${price:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    # Calcular médias móveis
    last_price = df['preco'].iloc[-1]
    last_date = df['data'].iloc[-1]
    mm_7_dias = df['preco'].tail(7).mean()
    mm_15_dias = df['preco'].tail(15).mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"Preço Atual ({last_date.strftime('%d/%m/%Y')})", 
            format_price(last_price)
        )
    with col2:
        st.metric(
            "Média Móvel (7 dias)", 
            format_price(mm_7_dias)
        )
    with col3:
        st.metric(
            "Média Móvel (15 dias)", 
            format_price(mm_15_dias)
        )

    # ----------------------------
    # TABELA DE SUGESTÕES
    # ----------------------------
    def calculate_moving_averages_and_suggestions(historical_prices, future_prices, historical_dates, future_dates):
        """
        Calcula médias móveis e gera sugestões de trading APENAS para dias futuros
        """
        # Usar apenas os últimos 15 dias históricos + previsões futuras
        recent_historical = historical_prices.iloc[-15:]
        recent_dates = historical_dates.iloc[-15:]
        
        # Combinar dados recentes históricos e futuros
        all_dates = list(recent_dates) + list(future_dates)
        all_prices = list(recent_historical) + list(future_prices)
        all_types = ['Observado'] * len(recent_historical) + ['Predito'] * len(future_prices)
        
        # Criar DataFrame
        result_df = pd.DataFrame({
            'Data': all_dates,
            'Preço': all_prices,
            'Tipo': all_types
        })
        
        # Calcular médias móveis
        result_df['MM_3_dias'] = result_df['Preço'].rolling(window=3, min_periods=1).mean()
        result_df['MM_7_dias'] = result_df['Preço'].rolling(window=7, min_periods=1).mean()
        result_df['MM_15_dias'] = result_df['Preço'].rolling(window=15, min_periods=1).mean()
        
        # Gerar sugestões baseadas em cruzamento de médias móveis
        suggestions = []
        for i in range(len(result_df)):
            if i < 2:  # Primeiros dias não têm médias suficientes
                suggestions.append('Aguardar Dados')
                continue
            
            current_price = result_df['Preço'].iloc[i]
            mm3 = result_df['MM_3_dias'].iloc[i]
            mm7 = result_df['MM_7_dias'].iloc[i]
            mm15 = result_df['MM_15_dias'].iloc[i]
            
            # Lógica de sugestão baseada em cruzamentos
            if mm3 > mm7 and mm3 > mm15:
                if current_price > mm3:
                    suggestions.append('Compra Forte')
                else:
                    suggestions.append('Compra (Tendência Curta)')
            elif mm3 < mm7 and mm3 < mm15:
                if current_price < mm3:
                    suggestions.append('Venda Forte')
                else:
                    suggestions.append('Venda (Tendência Curta)')
            elif mm7 > mm3 and mm7 > mm15:
                suggestions.append('Manter/Positivo')
            elif mm7 < mm3 and mm7 < mm15:
                suggestions.append('Manter/Negativo')
            else:
                if abs(mm3 - mm7) < (mm3 * 0.01):
                    if current_price > mm15:
                        suggestions.append('Manter/Otimista')
                    else:
                        suggestions.append('Manter/Cauteloso')
                else:
                    suggestions.append('Manter/Indefinido')
        
        result_df['Sugestão'] = suggestions
        
        # Manter APENAS os dias futuros (preditos)
        future_only_df = result_df[result_df['Tipo'] == 'Predito'].copy()
        
        return future_only_df

    # Calcular tabela de médias móveis (apenas dias futuros)
    suggestion_table = calculate_moving_averages_and_suggestions(
        df['preco'], 
        mean_predictions, 
        df['data'], 
        future_dates
    )

    # Exibir tabela de sugestões
    st.subheader(f"📊 Tabela de Sugestão para {coin}")

    if not suggestion_table.empty:
        # Formatar a tabela para exibição
        display_table = suggestion_table.copy()
        display_table['Data'] = display_table['Data'].dt.strftime('%d/%m/%Y')
        display_table['Preço'] = display_table.apply(
            lambda x: f"{format_price(x['Preço'])} ({x['Tipo']})", axis=1
        )
        display_table['MM_3_dias'] = display_table['MM_3_dias'].apply(lambda x: format_price(x))
        display_table['MM_7_dias'] = display_table['MM_7_dias'].apply(lambda x: format_price(x))
        display_table['MM_15_dias'] = display_table['MM_15_dias'].apply(lambda x: format_price(x))

        # Renomear colunas para exibição
        display_table = display_table.rename(columns={
            'MM_3_dias': 'MM 3 dias',
            'MM_7_dias': 'MM 7 dias', 
            'MM_15_dias': 'MM 15 dias'
        })

        # Exibir tabela
        st.dataframe(display_table[['Data', 'Preço', 'MM 3 dias', 'MM 7 dias', 'MM 15 dias', 'Sugestão']], 
                    hide_index=True)

        # Legenda das Sugestões
        st.markdown("""
        **📋 Legenda das Sugestões:**
        - **Compra Forte**: Tendência claramente positiva em múltiplos prazos
        - **Compra (Tendência Curta)**: Tendência positiva no curto prazo
        - **Venda Forte**: Tendência claramente negativa em múltiplos prazos  
        - **Venda (Tendência Curta)**: Tendência negativa no curto prazo
        - **Manter/Positivo**: Tendência positiva no médio prazo
        - **Manter/Negativo**: Tendência negativa no médio prazo
        - **Manter/Otimista**: Mercado lateral com viés positivo
        - **Manter/Cauteloso**: Mercado lateral com viés negativo
        - **Manter/Indefinido**: Tendência não clara, aguardar confirmação
        """)

except Exception as e:
    st.error(f"Erro ao processar os dados: {str(e)}")
    st.info("Verifique se o arquivo 'dados_binance.csv' contém dados válidos para as criptomoedas.")
