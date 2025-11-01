import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

# Configurar matplotlib para compatibilidade
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Predição de Criptomoedas", layout="centered")

st.title("🔮 Predição de Preços de Criptomoedas")
st.markdown("Este painel utiliza modelos preditivos de séries temporais para prever os preços das criptomoedas para os próximos 7 dias. Fonte de dados: API Binance")

# Parâmetros GARCH(1,1) estimados
garch_params = {
    "BTCUSDT": {"omega": 0.116832, "alpha": 0.043909, "beta": 0.932001, "mu": 0.141378},
    "ETHUSDT": {"omega": 7.859079, "alpha": 0.061944, "beta": 0.444161, "mu": 0.219644},
    "SOLUSDT": {"omega": 3.990503, "alpha": 0.078300, "beta": 0.730698, "mu": 0.166247}
}

# Seleção de moeda
coin = st.selectbox("Escolha a moeda para prever:", ["bitcoin", "ethereum", "solana"])
symbol_map = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "solana": "SOLUSDT"}
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

    # Calcular retornos
    df['retornos'] = np.log(df['preco'] / df['preco'].shift(1))
    df = df.dropna()

    # Simulação GARCH
    def simulate_garch(params, last_price, last_volatility, n_simulations=100, days=7):
        all_simulations = []
        for _ in range(n_simulations):
            simulated_prices = [last_price]
            current_volatility = last_volatility
            for _ in range(days):
                innovation = np.random.normal(0, np.sqrt(current_volatility))
                next_return = params["mu"] + innovation
                current_volatility = params["omega"] + params["alpha"] * innovation**2 + params["beta"] * current_volatility
                next_price = simulated_prices[-1] * np.exp(next_return)
                simulated_prices.append(next_price)
            all_simulations.append(simulated_prices[1:])
        return np.array(all_simulations)

    # Obter últimos valores
    last_price = df['preco'].iloc[-1]
    last_date = df['data'].iloc[-1]
    last_returns = df['retornos'].iloc[-30:]
    last_volatility = np.var(last_returns)

    # Executar simulação
    params = garch_params[symbol]
    simulations = simulate_garch(params, last_price, last_volatility)
    mean_predictions = np.mean(simulations, axis=0)

    # Preparar datas
    historical_dates = df['data']  # Todos os dados históricos
    historical_prices = df['preco']  # Todos os preços históricos
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

    # Gráfico com histórico completo
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical_dates, historical_prices, label='Histórico Completo', color='blue', linewidth=1.5)
    ax.plot(future_dates, mean_predictions, label='Previsão (Próximos 7 dias)', color='red', linewidth=2, marker='o')
    ax.set_title(f"{coin.capitalize()} ({symbol}) - Previsão de Preços", fontsize=14, fontweight='bold')
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # ----------------------------
    # KPIs Atualizados
    # ----------------------------
    st.subheader("📊 Previsões para os Próximos Dias")
    
    # Função para formatar preços com separadores
    def format_price(price):
        return f"${price:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"Preço Atual ({last_date.strftime('%d/%m/%Y')})", 
            format_price(last_price)
        )
    with col2:
        change_1d = ((mean_predictions[0] - last_price) / last_price) * 100
        st.metric(
            f"Previsão 1º Dia ({future_dates[0].strftime('%d/%m/%Y')})", 
            format_price(mean_predictions[0]), 
            f"{change_1d:+.2f}%"
        )
    with col3:
        change_2d = ((mean_predictions[1] - last_price) / last_price) * 100
        st.metric(
            f"Previsão 2º Dia ({future_dates[1].strftime('%d/%m/%Y')})", 
            format_price(mean_predictions[1]), 
            f"{change_2d:+.2f}%"
        )

    # Segunda linha de KPIs
    col4, col5, col6 = st.columns(3)
    with col4:
        change_3d = ((mean_predictions[2] - last_price) / last_price) * 100
        st.metric(
            f"Previsão 3º Dia ({future_dates[2].strftime('%d/%m/%Y')})", 
            format_price(mean_predictions[2]), 
            f"{change_3d:+.2f}%"
        )
    with col5:
        change_7d = ((mean_predictions[-1] - last_price) / last_price) * 100
        st.metric(
            f"Previsão 7º Dia ({future_dates[-1].strftime('%d/%m/%Y')})", 
            format_price(mean_predictions[-1]), 
            f"{change_7d:+.2f}%"
        )
    with col6:
        # Variação total no período
        total_change = ((mean_predictions[-1] - last_price) / last_price) * 100
        st.metric(
            "Variação no Período", 
            f"{total_change:+.2f}%",
            f"De {format_price(last_price)} para {format_price(mean_predictions[-1])}"
        )

    # ----------------------------
    # Tabela de Médias Móveis e Sugestões (APENAS DIAS FUTUROS)
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
        historical_prices, 
        mean_predictions, 
        historical_dates, 
        future_dates
    )

    # ----------------------------
    # Tabela de Sugestões Formatada
    # ----------------------------
    st.subheader(f"📊 Tabela de Sugestão para {symbol}")

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
    else:
        st.info("Não há dados de previsão disponíveis para exibir a tabela de sugestões.")

except Exception as e:
    st.error(f"Erro ao processar os dados: {str(e)}")
