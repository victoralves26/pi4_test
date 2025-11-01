import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Predição de Criptomoedas - GARCH", layout="centered")

st.title("🔮 Predição de Preços de Criptomoedas - Modelo GARCH(1,1)")
st.markdown("Este painel utiliza modelos GARCH(1,1) para prever os preços das criptomoedas nos próximos 7 dias, com base em dados salvos localmente.")

# ----------------------------
# Parâmetros GARCH(1,1) estimados
# ----------------------------
garch_params = {
    "BTCUSDT": {
        "omega": 0.116832,
        "alpha": 0.043909,
        "beta": 0.932001,
        "mu": 0.141378
    },
    "ETHUSDT": {
        "omega": 7.859079,
        "alpha": 0.061944,
        "beta": 0.444161,
        "mu": 0.219644
    },
    "SOLUSDT": {
        "omega": 3.990503,
        "alpha": 0.078300,
        "beta": 0.730698,
        "mu": 0.166247
    }
}

# ----------------------------
# Seleção de moeda
# ----------------------------
coin = st.selectbox("Escolha a moeda para prever:", ["bitcoin", "ethereum", "solana"])
symbol_map = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT"
}
symbol = symbol_map[coin]
csv_file = "dados_binance.csv"

# ----------------------------
# Verificação do arquivo
# ----------------------------
if not os.path.exists(csv_file):
    st.error(f"Arquivo de dados '{csv_file}' não encontrado. Execute o script 'dados_binance.py' para gerar os dados.")
    st.stop()

# ----------------------------
# Carregamento e preparação dos dados
# ----------------------------
df = pd.read_csv(csv_file)
df["data"] = pd.to_datetime(df["data"])
df = df[df["symbol"] == symbol].sort_values("data")

if df.empty:
    st.error(f"Não há dados disponíveis para {coin}.")
    st.stop()

# Calcular retornos logarítmicos (necessário para GARCH)
df['retornos'] = np.log(df['preco'] / df['preco'].shift(1))
df = df.dropna()

# ----------------------------
# Simulação GARCH(1,1) - Previsão de 7 dias
# ----------------------------
def simulate_garch(params, last_price, last_volatility, n_simulations=100, days=7):
    """
    Simula preços futuros usando modelo GARCH(1,1)
    """
    all_simulations = []
    
    for _ in range(n_simulations):
        simulated_prices = [last_price]
        current_volatility = last_volatility
        
        for _ in range(days):
            # Simular inovação (erro)
            innovation = np.random.normal(0, np.sqrt(current_volatility))
            
            # Calcular próximo retorno usando mu (média constante)
            next_return = params["mu"] + innovation
            
            # Atualizar volatilidade (equação GARCH)
            current_volatility = (params["omega"] + 
                                params["alpha"] * innovation**2 + 
                                params["beta"] * current_volatility)
            
            # Calcular próximo preço
            next_price = simulated_prices[-1] * np.exp(next_return)
            simulated_prices.append(next_price)
        
        all_simulations.append(simulated_prices[1:])  # Remover o preço inicial
    
    return np.array(all_simulations)

# Obter últimos valores para inicializar a simulação
last_price = df['preco'].iloc[-1]
last_returns = df['retornos'].iloc[-30:]  # Usar últimos 30 dias para volatilidade inicial
last_volatility = np.var(last_returns)

# Executar simulação
params = garch_params[symbol]
simulations = simulate_garch(params, last_price, last_volatility)

# Calcular estatísticas das simulações
mean_predictions = np.mean(simulations, axis=0)
std_predictions = np.std(simulations, axis=0)
confidence_upper = mean_predictions + 1.96 * std_predictions
confidence_lower = np.maximum(mean_predictions - 1.96 * std_predictions, 0)  # Preços não podem ser negativos

# ----------------------------
# Preparar dados para o gráfico
# ----------------------------
# Últimos 30 dias do histórico
historical_dates = df['data'].iloc[-30:]
historical_prices = df['preco'].iloc[-30:]

# Datas futuras (próximos 7 dias)
last_date = df['data'].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

# ----------------------------
# Tabela de Médias Móveis e Sugestões
# ----------------------------
def calculate_moving_averages_and_suggestions(historical_prices, future_prices, historical_dates, future_dates):
    """
    Calcula médias móveis e gera sugestões de trading
    """
    # Combinar dados históricos e futuros
    all_dates = list(historical_dates) + list(future_dates)
    all_prices = list(historical_prices) + list(future_prices)
    all_types = ['Observado'] * len(historical_prices) + ['Predito'] * len(future_prices)
    
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
            # Quando as médias estão próximas ou sem tendência clara
            if abs(mm3 - mm7) < (mm3 * 0.01):  # Diferença menor que 1%
                if current_price > mm15:
                    suggestions.append('Manter/Otimista')
                else:
                    suggestions.append('Manter/Cauteloso')
            else:
                suggestions.append('Manter/Indefinido')
    
    result_df['Sugestão'] = suggestions
    
    return result_df

# Calcular tabela de médias móveis
suggestion_table = calculate_moving_averages_and_suggestions(
    historical_prices.iloc[-10:],  # Últimos 10 dias históricos
    mean_predictions, 
    historical_dates.iloc[-10:], 
    future_dates
)

# ----------------------------
# Gráfico
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Plotar histórico
ax.plot(historical_dates, historical_prices, label='Histórico (30 dias)', color='blue', linewidth=2)

# Plotar previsão média
ax.plot(future_dates, mean_predictions, label='Previsão Média (GARCH)', color='red', linewidth=2, marker='o')

# Adicionar intervalo de confiança
ax.fill_between(future_dates, confidence_lower, confidence_upper, alpha=0.2, color='red', 
                label='Intervalo de Confiança (95%)')

ax.set_title(f"{coin.capitalize()} ({symbol}) - Previsão de Preços (Modelo GARCH(1,1))", fontsize=14, fontweight='bold')
ax.set_xlabel("Data")
ax.set_ylabel("Preço (USD)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# ----------------------------
# Exibir estatísticas
# ----------------------------
st.subheader("📊 Estatísticas da Previsão GARCH(1,1)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Preço Atual", f"${last_price:.2f}")
with col2:
    change_1d = ((mean_predictions[0] - last_price) / last_price) * 100
    st.metric("Previsão 1º Dia", f"${mean_predictions[0]:.2f}", f"{change_1d:+.2f}%")
with col3:
    change_7d = ((mean_predictions[-1] - last_price) / last_price) * 100
    st.metric("Previsão 7º Dia", f"${mean_predictions[-1]:.2f}", f"{change_7d:+.2f}%")

# ----------------------------
# Tabela de Sugestões com Médias Móveis
# ----------------------------
st.subheader(f"📊 Tabela de Sugestão (Média GARCH) para {symbol}")

# Formatar a tabela para exibição
display_table = suggestion_table.copy()
display_table['Data'] = display_table['Data'].dt.strftime('%d/%m/%Y')
display_table['Preço'] = display_table.apply(
    lambda x: f"${x['Preço']:,.2f} ({x['Tipo']})", axis=1
)
display_table['MM_3_dias'] = display_table['MM_3_dias'].apply(lambda x: f"${x:,.2f}")
display_table['MM_7_dias'] = display_table['MM_7_dias'].apply(lambda x: f"${x:,.2f}")
display_table['MM_15_dias'] = display_table['MM_15_dias'].apply(lambda x: f"${x:,.2f}")

# Renomear colunas para exibição
display_table = display_table.rename(columns={
    'MM_3_dias': 'MM 3 dias',
    'MM_7_dias': 'MM 7 dias', 
    'MM_15_dias': 'MM 15 dias'
})

# Exibir tabela
st.dataframe(display_table[['Data', 'Preço', 'MM 3 dias', 'MM 7 dias', 'MM 15 dias', 'Sugestão']], 
             hide_index=True)

# ----------------------------
# Legenda das Sugestões
# ----------------------------
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

# ----------------------------
# Exibir parâmetros do modelo
# ----------------------------
st.subheader("⚙️ Parâmetros do Modelo GARCH(1,1) Utilizado")
params_df = pd.DataFrame({
    'Parâmetro': ['Omega (ω)', 'Alpha (α₁)', 'Beta (β₁)', 'Mu (μ)'],
    'Valor': [params['omega'], params['alpha'], params['beta'], params['mu']],
    'Descrição': [
        'Termo constante da volatilidade',
        'Efeito dos choques passados (inovação)',
        'Persistência da volatilidade', 
        'Retorno médio constante'
    ]
})
st.dataframe(params_df, hide_index=True)

st.info("""
**Interpretação dos Parâmetros GARCH(1,1):**
- **Alpha (α₁)**: Mede o impacto de choques recentes na volatilidade. Valores mais altos indicam que notícias recentes têm maior impacto.
- **Beta (β₁)**: Mede a persistência da volatilidade. Valores próximos de 1 indicam que a volatilidade é altamente persistente.
- **Soma (α + β)**: Indica a persistência total da volatilidade. Valores próximos de 1 sugerem que choques na volatilidade são longos.
""")