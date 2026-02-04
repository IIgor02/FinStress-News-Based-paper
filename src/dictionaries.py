"""
Portuguese Financial Dictionaries for FSI
==========================================
Based on Baker et al. (2019) methodology adapted for Brazilian Portuguese.

Three categories for three-way co-occurrence:
1. Financial Terms - Identify financial context
2. Stress Terms - Identify stress/uncertainty context
3. Negative Terms - Identify negative sentiment
"""

# =============================================================================
# CATEGORY 1: FINANCIAL TERMS
# =============================================================================
# Terms that indicate financial/economic context
# A post must contain at least one of these to be considered financial

FINANCIAL_TERMS = [
    # Markets & Exchanges
    'mercado', 'mercados', 'bolsa', 'bolsas', 'ibovespa', 'b3', 'bovespa',
    'pregão', 'nasdaq', 'dow jones', 'nyse', 's&p',

    # Financial instruments
    'ações', 'ação', 'ativo', 'ativos', 'papel', 'papéis',
    'título', 'títulos', 'bond', 'bonds', 'debênture', 'debêntures',
    'derivativo', 'derivativos', 'opção', 'opções', 'futuro', 'futuros',
    'fundo', 'fundos', 'etf', 'cota', 'cotas',

    # Banking & Credit
    'banco', 'bancos', 'bancário', 'bancária', 'bancários',
    'crédito', 'empréstimo', 'empréstimos', 'financiamento',
    'juros', 'selic', 'taxa', 'spread', 'cdi',

    # Economy & Finance
    'financeiro', 'financeira', 'financeiros', 'economia', 'econômico',
    'econômica', 'pib', 'inflação', 'ipca', 'igpm',

    # Investment
    'investimento', 'investimentos', 'investidor', 'investidores',
    'carteira', 'portfólio', 'alocação', 'diversificação',
    'rentabilidade', 'retorno', 'rendimento', 'dividendo', 'dividendos',

    # Currency & Exchange
    'dólar', 'câmbio', 'moeda', 'real', 'forex', 'brl', 'usd',
    'desvalorização cambial', 'valorização cambial',

    # Institutions
    'bacen', 'banco central', 'cvm', 'tesouro', 'copom',
    'bndes', 'caixa', 'itaú', 'bradesco', 'santander',

    # Corporate
    'empresa', 'empresas', 'companhia', 'companhias', 'corporativo',
    'balanço', 'resultado', 'lucro', 'receita', 'faturamento',
    'ebitda', 'margem', 'endividamento', 'alavancagem',
]

# =============================================================================
# CATEGORY 2: STRESS TERMS
# =============================================================================
# Terms that indicate stress, uncertainty, or volatility
# These capture the "stress" component of financial stress

STRESS_TERMS = [
    # Crisis & Instability
    'crise', 'crises', 'risco', 'riscos', 'volatilidade',
    'instabilidade', 'incerteza', 'incertezas', 'turbulência',

    # Panic & Fear
    'pânico', 'medo', 'temor', 'apreensão', 'preocupação',
    'nervosismo', 'ansiedade', 'histeria', 'desespero',

    # Collapse & Disaster
    'colapso', 'desastre', 'catástrofe', 'caos', 'caótico',
    'destruição', 'derrocada', 'debacle', 'crash',

    # Uncertainty & Doubt
    'dúvida', 'dúvidas', 'indefinição', 'imprevisibilidade',
    'insegurança', 'instável', 'incerto', 'imprevisível',

    # Stress indicators
    'tensão', 'pressão', 'estresse', 'alerta', 'alarme',
    'emergência', 'urgência', 'grave', 'gravidade', 'crítico',

    # Market stress terms
    'circuit breaker', 'sell-off', 'selloff', 'bear market',
    'correção', 'desvalorização', 'derretimento', 'sangria',

    # Economic stress
    'recessão', 'depressão', 'estagflação', 'contração',
    'desaceleração', 'enfraquecimento', 'deterioração',

    # Systemic risk
    'sistêmico', 'sistêmica', 'contágio', 'cascata',
    'dominó', 'spillover', 'exposição',
]

# =============================================================================
# CATEGORY 3: NEGATIVE TERMS
# =============================================================================
# Negative sentiment words adapted from financial context
# Expanded list (~100 terms) for better coverage

NEGATIVE_TERMS = [
    # Losses & Declines
    'queda', 'quedas', 'perda', 'perdas', 'prejuízo', 'prejuízos',
    'desvalorização', 'depreciação', 'redução', 'diminuição',
    'retração', 'encolhimento', 'recuo', 'baixa',

    # Strong negative verbs
    'despencou', 'despenca', 'despencar', 'despencando',
    'derreteu', 'derrete', 'derreter', 'derretendo',
    'afundou', 'afunda', 'afundar', 'afundando',
    'caiu', 'cai', 'cair', 'caindo',
    'sangrou', 'sangra', 'sangrar', 'sangrando',
    'desabou', 'desaba', 'desabar', 'desabando',
    'tombou', 'tomba', 'tombar', 'tombando',
    'evaporou', 'evapora', 'evaporar',
    'ruiu', 'ruindo', 'desmoronou', 'desmorona',

    # Failure & Bankruptcy
    'falência', 'falências', 'faliu', 'falir', 'quebra', 'quebras',
    'quebrou', 'quebrar', 'insolvência', 'insolvente',
    'inadimplência', 'inadimplente', 'calote', 'default',
    'moratória', 'concordata', 'recuperação judicial',
    'fracasso', 'fracassar', 'fracassou', 'malogro',

    # Problems & Threats
    'problema', 'problemas', 'problemático', 'dificuldade', 'dificuldades',
    'ameaça', 'ameaças', 'ameaçar', 'ameaçado',
    'obstáculo', 'obstáculos', 'entrave', 'entraves',
    'barreira', 'barreiras', 'impedimento',

    # Deterioration & Worsening
    'piora', 'piorou', 'piorar', 'piorando', 'pior', 'piores',
    'agravamento', 'agravar', 'agravou', 'deterioração',
    'deteriorar', 'deteriorou', 'decadência', 'declínio',

    # Negative sentiment words
    'negativo', 'negativa', 'negativos', 'negativas',
    'ruim', 'ruins', 'péssimo', 'péssima', 'terrível',
    'horrível', 'desastroso', 'desastrosa', 'catastrófico',
    'lamentável', 'deplorável', 'lastimável',

    # Fear & Pessimism
    'pessimista', 'pessimismo', 'pessimistas',
    'preocupante', 'alarmante', 'assustador', 'assustadora',
    'temeroso', 'receoso', 'amedrontado', 'apavorado',

    # Market-specific negative
    'bear', 'bearish', 'vendedor', 'venda forçada',
    'liquidação', 'liquidando', 'realizando',
    'fuga', 'êxodo', 'saída', 'retirada', 'resgate',
    'desalavancagem', 'margin call',

    # Economic negative
    'desemprego', 'demissão', 'demissões', 'corte', 'cortes',
    'austeridade', 'arrocho', 'aperto', 'escassez',
    'déficit', 'dívida', 'dívidas', 'endividado',

    # Rejection & Downgrade
    'rebaixamento', 'rebaixou', 'downgrade',
    'suspensão', 'suspender', 'cancelamento', 'cancelar',
    'rejeição', 'rejeitar', 'rejeitado',

    # Warnings & Alerts
    'alerta', 'alertas', 'aviso', 'avisos', 'advertência',
    'sinal vermelho', 'luz amarela', 'bandeira vermelha',
]

# =============================================================================
# COMBINED DICTIONARY OBJECT
# =============================================================================

DICTIONARIES = {
    'financial': set(FINANCIAL_TERMS),
    'stress': set(STRESS_TERMS),
    'negative': set(NEGATIVE_TERMS),
}

# =============================================================================
# CRISIS EPISODE MARKERS FOR VISUALIZATION
# =============================================================================

CRISIS_EPISODES = {
    # Format: (start_date, end_date, label)
    ('2014-09-01', '2014-12-31'): 'Petrobras Scandal',
    ('2015-08-01', '2016-08-31'): 'Brazilian Recession / Dilma Impeachment',
    ('2017-05-01', '2017-06-30'): 'JBS Scandal (Joesley Day)',
    ('2018-05-01', '2018-06-30'): 'Truckers Strike',
    ('2020-02-15', '2020-04-30'): 'COVID-19 Crash',
    ('2021-09-01', '2021-12-31'): 'Fiscal Concerns (PEC)',
    ('2022-09-01', '2022-11-30'): 'Election Uncertainty',
    ('2023-01-01', '2023-01-15'): 'January 8 Events',
}


def get_dictionary_stats():
    """Return statistics about the dictionaries."""
    return {
        'financial_terms': len(FINANCIAL_TERMS),
        'stress_terms': len(STRESS_TERMS),
        'negative_terms': len(NEGATIVE_TERMS),
        'total_unique': len(DICTIONARIES['financial'] |
                          DICTIONARIES['stress'] |
                          DICTIONARIES['negative']),
        'overlap_stress_negative': len(DICTIONARIES['stress'] &
                                       DICTIONARIES['negative']),
    }


if __name__ == '__main__':
    stats = get_dictionary_stats()
    print("Dictionary Statistics:")
    print(f"  Financial terms: {stats['financial_terms']}")
    print(f"  Stress terms: {stats['stress_terms']}")
    print(f"  Negative terms: {stats['negative_terms']}")
    print(f"  Total unique terms: {stats['total_unique']}")
    print(f"  Overlap (stress & negative): {stats['overlap_stress_negative']}")
