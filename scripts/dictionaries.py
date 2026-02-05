#!/usr/bin/env python3
"""
Portuguese Financial Dictionaries & Google Trends Queries
=========================================================
Script 1 of 3: Dictionary and Query definitions

Contains:
1. Text dictionaries for Twitter three-way co-occurrence (Baker et al. 2019)
2. Google Trends pre-combined stress queries (Da et al. 2011 approach)

Usage:
    # Import in other scripts
    from scripts.dictionaries import FINANCIAL_TERMS, STRESS_QUERIES_GT

    # Run standalone to see statistics
    python scripts/dictionaries.py
"""

# =============================================================================
# PART 1: TEXT DICTIONARIES (For Twitter Classification)
# =============================================================================

# -----------------------------------------------------------------------------
# CATEGORY 1: FINANCIAL TERMS (~270 terms)
# -----------------------------------------------------------------------------
FINANCIAL_TERMS = [
    # Stock Markets & Exchanges
    'mercado', 'mercados', 'mercado financeiro', 'mercado de capitais',
    'bolsa', 'bolsas', 'bolsa de valores', 'ibovespa', 'ibov', 'b3', 'bovespa',
    'pregão', 'pregões', 'after market', 'leilão',
    'nasdaq', 'nyse', 'dow jones', 's&p', 's&p500', 'sp500',
    'ftse', 'dax', 'nikkei', 'hang seng',

    # Financial Instruments
    'ação', 'ações', 'ativo', 'ativos', 'papel', 'papéis',
    'título', 'títulos', 'título público', 'tesouro direto',
    'bond', 'bonds', 'debênture', 'debêntures', 'cri', 'cra',
    'lci', 'lca', 'cdb', 'rdb', 'poupança',
    'derivativo', 'derivativos', 'opção', 'opções', 'call', 'put',
    'futuro', 'futuros', 'contrato futuro', 'mini índice', 'mini dólar',
    'swap', 'swaps', 'hedge', 'hedging',
    'fundo', 'fundos', 'fundo de investimento', 'fii', 'fiis',
    'etf', 'etfs', 'cota', 'cotas', 'bdr', 'bdrs',
    'renda fixa', 'renda variável', 'multimercado',

    # Banking & Credit
    'banco', 'bancos', 'bancário', 'bancária', 'bancários', 'setor bancário',
    'crédito', 'empréstimo', 'empréstimos', 'financiamento', 'financiamentos',
    'juros', 'taxa de juros', 'selic', 'taxa selic', 'copom',
    'spread', 'spread bancário', 'cdi', 'di', 'overnight',
    'cheque especial', 'cartão de crédito', 'consignado', 'rotativo',
    'inadimplência', 'calote', 'default', 'atraso',

    # Economy & Macroeconomics
    'economia', 'econômico', 'econômica', 'economia brasileira',
    'pib', 'produto interno bruto', 'crescimento econômico',
    'inflação', 'ipca', 'igpm', 'igp-m', 'inpc', 'deflação', 'hiperinflação',
    'fiscal', 'política fiscal', 'ajuste fiscal', 'teto de gastos',
    'monetário', 'monetária', 'política monetária',
    'câmbio', 'taxa de câmbio', 'dólar', 'euro', 'moeda', 'real',
    'forex', 'divisas', 'brl', 'usd', 'ptax',
    'balança comercial', 'exportação', 'importação', 'superávit', 'déficit',
    'dívida pública', 'dívida externa', 'reservas internacionais',

    # Investment & Finance
    'investimento', 'investimentos', 'investidor', 'investidores',
    'investidor pessoa física', 'investidor institucional', 'estrangeiro',
    'carteira', 'portfólio', 'alocação', 'diversificação', 'rebalanceamento',
    'rentabilidade', 'retorno', 'rendimento', 'yield', 'dividend yield',
    'dividendo', 'dividendos', 'jcp', 'juros sobre capital próprio',
    'valorização', 'desvalorização', 'ganho', 'lucro', 'prejuízo',
    'liquidez', 'volume', 'giro', 'negociação', 'trade', 'trading',
    'compra', 'venda', 'ordem', 'book', 'oferta', 'demanda',

    # Financial Institutions
    'bacen', 'banco central', 'bcb', 'bc', 'cvm', 'anbima',
    'tesouro', 'tesouro nacional', 'receita federal',
    'bndes', 'caixa', 'bb', 'banco do brasil',
    'itaú', 'bradesco', 'santander', 'nubank', 'btg', 'xp',
    'corretora', 'corretoras', 'gestora', 'gestoras', 'asset',
    'fundo soberano', 'private equity', 'venture capital',

    # Corporate Finance
    'empresa', 'empresas', 'companhia', 'companhias', 'corporativo',
    'balanço', 'balanço patrimonial', 'demonstração', 'resultado',
    'receita', 'faturamento', 'custo', 'despesa', 'margem',
    'ebitda', 'ebit', 'lucro líquido', 'lucro bruto', 'prejuízo líquido',
    'endividamento', 'alavancagem', 'dívida líquida', 'passivo', 'patrimônio',
    'ipo', 'oferta pública', 'follow-on', 'subscrição', 'bonificação',
    'm&a', 'fusão', 'aquisição', 'incorporação', 'cisão',
    'governança', 'governança corporativa', 'conselho', 'acionista',

    # Brazilian Specific
    'petrobras', 'petr4', 'petr3', 'vale', 'vale3',
    'itub4', 'bbdc4', 'bbas3', 'sanb11', 'abev3',
    'wege3', 'rent3', 'lren3', 'mglu3', 'vvar3',
    'ifix', 'idiv', 'small caps', 'ibovespa futuro',
    'previdência', 'previdência privada', 'pgbl', 'vgbl',
]

# -----------------------------------------------------------------------------
# CATEGORY 2: STRESS TERMS (~160 terms)
# -----------------------------------------------------------------------------
STRESS_TERMS = [
    # Crisis & Instability
    'crise', 'crises', 'crise financeira', 'crise econômica',
    'crise política', 'crise institucional', 'crise fiscal',
    'instabilidade', 'instável', 'desestabilização', 'desequilíbrio',
    'turbulência', 'turbulento', 'agitação', 'tumulto',
    'caos', 'caótico', 'desordem', 'confusão',

    # Risk & Uncertainty
    'risco', 'riscos', 'risco sistêmico', 'risco de crédito',
    'risco de mercado', 'risco político', 'risco país', 'risco brasil',
    'incerteza', 'incertezas', 'incerto', 'indefinição', 'indefinido',
    'imprevisibilidade', 'imprevisível', 'volatilidade', 'volátil',
    'insegurança', 'inseguro', 'vulnerabilidade', 'vulnerável',
    'exposição', 'exposto', 'fragilidade', 'frágil',

    # Fear & Panic
    'pânico', 'medo', 'temor', 'pavor', 'terror',
    'apreensão', 'preocupação', 'preocupante', 'inquietação',
    'nervosismo', 'nervoso', 'ansiedade', 'ansioso',
    'desespero', 'desesperado', 'histeria', 'frenesi',
    'fuga', 'êxodo', 'corrida', 'debandada',

    # Collapse & Disaster
    'colapso', 'desabamento', 'derrocada', 'ruína',
    'desmoronamento', 'desintegração', 'implosão', 'explosão',
    'desastre', 'catástrofe', 'catastrófico', 'calamidade',
    'destruição', 'devastação', 'arrasamento',
    'crash', 'quebra', 'estouro', 'bolha',
    'debacle', 'fiasco', 'fracasso', 'malogro',

    # Market Stress Indicators
    'circuit breaker', 'sell-off', 'selloff', 'sell off',
    'bear market', 'mercado bear', 'baixista',
    'correção', 'correção forte', 'ajuste', 'realização',
    'derretimento', 'sangria', 'hemorragia', 'carnificina',
    'capitulação', 'liquidação forçada', 'margin call',
    'flash crash', 'black monday', 'black friday',

    # Economic Stress
    'recessão', 'recessivo', 'depressão', 'depressivo',
    'estagflação', 'estagnação', 'contração', 'retração',
    'desaceleração', 'enfraquecimento', 'deterioração',
    'colapso econômico', 'crise sistêmica', 'contágio',
    'efeito dominó', 'cascata', 'spillover', 'transbordamento',

    # Urgency & Severity
    'urgente', 'urgência', 'emergência', 'emergencial',
    'alerta', 'alarme', 'aviso', 'atenção',
    'grave', 'gravidade', 'gravíssimo', 'sério', 'crítico',
    'extremo', 'severo', 'drástico', 'radical',
    'sem precedentes', 'histórico', 'recorde negativo',

    # Brazilian Political/Economic Stress
    'impeachment', 'cassação', 'intervenção', 'interventor',
    'greve', 'paralisação', 'lockout', 'protestos',
    'reforma', 'reformas', 'pec', 'medida provisória',
    'delação', 'lava jato', 'corrupção', 'escândalo',
]

# -----------------------------------------------------------------------------
# CATEGORY 3: NEGATIVE TERMS (~440 terms)
# -----------------------------------------------------------------------------
NEGATIVE_TERMS = [
    # Losses & Declines (Strong)
    'queda', 'quedas', 'cair', 'caiu', 'cai', 'caindo', 'caíram',
    'perda', 'perdas', 'perder', 'perdeu', 'perde', 'perdendo', 'perderam',
    'prejuízo', 'prejuízos', 'prejuízo bilionário', 'prejuízo recorde',
    'desvalorização', 'desvalorizar', 'desvalorizou', 'desvalorizado',
    'depreciação', 'depreciar', 'depreciou', 'depreciado',
    'retração', 'retrair', 'retraiu', 'retraído', 'encolhimento',
    'recuo', 'recuar', 'recuou', 'recuando', 'recuaram',
    'baixa', 'baixas', 'baixar', 'baixou', 'em baixa',
    'redução', 'reduzir', 'reduziu', 'reduzido', 'reduzindo',
    'diminuição', 'diminuir', 'diminuiu', 'diminuído',
    'declínio', 'declinar', 'declinou', 'declinante',
    'contração', 'contrair', 'contraiu', 'contraído',

    # Dramatic Decline Verbs
    'despencar', 'despencou', 'despenca', 'despencando', 'despencaram',
    'desabar', 'desabou', 'desaba', 'desabando', 'desabaram',
    'derreter', 'derreteu', 'derrete', 'derretendo', 'derreteram',
    'afundar', 'afundou', 'afunda', 'afundando', 'afundaram',
    'desmoronar', 'desmoronou', 'desmorona', 'desmoronando',
    'ruir', 'ruiu', 'rui', 'ruindo', 'ruíram',
    'tombar', 'tombou', 'tomba', 'tombando', 'tombaram',
    'sangrar', 'sangrou', 'sangra', 'sangrando', 'sangraram',
    'evaporar', 'evaporou', 'evapora', 'evaporando', 'evaporaram',
    'sumir', 'sumiu', 'some', 'sumindo', 'sumiram',
    'detonar', 'detonou', 'detona', 'detonando',
    'implodir', 'implodiu', 'implode', 'implodindo',
    'explodir', 'explodiu', 'explode', 'explodindo',

    # Failure & Bankruptcy
    'falência', 'falências', 'falir', 'faliu', 'falido', 'falida',
    'quebra', 'quebras', 'quebrar', 'quebrou', 'quebrado', 'quebrada',
    'insolvência', 'insolvente', 'iliquidez', 'ilíquido',
    'inadimplência', 'inadimplente', 'inadimplentes',
    'calote', 'calotes', 'caloteiro', 'dar calote',
    'default', 'moratória', 'reestruturação de dívida',
    'concordata', 'recuperação judicial', 'rj', 'falência decretada',
    'liquidação', 'liquidar', 'liquidado', 'liquidação extrajudicial',
    'encerramento', 'encerrar', 'encerrou', 'fechamento',
    'intervenção', 'intervir', 'interveio', 'interventor',

    # Fraud & Scandal
    'fraude', 'fraudes', 'fraudulento', 'fraudar', 'fraudado',
    'golpe', 'golpes', 'golpista', 'esquema', 'pirâmide',
    'irregularidade', 'irregularidades', 'irregular',
    'ilegalidade', 'ilegal', 'ilegais', 'ilícito', 'ilícitos',
    'corrupção', 'corrupto', 'corruptos', 'propina', 'suborno',
    'escândalo', 'escândalos', 'escandaloso',
    'manipulação', 'manipular', 'manipulado', 'manipulador',
    'insider', 'insider trading', 'informação privilegiada',
    'lavagem', 'lavagem de dinheiro', 'evasão', 'sonegação',

    # Problems & Difficulties
    'problema', 'problemas', 'problemático', 'problemática',
    'dificuldade', 'dificuldades', 'difícil', 'dificílimo',
    'obstáculo', 'obstáculos', 'barreira', 'barreiras', 'entrave', 'entraves',
    'impedimento', 'impedir', 'impedido', 'bloqueio', 'bloqueado',
    'complicação', 'complicações', 'complicado', 'complicada',
    'adversidade', 'adversidades', 'adverso', 'adversa',
    'contratempo', 'contratempos', 'revés', 'reveses',

    # Negative Outlook
    'pessimista', 'pessimismo', 'pessimistas',
    'negativo', 'negativa', 'negativos', 'negativas', 'negativamente',
    'desfavorável', 'desfavoráveis', 'contrário', 'contrária',
    'ruim', 'ruins', 'péssimo', 'péssima', 'péssimos',
    'terrível', 'terríveis', 'horrível', 'horríveis', 'horroroso',
    'desastroso', 'desastrosa', 'catastrófico', 'catastrófica',
    'lamentável', 'deplorável', 'lastimável', 'trágico', 'trágica',

    # Deterioration & Worsening
    'piora', 'pioras', 'piorar', 'piorou', 'piorando', 'pioram',
    'pior', 'piores', 'o pior', 'ainda pior',
    'agravar', 'agravou', 'agravamento', 'agravando', 'agravado',
    'deteriorar', 'deteriorou', 'deterioração', 'deteriorando',
    'degradar', 'degradou', 'degradação', 'degradando',
    'decair', 'decaiu', 'decadência', 'decaindo',
    'degenerar', 'degenerou', 'degeneração', 'degenerando',

    # Warnings & Threats
    'ameaça', 'ameaças', 'ameaçar', 'ameaçado', 'ameaçando',
    'alerta', 'alertas', 'alertar', 'alertado', 'alertando',
    'aviso', 'avisos', 'avisar', 'avisado', 'avisando',
    'advertência', 'advertências', 'advertir', 'advertido',
    'sinal de alerta', 'bandeira vermelha', 'luz vermelha',
    'sinal amarelo', 'preocupante', 'preocupado', 'preocupar',

    # Rejection & Downgrade
    'rebaixamento', 'rebaixar', 'rebaixou', 'rebaixado',
    'downgrade', 'downgrades', 'corte', 'cortes', 'cortado',
    'revisão para baixo', 'revisão negativa', 'outlook negativo',
    'suspensão', 'suspender', 'suspendeu', 'suspenso', 'suspensa',
    'cancelamento', 'cancelar', 'cancelou', 'cancelado', 'cancelada',
    'rejeição', 'rejeitar', 'rejeitou', 'rejeitado', 'rejeitada',
    'veto', 'vetar', 'vetou', 'vetado', 'reprovação', 'reprovar',

    # Economic Negative
    'desemprego', 'desempregado', 'desempregados', 'demissão', 'demissões',
    'demitir', 'demitiu', 'demitido', 'demitidos', 'corte de pessoal',
    'austeridade', 'arrocho', 'aperto', 'contenção', 'restrição',
    'escassez', 'falta', 'carência', 'penúria', 'racionamento',
    'déficit', 'deficitário', 'rombo', 'buraco', 'buraco fiscal',
    'dívida', 'dívidas', 'endividado', 'endividamento', 'superendividado',
    'inflação alta', 'inflação elevada', 'carestia', 'encarecimento',

    # Market Negative Actions
    'vender', 'vendeu', 'vende', 'vendendo', 'venderam',
    'venda forçada', 'vendas massivas', 'pressão vendedora',
    'realizar', 'realizou', 'realização', 'realizando',
    'resgatar', 'resgatou', 'resgate', 'resgatando', 'resgates',
    'saída', 'saídas', 'sair', 'saiu', 'saindo', 'saíram',
    'retirada', 'retirar', 'retirou', 'retirando', 'retiraram',
    'fuga de capital', 'fuga de investidores', 'debandada',
    'desalavancagem', 'desalavancar', 'reduzir posição',

    # Uncertainty Negative
    'dúvida', 'dúvidas', 'duvidar', 'duvidoso', 'duvidosa',
    'incerto', 'incerta', 'incertos', 'incerteza', 'incertezas',
    'questionável', 'questionar', 'questionado', 'questionamento',
    'suspeito', 'suspeita', 'suspeitos', 'suspeitar', 'sob suspeita',
    'desconfiança', 'desconfiar', 'desconfiado', 'desconfiança',
]

# =============================================================================
# PART 2: GOOGLE TRENDS QUERIES (Pre-combined Stress Phrases)
# =============================================================================
# These are search queries that already combine financial + stress + negative
# Following Da, Engelberg, Gao (2011) and Dzielinski (2012) approach

# -----------------------------------------------------------------------------
# TIER 1: Financial Crises (Highest Priority)
# -----------------------------------------------------------------------------
STRESS_QUERIES_TIER1 = [
    "crise financeira",           # Financial crisis
    "crise econômica",            # Economic crisis
    "pânico financeiro",          # Financial panic
    "colapso econômico",          # Economic collapse
    "crash bolsa",                # Stock market crash
]

# -----------------------------------------------------------------------------
# TIER 2: Market Stress
# -----------------------------------------------------------------------------
STRESS_QUERIES_TIER2 = [
    "queda ibovespa",             # Ibovespa fall
    "volatilidade bolsa",         # Stock volatility
    "bear market brasil",         # Bear market Brazil
    "fuga capitais",              # Capital flight
    "bolsa despenca",             # Stock market plunges
]

# -----------------------------------------------------------------------------
# TIER 3: Macro/Sovereign Stress
# -----------------------------------------------------------------------------
STRESS_QUERIES_TIER3 = [
    "risco default brasil",       # Brazil default risk
    "rebaixamento rating brasil", # Brazil rating downgrade
    "dívida pública brasil",      # Brazil public debt
    "recessão brasil",            # Brazil recession
    "desemprego brasil",          # Brazil unemployment
]

# -----------------------------------------------------------------------------
# TIER 4: Currency/Banking Stress
# -----------------------------------------------------------------------------
STRESS_QUERIES_TIER4 = [
    "desvalorização real",        # Real devaluation
    "dólar dispara",              # Dollar surges
    "crise bancária",             # Banking crisis
    "corrida bancária",           # Bank run
    "alta juros brasil",          # Brazil interest rate hike
]

# -----------------------------------------------------------------------------
# TIER 5: Investor Sentiment/Behavior
# -----------------------------------------------------------------------------
STRESS_QUERIES_TIER5 = [
    "investir agora",             # Invest now (contrarian indicator)
    "vender ações",               # Sell stocks
    "mercado vai cair",           # Market will fall
    "proteção carteira",          # Portfolio protection
    "como proteger investimentos", # How to protect investments
]

# All Google Trends queries combined
STRESS_QUERIES_GT = (
    STRESS_QUERIES_TIER1 +
    STRESS_QUERIES_TIER2 +
    STRESS_QUERIES_TIER3 +
    STRESS_QUERIES_TIER4 +
    STRESS_QUERIES_TIER5
)

# Query weights (higher = more important for FSI)
QUERY_WEIGHTS = {
    # Tier 1: Weight 1.5 (highest priority)
    "crise financeira": 1.5,
    "crise econômica": 1.5,
    "pânico financeiro": 1.5,
    "colapso econômico": 1.5,
    "crash bolsa": 1.5,
    # Tier 2: Weight 1.2
    "queda ibovespa": 1.2,
    "volatilidade bolsa": 1.2,
    "bear market brasil": 1.2,
    "fuga capitais": 1.2,
    "bolsa despenca": 1.2,
    # Tier 3-5: Weight 1.0 (default)
}

# =============================================================================
# CRISIS EPISODES FOR BRAZIL (For Visualization)
# =============================================================================

CRISIS_EPISODES = {
    ('2008-09-15', '2009-03-31'): 'Global Financial Crisis',
    ('2014-09-01', '2014-12-31'): 'Petrobras Scandal Begins',
    ('2015-04-01', '2016-08-31'): 'Brazilian Recession / Dilma Impeachment',
    ('2017-05-17', '2017-06-30'): 'JBS Scandal (Joesley Day)',
    ('2018-05-21', '2018-06-10'): 'Truckers Strike',
    ('2020-02-21', '2020-04-30'): 'COVID-19 Crash',
    ('2021-09-01', '2021-12-31'): 'Fiscal Concerns (PEC dos Precatórios)',
    ('2022-09-01', '2022-11-15'): 'Election Uncertainty',
    ('2023-01-08', '2023-01-15'): 'January 8 Events',
    ('2024-04-01', '2024-05-31'): 'Fiscal Framework Concerns',
}

# =============================================================================
# DICTIONARY SETS (For Efficient Lookup)
# =============================================================================

DICTIONARIES = {
    'financial': set(FINANCIAL_TERMS),
    'stress': set(STRESS_TERMS),
    'negative': set(NEGATIVE_TERMS),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_dictionary_stats() -> dict:
    """Return statistics about the text dictionaries."""
    all_terms = (
        DICTIONARIES['financial'] |
        DICTIONARIES['stress'] |
        DICTIONARIES['negative']
    )

    return {
        'financial_terms': len(FINANCIAL_TERMS),
        'stress_terms': len(STRESS_TERMS),
        'negative_terms': len(NEGATIVE_TERMS),
        'total_unique': len(all_terms),
        'google_trends_queries': len(STRESS_QUERIES_GT),
        'overlap_stress_negative': len(
            DICTIONARIES['stress'] & DICTIONARIES['negative']
        ),
    }


def get_query_weight(query: str) -> float:
    """Get weight for a Google Trends query."""
    return QUERY_WEIGHTS.get(query, 1.0)


def search_term(term: str) -> dict:
    """Check which dictionaries contain a term."""
    term_lower = term.lower()
    return {
        'financial': term_lower in DICTIONARIES['financial'],
        'stress': term_lower in DICTIONARIES['stress'],
        'negative': term_lower in DICTIONARIES['negative'],
    }


def print_dictionary_report():
    """Print detailed dictionary and query statistics."""
    stats = get_dictionary_stats()

    print("=" * 60)
    print("PORTUGUESE FSI DICTIONARIES & GOOGLE TRENDS QUERIES")
    print("=" * 60)

    print(f"\n📚 TEXT DICTIONARIES (Twitter Classification):")
    print(f"   Financial terms: {stats['financial_terms']:,}")
    print(f"   Stress terms:    {stats['stress_terms']:,}")
    print(f"   Negative terms:  {stats['negative_terms']:,}")
    print(f"   ─────────────────────────")
    print(f"   Total unique:    {stats['total_unique']:,}")

    print(f"\n🔍 GOOGLE TRENDS QUERIES:")
    print(f"   Total queries:   {stats['google_trends_queries']}")
    print(f"\n   Tier 1 (Crisis):     {len(STRESS_QUERIES_TIER1)} queries")
    for q in STRESS_QUERIES_TIER1:
        print(f"      • {q}")
    print(f"\n   Tier 2 (Market):     {len(STRESS_QUERIES_TIER2)} queries")
    for q in STRESS_QUERIES_TIER2:
        print(f"      • {q}")
    print(f"\n   Tier 3 (Macro):      {len(STRESS_QUERIES_TIER3)} queries")
    for q in STRESS_QUERIES_TIER3:
        print(f"      • {q}")
    print(f"\n   Tier 4 (Currency):   {len(STRESS_QUERIES_TIER4)} queries")
    for q in STRESS_QUERIES_TIER4:
        print(f"      • {q}")
    print(f"\n   Tier 5 (Sentiment):  {len(STRESS_QUERIES_TIER5)} queries")
    for q in STRESS_QUERIES_TIER5:
        print(f"      • {q}")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print_dictionary_report()

    # Test term lookup
    print("\n🔍 Term Lookup Examples:")
    test_terms = ['crise', 'mercado', 'queda', 'investimento', 'pânico']
    for term in test_terms:
        result = search_term(term)
        categories = [k for k, v in result.items() if v]
        print(f"   '{term}': {', '.join(categories) if categories else 'not found'}")
