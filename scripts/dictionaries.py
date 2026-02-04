#!/usr/bin/env python3
"""
Portuguese Financial Dictionaries for FSI
==========================================
Script 1 of 3: Dictionary definitions

Expanded dictionaries based on:
- Baker et al. (2019) methodology
- Loughran-McDonald financial sentiment adapted for Portuguese
- Brazilian market-specific terminology

Three categories for three-way co-occurrence:
1. Financial Terms - Identify financial/economic context
2. Stress Terms - Identify stress/uncertainty/crisis context
3. Negative Terms - Identify negative sentiment

Usage:
    # Import dictionaries in other scripts
    from scripts.dictionaries import FINANCIAL_TERMS, STRESS_TERMS, NEGATIVE_TERMS

    # Or run standalone to see statistics
    python scripts/dictionaries.py
"""

# =============================================================================
# CATEGORY 1: FINANCIAL TERMS (~200 terms)
# =============================================================================
# Terms that indicate financial/economic context
# A post must contain at least one of these to be considered financial

FINANCIAL_TERMS = [
    # -------------------------------------------------------------------------
    # Stock Markets & Exchanges
    # -------------------------------------------------------------------------
    'mercado', 'mercados', 'mercado financeiro', 'mercado de capitais',
    'bolsa', 'bolsas', 'bolsa de valores', 'ibovespa', 'ibov', 'b3', 'bovespa',
    'pregão', 'pregões', 'after market', 'leilão',
    'nasdaq', 'nyse', 'dow jones', 's&p', 's&p500', 'sp500',
    'ftse', 'dax', 'nikkei', 'hang seng',

    # -------------------------------------------------------------------------
    # Financial Instruments
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Banking & Credit
    # -------------------------------------------------------------------------
    'banco', 'bancos', 'bancário', 'bancária', 'bancários', 'setor bancário',
    'crédito', 'empréstimo', 'empréstimos', 'financiamento', 'financiamentos',
    'juros', 'taxa de juros', 'selic', 'taxa selic', 'copom',
    'spread', 'spread bancário', 'cdi', 'di', 'overnight',
    'cheque especial', 'cartão de crédito', 'consignado', 'rotativo',
    'inadimplência', 'calote', 'default', 'atraso',

    # -------------------------------------------------------------------------
    # Economy & Macroeconomics
    # -------------------------------------------------------------------------
    'economia', 'econômico', 'econômica', 'economia brasileira',
    'pib', 'produto interno bruto', 'crescimento econômico',
    'inflação', 'ipca', 'igpm', 'igp-m', 'inpc', 'deflação', 'hiperinflação',
    'fiscal', 'política fiscal', 'ajuste fiscal', 'teto de gastos',
    'monetário', 'monetária', 'política monetária',
    'câmbio', 'taxa de câmbio', 'dólar', 'euro', 'moeda', 'real',
    'forex', 'divisas', 'brl', 'usd', 'ptax',
    'balança comercial', 'exportação', 'importação', 'superávit', 'déficit',
    'dívida pública', 'dívida externa', 'reservas internacionais',

    # -------------------------------------------------------------------------
    # Investment & Finance
    # -------------------------------------------------------------------------
    'investimento', 'investimentos', 'investidor', 'investidores',
    'investidor pessoa física', 'investidor institucional', 'estrangeiro',
    'carteira', 'portfólio', 'alocação', 'diversificação', 'rebalanceamento',
    'rentabilidade', 'retorno', 'rendimento', 'yield', 'dividend yield',
    'dividendo', 'dividendos', 'jcp', 'juros sobre capital próprio',
    'valorização', 'desvalorização', 'ganho', 'lucro', 'prejuízo',
    'liquidez', 'volume', 'giro', 'negociação', 'trade', 'trading',
    'compra', 'venda', 'ordem', 'book', 'oferta', 'demanda',

    # -------------------------------------------------------------------------
    # Financial Institutions
    # -------------------------------------------------------------------------
    'bacen', 'banco central', 'bcb', 'bc', 'cvm', 'anbima',
    'tesouro', 'tesouro nacional', 'receita federal',
    'bndes', 'caixa', 'bb', 'banco do brasil',
    'itaú', 'bradesco', 'santander', 'nubank', 'btg', 'xp',
    'corretora', 'corretoras', 'gestora', 'gestoras', 'asset',
    'fundo soberano', 'private equity', 'venture capital',

    # -------------------------------------------------------------------------
    # Corporate Finance
    # -------------------------------------------------------------------------
    'empresa', 'empresas', 'companhia', 'companhias', 'corporativo',
    'balanço', 'balanço patrimonial', 'demonstração', 'resultado',
    'receita', 'faturamento', 'custo', 'despesa', 'margem',
    'ebitda', 'ebit', 'lucro líquido', 'lucro bruto', 'prejuízo líquido',
    'endividamento', 'alavancagem', 'dívida líquida', 'passivo', 'patrimônio',
    'ipo', 'oferta pública', 'follow-on', 'subscrição', 'bonificação',
    'm&a', 'fusão', 'aquisição', 'incorporação', 'cisão',
    'governança', 'governança corporativa', 'conselho', 'acionista',

    # -------------------------------------------------------------------------
    # Brazilian Specific (Companies & Indices)
    # -------------------------------------------------------------------------
    'petrobras', 'petr4', 'petr3', 'vale', 'vale3',
    'itub4', 'bbdc4', 'bbas3', 'sanb11', 'abev3',
    'wege3', 'rent3', 'lren3', 'mglu3', 'vvar3',
    'ifix', 'idiv', 'small caps', 'ibovespa futuro',
    'previdência', 'previdência privada', 'pgbl', 'vgbl',
]

# =============================================================================
# CATEGORY 2: STRESS TERMS (~120 terms)
# =============================================================================
# Terms that indicate stress, uncertainty, crisis, or volatility
# These capture the "stress" component of financial stress

STRESS_TERMS = [
    # -------------------------------------------------------------------------
    # Crisis & Instability
    # -------------------------------------------------------------------------
    'crise', 'crises', 'crise financeira', 'crise econômica',
    'crise política', 'crise institucional', 'crise fiscal',
    'instabilidade', 'instável', 'desestabilização', 'desequilíbrio',
    'turbulência', 'turbulento', 'agitação', 'tumulto',
    'caos', 'caótico', 'desordem', 'confusão',

    # -------------------------------------------------------------------------
    # Risk & Uncertainty
    # -------------------------------------------------------------------------
    'risco', 'riscos', 'risco sistêmico', 'risco de crédito',
    'risco de mercado', 'risco político', 'risco país', 'risco brasil',
    'incerteza', 'incertezas', 'incerto', 'indefinição', 'indefinido',
    'imprevisibilidade', 'imprevisível', 'volatilidade', 'volátil',
    'insegurança', 'inseguro', 'vulnerabilidade', 'vulnerável',
    'exposição', 'exposto', 'fragilidade', 'frágil',

    # -------------------------------------------------------------------------
    # Fear & Panic
    # -------------------------------------------------------------------------
    'pânico', 'medo', 'temor', 'pavor', 'terror',
    'apreensão', 'preocupação', 'preocupante', 'inquietação',
    'nervosismo', 'nervoso', 'ansiedade', 'ansioso',
    'desespero', 'desesperado', 'histeria', 'frenesi',
    'fuga', 'êxodo', 'corrida', 'debandada',

    # -------------------------------------------------------------------------
    # Collapse & Disaster
    # -------------------------------------------------------------------------
    'colapso', 'desabamento', 'derrocada', 'ruína',
    'desmoronamento', 'desintegração', 'implosão', 'explosão',
    'desastre', 'catástrofe', 'catastrófico', 'calamidade',
    'destruição', 'devastação', 'arrasamento',
    'crash', 'quebra', 'estouro', 'bolha',
    'debacle', 'fiasco', 'fracasso', 'malogro',

    # -------------------------------------------------------------------------
    # Market Stress Indicators
    # -------------------------------------------------------------------------
    'circuit breaker', 'sell-off', 'selloff', 'sell off',
    'bear market', 'mercado bear', 'baixista',
    'correção', 'correção forte', 'ajuste', 'realização',
    'derretimento', 'sangria', 'hemorragia', 'carnificina',
    'capitulação', 'liquidação forçada', 'margin call',
    'flash crash', 'black monday', 'black friday',

    # -------------------------------------------------------------------------
    # Economic Stress
    # -------------------------------------------------------------------------
    'recessão', 'recessivo', 'depressão', 'depressivo',
    'estagflação', 'estagnação', 'contração', 'retração',
    'desaceleração', 'enfraquecimento', 'deterioração',
    'colapso econômico', 'crise sistêmica', 'contágio',
    'efeito dominó', 'cascata', 'spillover', 'transbordamento',

    # -------------------------------------------------------------------------
    # Urgency & Severity
    # -------------------------------------------------------------------------
    'urgente', 'urgência', 'emergência', 'emergencial',
    'alerta', 'alarme', 'aviso', 'atenção',
    'grave', 'gravidade', 'gravíssimo', 'sério', 'crítico',
    'extremo', 'severo', 'drástico', 'radical',
    'sem precedentes', 'histórico', 'recorde negativo',

    # -------------------------------------------------------------------------
    # Brazilian Political/Economic Stress
    # -------------------------------------------------------------------------
    'impeachment', 'cassação', 'intervenção', 'interventor',
    'greve', 'paralisação', 'lockout', 'protestos',
    'reforma', 'reformas', 'pec', 'medida provisória',
    'delação', 'lava jato', 'corrupção', 'escândalo',
]

# =============================================================================
# CATEGORY 3: NEGATIVE TERMS (~250 terms)
# =============================================================================
# Negative sentiment words adapted from Loughran-McDonald for Portuguese
# Focus on financial/economic negative context

NEGATIVE_TERMS = [
    # -------------------------------------------------------------------------
    # Losses & Declines (Strong)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Dramatic Decline Verbs
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Failure & Bankruptcy
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Fraud & Scandal
    # -------------------------------------------------------------------------
    'fraude', 'fraudes', 'fraudulento', 'fraudar', 'fraudado',
    'golpe', 'golpes', 'golpista', 'esquema', 'pirâmide',
    'irregularidade', 'irregularidades', 'irregular',
    'ilegalidade', 'ilegal', 'ilegais', 'ilícito', 'ilícitos',
    'corrupção', 'corrupto', 'corruptos', 'propina', 'suborno',
    'escândalo', 'escândalos', 'escandaloso',
    'manipulação', 'manipular', 'manipulado', 'manipulador',
    'insider', 'insider trading', 'informação privilegiada',
    'lavagem', 'lavagem de dinheiro', 'evasão', 'sonegação',

    # -------------------------------------------------------------------------
    # Problems & Difficulties
    # -------------------------------------------------------------------------
    'problema', 'problemas', 'problemático', 'problemática',
    'dificuldade', 'dificuldades', 'difícil', 'dificílimo',
    'obstáculo', 'obstáculos', 'barreira', 'barreiras', 'entrave', 'entraves',
    'impedimento', 'impedir', 'impedido', 'bloqueio', 'bloqueado',
    'complicação', 'complicações', 'complicado', 'complicada',
    'adversidade', 'adversidades', 'adverso', 'adversa',
    'contratempo', 'contratempos', 'revés', 'reveses',

    # -------------------------------------------------------------------------
    # Negative Outlook
    # -------------------------------------------------------------------------
    'pessimista', 'pessimismo', 'pessimistas',
    'negativo', 'negativa', 'negativos', 'negativas', 'negativamente',
    'desfavorável', 'desfavoráveis', 'contrário', 'contrária',
    'ruim', 'ruins', 'péssimo', 'péssima', 'péssimos',
    'terrível', 'terríveis', 'horrível', 'horríveis', 'horroroso',
    'desastroso', 'desastrosa', 'catastrófico', 'catastrófica',
    'lamentável', 'deplorável', 'lastimável', 'trágico', 'trágica',

    # -------------------------------------------------------------------------
    # Deterioration & Worsening
    # -------------------------------------------------------------------------
    'piora', 'pioras', 'piorar', 'piorou', 'piorando', 'pioram',
    'pior', 'piores', 'o pior', 'ainda pior',
    'agravar', 'agravou', 'agravamento', 'agravando', 'agravado',
    'deteriorar', 'deteriorou', 'deterioração', 'deteriorando',
    'degradar', 'degradou', 'degradação', 'degradando',
    'decair', 'decaiu', 'decadência', 'decaindo',
    'degenerar', 'degenerou', 'degeneração', 'degenerando',

    # -------------------------------------------------------------------------
    # Warnings & Threats
    # -------------------------------------------------------------------------
    'ameaça', 'ameaças', 'ameaçar', 'ameaçado', 'ameaçando',
    'alerta', 'alertas', 'alertar', 'alertado', 'alertando',
    'aviso', 'avisos', 'avisar', 'avisado', 'avisando',
    'advertência', 'advertências', 'advertir', 'advertido',
    'sinal de alerta', 'bandeira vermelha', 'luz vermelha',
    'sinal amarelo', 'preocupante', 'preocupado', 'preocupar',

    # -------------------------------------------------------------------------
    # Rejection & Downgrade
    # -------------------------------------------------------------------------
    'rebaixamento', 'rebaixar', 'rebaixou', 'rebaixado',
    'downgrade', 'downgrades', 'corte', 'cortes', 'cortado',
    'revisão para baixo', 'revisão negativa', 'outlook negativo',
    'suspensão', 'suspender', 'suspendeu', 'suspenso', 'suspensa',
    'cancelamento', 'cancelar', 'cancelou', 'cancelado', 'cancelada',
    'rejeição', 'rejeitar', 'rejeitou', 'rejeitado', 'rejeitada',
    'veto', 'vetar', 'vetou', 'vetado', 'reprovação', 'reprovar',

    # -------------------------------------------------------------------------
    # Economic Negative
    # -------------------------------------------------------------------------
    'desemprego', 'desempregado', 'desempregados', 'demissão', 'demissões',
    'demitir', 'demitiu', 'demitido', 'demitidos', 'corte de pessoal',
    'austeridade', 'arrocho', 'aperto', 'contenção', 'restrição',
    'escassez', 'falta', 'carência', 'penúria', 'racionamento',
    'déficit', 'deficitário', 'rombo', 'buraco', 'buraco fiscal',
    'dívida', 'dívidas', 'endividado', 'endividamento', 'superendividado',
    'inflação alta', 'inflação elevada', 'carestia', 'encarecimento',

    # -------------------------------------------------------------------------
    # Market Negative Actions
    # -------------------------------------------------------------------------
    'vender', 'vendeu', 'vende', 'vendendo', 'venderam',
    'venda forçada', 'vendas massivas', 'pressão vendedora',
    'realizar', 'realizou', 'realização', 'realizando',
    'resgatar', 'resgatou', 'resgate', 'resgatando', 'resgates',
    'saída', 'saídas', 'sair', 'saiu', 'saindo', 'saíram',
    'retirada', 'retirar', 'retirou', 'retirando', 'retiraram',
    'fuga de capital', 'fuga de investidores', 'debandada',
    'desalavancagem', 'desalavancar', 'reduzir posição',

    # -------------------------------------------------------------------------
    # Uncertainty Negative
    # -------------------------------------------------------------------------
    'dúvida', 'dúvidas', 'duvidar', 'duvidoso', 'duvidosa',
    'incerto', 'incerta', 'incertos', 'incerteza', 'incertezas',
    'questionável', 'questionar', 'questionado', 'questionamento',
    'suspeito', 'suspeita', 'suspeitos', 'suspeitar', 'sob suspeita',
    'desconfiança', 'desconfiar', 'desconfiado', 'desconfiança',
]

# =============================================================================
# COMBINED DICTIONARY SETS (for efficient lookup)
# =============================================================================

DICTIONARIES = {
    'financial': set(FINANCIAL_TERMS),
    'stress': set(STRESS_TERMS),
    'negative': set(NEGATIVE_TERMS),
}

# =============================================================================
# CRISIS EPISODES FOR BRAZIL (for visualization)
# =============================================================================

CRISIS_EPISODES = {
    # Format: (start_date, end_date): label
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
# HELPER FUNCTIONS
# =============================================================================

def get_dictionary_stats() -> dict:
    """Return statistics about the dictionaries."""
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
        'overlap_financial_stress': len(
            DICTIONARIES['financial'] & DICTIONARIES['stress']
        ),
        'overlap_financial_negative': len(
            DICTIONARIES['financial'] & DICTIONARIES['negative']
        ),
        'overlap_stress_negative': len(
            DICTIONARIES['stress'] & DICTIONARIES['negative']
        ),
    }


def search_term(term: str) -> dict:
    """Check which dictionaries contain a term."""
    term_lower = term.lower()
    return {
        'financial': term_lower in DICTIONARIES['financial'],
        'stress': term_lower in DICTIONARIES['stress'],
        'negative': term_lower in DICTIONARIES['negative'],
    }


def print_dictionary_report():
    """Print detailed dictionary statistics."""
    stats = get_dictionary_stats()

    print("=" * 60)
    print("PORTUGUESE FINANCIAL DICTIONARIES - STATISTICS")
    print("=" * 60)

    print(f"\n📚 Term Counts:")
    print(f"   Financial terms: {stats['financial_terms']:,}")
    print(f"   Stress terms:    {stats['stress_terms']:,}")
    print(f"   Negative terms:  {stats['negative_terms']:,}")
    print(f"   ─────────────────────────")
    print(f"   Total unique:    {stats['total_unique']:,}")

    print(f"\n🔗 Overlaps (terms in multiple categories):")
    print(f"   Financial ∩ Stress:   {stats['overlap_financial_stress']}")
    print(f"   Financial ∩ Negative: {stats['overlap_financial_negative']}")
    print(f"   Stress ∩ Negative:    {stats['overlap_stress_negative']}")

    print(f"\n📋 Sample Terms:")
    print(f"   Financial: {', '.join(list(DICTIONARIES['financial'])[:5])}...")
    print(f"   Stress:    {', '.join(list(DICTIONARIES['stress'])[:5])}...")
    print(f"   Negative:  {', '.join(list(DICTIONARIES['negative'])[:5])}...")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print_dictionary_report()

    # Test some terms
    print("\n🔍 Term Lookup Examples:")
    test_terms = ['crise', 'mercado', 'queda', 'investimento', 'pânico']
    for term in test_terms:
        result = search_term(term)
        categories = [k for k, v in result.items() if v]
        print(f"   '{term}': {', '.join(categories) if categories else 'not found'}")
