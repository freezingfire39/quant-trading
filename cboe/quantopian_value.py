"""
Original by: Christopher Cain, CMT & Larry Connors
Posted here: https://www.quantopian.com/posts/new-strategy-presenting-the-quality-companies-in-an-uptrend-model-1
(Dan Whitnabl version with fixed bonds weights)
(Nathan Wells modified for performace and logging)
"""
# Quality companies in an uptrend
import quantopian.algorithm as algo
 
# import things need to run pipeline
from quantopian.pipeline import Pipeline
 
# import any built-in factors and filters being used
from quantopian.pipeline.filters import Q500US, Q1500US, Q3000US, QTradableStocksUS, StaticAssets
from quantopian.pipeline.factors import SimpleMovingAverage as SMA
from quantopian.pipeline.factors import ExponentialWeightedMovingAverage as EMA
from quantopian.pipeline.factors import CustomFactor, Returns
 
# import any needed datasets
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.morningstar import Fundamentals as ms
 
# import optimize for trade execution
import quantopian.optimize as opt
# import numpy and pandas because they rock
import numpy as np
import pandas as pd
 
def initialize(context):
    # Set algo 'constants'...
    # List of bond ETFs when market is down. Can be more than one.
    context.BONDS = [symbol('IEF'), symbol('TLT')]
 
    # Set target number of securities to hold and top ROE qty to filter
    context.TARGET_SECURITIES = 5
    context.TOP_ROE_QTY = 50 #First sort by ROE
 
    # This is for the trend following filter
    context.SPY = symbol('SPY')
    context.TF_LOOKBACK = 200
    context.TF_CURRENT_LOOKBACK = 20
 
    # This is for the determining momentum
    context.MOMENTUM_LOOKBACK_DAYS = 126 #Momentum lookback
    context.MOMENTUM_SKIP_DAYS = 10
    # Initialize any other variables before being used
    context.stock_weights = pd.Series()
    context.bond_weights = pd.Series()
 
    # Should probably comment out the slippage and using the default
    # set_slippage(slippage.FixedSlippage(spread = 0.0))
    # Create and attach pipeline for fetching all data
    algo.attach_pipeline(make_pipeline(context), 'pipeline')
    # Schedule functions
    # Separate the stock selection from the execution for flexibility
    schedule_function(
        select_stocks_and_set_weights,
        date_rules.month_end(days_offset = 7),
        time_rules.market_close(minutes = 30)
    )
    schedule_function(
        trade,
        date_rules.month_end(days_offset = 7),
        time_rules.market_close(minutes = 30)
    )
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())
 
def make_pipeline(context):
    universe = Q1500US()
    '''
    spy_ma50_slice = SMA(inputs=[USEquityPricing.close],
                         window_length=context.TF_CURRENT_LOOKBACK)[context.SPY]
    spy_ma200_slice = SMA(inputs=[USEquityPricing.close],
                          window_length=context.TF_LOOKBACK)[context.SPY]
    spy_ma_fast = SMA(inputs=[spy_ma50_slice], window_length=1)
    spy_ma_slow = SMA(inputs=[spy_ma200_slice], window_length=1)
    trend_up = spy_ma_fast > spy_ma_slow
     '''
    spy_ma50_slice = SMA(inputs=[USEquityPricing.close],
                         window_length=context.TF_CURRENT_LOOKBACK)[context.SPY]
    
    
    spy_ma200_slice = SMA(inputs=[USEquityPricing.close],
                          window_length=context.TF_LOOKBACK)[context.SPY]
    spy_ma_fast = SMA(inputs=[spy_ma50_slice], window_length=1)
    spy_ma_slow = SMA(inputs=[spy_ma200_slice], window_length=1)
    trend_up = spy_ma_fast > spy_ma_slow
    
    
    
    
    cash_return = ms.cash_return.latest.rank(mask=universe) #(mask=universe)
    fcf_yield = ms.fcf_yield.latest.rank(mask=universe) #(mask=universe)
    roic = ms.roic.latest.rank(mask=universe) #(mask=universe)
    ltd_to_eq = ms.long_term_debt_equity_ratio.latest.rank(mask=universe) #, mask=universe)
    
    ent_to_eb = ms.ev_to_ebitda.latest.rank(mask=universe)
    value = (cash_return + fcf_yield+ ent_to_eb).rank() #(mask=universe)
    quality = roic + ltd_to_eq + value
    # Create a 'momentum' factor. Could also have been done with a custom factor.
    returns_overall = Returns(window_length=context.MOMENTUM_LOOKBACK_DAYS+context.MOMENTUM_SKIP_DAYS)
    returns_recent = Returns(window_length=context.MOMENTUM_SKIP_DAYS)
    ### momentum = returns_overall.log1p() - returns_recent.log1p()
    momentum = returns_overall - returns_recent
    # Filters for top quality and momentum to use in our selection criteria
    top_quality = quality.top(context.TOP_ROE_QTY, mask=universe)
    top_quality_momentum = momentum.top(context.TARGET_SECURITIES, mask=top_quality)
    # Only return values we will use in our selection criteria
    pipe = Pipeline(columns={
                        'trend_up': trend_up,
                        'top_quality_momentum': top_quality_momentum,
                        },
                    screen=top_quality_momentum
                   )
    return pipe
 
def select_stocks_and_set_weights(context, data):
    """
    Select the stocks to hold based upon data fetched in pipeline.
    Then determine weight for stocks.
    Finally, set bond weight to 1-total stock weight to keep portfolio fully invested
    Sets context.stock_weights and context.bond_weights used in trade function
    """
    # Get pipeline output and select stocks
    df = algo.pipeline_output('pipeline')
    current_holdings = context.portfolio.positions
    # Define our rule to open/hold positions
    # top momentum and don't open in a downturn but, if held, then keep
    rule = 'top_quality_momentum & (trend_up or (not trend_up & index in @current_holdings))'
    stocks_to_hold = df.query(rule).index
    
    # Set desired stock weights
    # Equally weight
    stock_weight = 1.0 / context.TARGET_SECURITIES
    context.stock_weights = pd.Series(index=stocks_to_hold, data=stock_weight)
    # Set desired bond weight
    # Open bond position to fill unused portfolio balance
    # But always have at least 1 'share' of bonds
    ### bond_weight = max(1.0 - context.stock_weights.sum(), stock_weight) / len(context.BONDS)
    bond_weight = max(1.0 - context.stock_weights.sum(), 0) / len(context.BONDS)
    context.bond_weights = pd.Series(index=context.BONDS, data=bond_weight)
    #print("Stocks to buy " + str(stocks_to_hold))
    print("Stocks to buy: " + str([ str(s.symbol) for s in stocks_to_hold ]) )
    
    #print("Bonds weight " + str(bond_weight))
def trade(context, data):
    """
    Execute trades using optimize.
    Expects securities (stocks and bonds) with weights to be in context.weights
    """
    # Create a single series from our stock and bond weights
    total_weights = pd.concat([context.stock_weights, context.bond_weights])
 
    # Create a TargetWeights objective
    target_weights = opt.TargetWeights(total_weights)
 
    # Execute the order_optimal_portfolio method with above objective and any constraint
    order_optimal_portfolio(
        objective = target_weights,
        constraints = []
        )
    #Log our holdings
    log.info( [ str(s.symbol) for s in sorted(context.portfolio.positions) ] )
    #print("Cash: " + str(context.portfolio.cash))
    # Record our weights for insight into stock/bond mix and impact of trend following
    record(stocks=context.stock_weights.sum(), bonds=context.bond_weights.sum())
def record_vars(context, data):
    record(leverage = context.account.leverage)
    longs = shorts = 0
    for stock in context.portfolio.positions:
        if context.portfolio.positions[stock].amount > 0:
            longs += 1
        elif  context.portfolio.positions[stock].amount < 0:
            shorts += 1
    record(longs = longs)
    record(shorts = shorts)
