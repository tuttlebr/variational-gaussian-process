import statsmodels.tsa.stattools as ts


def cointegration(id, y0, y1):
    """
    Cointegration Test
    The Null hypothesis is that there is no cointegration, the alternative
    hypothesis is that there is cointegrating relationship. If the pvalue
    is small, below a critical size, then we can reject the hypothesis that
    there is no cointegrating relationship.

    P-values and critical values are obtained through regression surface
    approximation from MacKinnon 1994 and 2010.
    """

    pvalue = ts.coint(
        y0,
        y1,
        trend="c",
        method="aeg",
        maxlag=None,
        autolag="t-stat",
        return_results=None,
    )
    return [id, pvalue[1]]
