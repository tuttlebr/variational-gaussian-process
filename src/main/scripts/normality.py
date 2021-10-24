from logging import basicConfig, info

basicConfig(
    format="%(asctime)s %(message)s",
    level="INFO",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import matplotlib.pyplot as plt
from scipy.stats import anderson, normaltest, shapiro


def test_normality(data, alpha=0.1):

    _ = plt.hist(data, bins="auto")
    plt.title("Histogram with 'auto' bins")
    plt.show()

    info(u"\u03B1 = {}".format(alpha))

    """
    The Shapiro-Wilk test evaluates a data sample and quantifies how likely it
    is that the data was drawn from a Gaussian distribution, named for Samuel
    Shapiro and Martin Wilk. In practice, the Shapiro-Wilk test is believed to
    be a reliable test of normality, although there is some suggestion that the
    test may be suitable for smaller samples of data, e.g. thousands of
    observations or fewer. The shapiro() SciPy function will calculate the
    Shapiro-Wilk on a given dataset. The function returns both the W-statistic
    calculated by the test and the p-value.
    """

    (stat, p) = shapiro(data)
    info("Statistics={:.3%}, p={:.3%}".format(stat, p))
    if p > alpha:
        info("Sample looks Gaussian (Shapiro-Wilk Test)")
    else:
        info("Sample does not look Gaussian (Shapiro-Wilk Test)")

    """
    The D'Agostino's K^2 test calculates summary statistics from the data,
    namely kurtosis and skewness, to determine if the data distribution departs
    from the normal distribution, named for Ralph D'Agostino. Skew is a
    quantification of how much a distribution is pushed left or right, a measure
    of asymmetry in the distribution. Kurtosis quantifies how much of the
    distribution is in the tail. It is a simple and commonly used statistical
    test for normality. The D'Agostino's K^2 test is available via the
    normaltest() SciPy function and returns the test statistic and the p-value.
    """
    (stat, p) = normaltest(data)
    info("Statistics={:.3%}, p={:.3%}".format(stat, p))
    if p > alpha:
        info("Sample looks Gaussian (D'Agostino's K^2 Test)")
    else:
        info("Sample does not look Gaussian (D'Agostino's K^2 Test)")

    """
    Anderson-Darling Test is a statistical test that can be used to evaluate
    whether a data sample comes from one of among many known data samples, named
    for Theodore Anderson and Donald Darling. It can be used to check whether a
    data sample is normal. The test is a modified version of a more
    sophisticated nonparametric goodness-of-fit statistical test called the
    Kolmogorov-Smirnov test. A feature of the Anderson-Darling test is that it
    returns a list of critical values rather than a single p-value. This can
    provide the basis for a more thorough interpretation of the result. The
    anderson() SciPy function implements the Anderson-Darling test. It takes as
    parameters the data sample and the name of the distribution to test it
    against. By default, the test will check against the Gaussian distribution
    (dist='norm").
    """
    result = anderson(data)
    info("Statistics={:.3%}".format(result.statistic))
    for i in range(len(result.critical_values)):
        (sl, cv) = (result.significance_level[i], result.critical_values[i])
    if result.statistic < result.critical_values[i]:
        info(
            "significance_level %.3f, critical_values %.3f, data looks normal (Anderson-Darling Test)"
            % (sl, cv)
        )
    else:
        info(
            "significance_level %.3f, critical_values %.3f, data does not look normal (Anderson-Darling Test)"
            % (sl, cv)
        )
