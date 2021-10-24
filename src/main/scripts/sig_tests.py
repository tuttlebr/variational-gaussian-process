from logging import basicConfig, info

basicConfig(
    format="%(asctime)s %(message)s", level="INFO", datefmt="%Y-%m-%d %H:%M:%S",
)
from scipy.stats import mannwhitneyu, ttest_ind


def test_sig(data1, data2, p_threshold=0.05):
    """Mann-Whitney U Test:
    Tests whether the distributions of two independent samples are equal
    or not.

    Assumptions

    Observations in each sample are independent and identically
    distributed (iid).
    Observations in each sample can be ranked.
    Interpretation

    H0: the distributions of both samples are equal.
    H1: the distributions of both samples are not equal."""
    stat, p = mannwhitneyu(data1, data2)
    info("Mann-Whitney U Test stat=%.5f, p=%.5f" % (stat, p))
    if p > p_threshold:
        info("Probably the same distribution (fail to reject H0)")
    else:
        info("Probably different distributions (reject H0) <--")


def t_test(data1, data2, alpha=0.05):
    """Paired t-test:
    Calculate the T-test for the means of *two independent* samples of
    scores. This is a two-sided test for the null hypothesis that 2
    independent samples have identical average (expected) values. This
    test assumes that the populations have identical variances by default.

    Assumptions

    Observations in each sample are independent and identically distributed
    (iid).
    Observations in each sample are normally distributed.
    Observations in each sample have the same variance.
    Observations across each sample are paired.
    Interpretation

    H0: the means of the samples are equal.
    H1: the means of the samples are unequal."""
    stat, p = ttest_ind(data1, data2)
    info("t-test Statistics=%.5f, p=%.5f" % (stat, p))
    # interpret
    if p > alpha:
        info("Probably the same distribution (fail to reject H0)")
    else:
        info("Probably different distributions (reject H0) <--")
