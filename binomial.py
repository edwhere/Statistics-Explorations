""" Plots of a Binomial distribution using the ployly library
    The Binomial distribution describes a sequence of n trials. Each trial can result in a positive result
    with probability p and a negative result with probability 1 - p. The parameters of this distribution are p and n.

    The probability mass function is:
         Pr[k successes in n trials] = Pr[X = k] = binomial_coeff(n, k)  p^k  (1 - p)^(n-k)

    Example of using this program:
         python binomial.py -p 0.25 -n 40

    Dependencies: Python 3.7, plotly, pandas, scipy
"""

import sys
import argparse
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.stats import binom


def parse_arguments():
    """ Parses the command line and returns an object containing the desired distribution parameters (p and n)"""

    describe = "Computes the probability of k successes in n trials, where p is the probability "
    describe += "of success in any trial. "
    parser = argparse.ArgumentParser(description=describe)

    required = parser.add_argument_group("required arguments")
    required.add_argument('-p', '--probability', help="success probability", type=float, required=True)
    required.add_argument('-n', '--num_trials', help="number of trials", type=int, required=True)

    args = parser.parse_args()

    # Verify the quality of the arguments

    if args.probability > 1.0 or args.probability < 0:
        print("[Error] The probability value must belong to [0.0, ...  1.0]")
        print("        Current value is {}".format(args.probability))
        sys.exit(1)

    if args.num_trials <= 0:
        print("[Error] Parameter n (num trials) must be non-zero positive")
        print("        Current value is {}".format(args.num_trials))

    # Return arguments as a class object

    class Params:
        p = args.probability
        n = args.num_trials

    return Params


def binomial_pmf_experimental(params):
    """ Computes the PMF for a binomial distribution from 10,0000 samples produced by a random number generator
    Args:
        params: An object containing two attributes: n and p. These are the distribution parameters
    Returns:
        A pandas data frame with 2 columns: 'k' and 'prob_exp'. The first column contains integers from 0 to n
        (representing the number of successful trials). The second column contains the probabilities Pr[X=k]. The
        second column name is 'prob_exp'
    """
    num_experiments = 10000  # Defines the number of random samples
    xvals = range(0, params.n + 1)  # Defines all possible values of k (number of successes in n trials)

    # generate samples using a random number generator
    random_samples = binom.rvs(n=params.n, p=params.p, size=num_experiments)

    # create a dictionary to count how many events of X=k happened in the randomly generated samples
    zvals = [0] * len(xvals)            # list of 0's having the same size as xvals
    zipped_vals = zip(xvals, zvals)     # create a list of pairs, with each pair representing (k, 0)
    dict_vals = dict(zipped_vals)       # convert list into a dictionary with elements like  "k": 0

    for rs in random_samples:
        dict_vals[rs] += 1  # Count number of occurrences for randomly generated value

    # convert dictionary of results to a pandas data frame
    res_df = pd.DataFrame(list(dict_vals.items()), columns=['k', 'raw_count'])

    # normalize results to obtain probabilities and delete original raw counts
    res_df['prob_exp'] = res_df['raw_count'] / num_experiments
    del res_df['raw_count']

    return res_df


def binomial_pmf_theoretical(params):
    """ Computes the PMF for a binomial distribution using the explicit formula
        Args:
            params: An object containing two attributes: n and p. These are the distribution parameters
        Returns:
            A pandas data frame with 2 columns called 'k' and 'prob_thr'. The first column contains integers
            from 0 to n representing the number of successful trials. The second column contains the
            probabilities Pr[X=k].
        """
    xvals = range(0, params.n + 1)   # X is the random variable in Pr[X=k]. It represents number of successes
    probs = [binom.pmf(k=v, n=params.n, p=params.p) for v in xvals] # Generate p(k) for each possible value of k

    # Create a pandas dataframe with the results and return
    res_df = pd.DataFrame(data=list(zip(xvals, probs)), columns=["k", "prob_thr"])

    return res_df


def main(params):
    """ Orchestrates the different procedures for data collection and later displays the results
    Args:
        params: An object with two members p and n, which are the distribution parameters
    Returns:
        Nothing to return. This function displays the resulting charts in a local browser
    """
    # compute a theoretical pmf using the formula for a binomial distribution
    thr_df = binomial_pmf_theoretical(params)

    # compute an experimental pmf using a binomial random number generator
    exp_df = binomial_pmf_experimental(params)

    chart_title = "Probability Mass Function (PMF) of a Binomial Distribution with p = {} and n = {}".format(params.n,
                                                                                                             params.p)

    # merge results and plot the charts
    merged_df = pd.merge(thr_df, exp_df, on='k')
    display_2_charts(df=merged_df, xcol='k', y1col='prob_thr', y2col="prob_exp", xlabel="k",
                     y1label='Pr[X=k]', y2label='Pr[X=k]',
                     y1title="Theoretical PMF", y2title="Experimental PMF (from 10,000 samples)",
                     title=chart_title)


def display_2_charts(df, xcol, y1col, y2col, xlabel="", y1label="", y2label="", y1title="", y2title="", title=""):
    """ Displays two charts using the browser of the machine that runs the program. Uses plotly to generate
    the composition
    """
    pio.renderers.default = "browser"

    fig = make_subplots(rows=2, cols=1, subplot_titles=(y1title, y2title))

    # Top and bottom figures
    fig.add_trace(go.Bar(x=df[xcol], y=df[y1col]), row=1, col=1)
    fig.add_trace(go.Bar(x=df[xcol], y=df[y2col]), row=2, col=1)

    # Add axis information
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_xaxes(title_text=xlabel, row=2, col=1)

    fig.update_yaxes(title_text=y1label, row=1, col=1)
    fig.update_yaxes(title_text=y2label, row=2, col=1)

    # Add main title
    fig.update_layout(title_text=title, width=800, showlegend=False)

    fig.show()


if __name__ == "__main__":
    main(parse_arguments())
