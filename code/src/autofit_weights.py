import numpy as np

"""minor modification of: https://github.com/N-Wouda/ALNS"""

def autofit(
        cls,
        init_obj: float,
        worse: float,
        accept_prob: float,
        num_iters: int,
        method: str = "exponential",
):
    """
    Returns an SA object with initial temperature such that there is a
    ``accept_prob`` chance of selecting a solution up to ``worse`` percent
    worse than the initial solution. The step parameter is then chosen such
    that the temperature reaches 1 in ``num_iters`` iterations.

    This procedure was originally proposed by Ropke and Pisinger (2006),
    and has seen some use since - i.a. Roozbeh et al. (2018).

    Parameters
    ----------
    init_obj
        The initial solution objective.
    worse
        Percentage (in (0, 1), exclusive) the candidate solution may be
        worse than initial solution for it to be accepted with probability
        ``accept_prob``.
    accept_prob
        Initial acceptance probability (in [0, 1]) for a solution at most
        ``worse`` worse than the initial solution.
    num_iters
        Number of iterations the ALNS algorithm will run.
    method
        The updating method, one of {'linear', 'exponential'}. Default
        'exponential'.

    Raises
    ------
    ValueError
        When the parameters do not meet requirements.

    Returns
    -------
    SimulatedAnnealing
        An autofitted SimulatedAnnealing acceptance criterion.

    References
    ----------
    .. [1] Ropke, Stefan, and David Pisinger. 2006. "An Adaptive Large
           Neighborhood Search Heuristic for the Pickup and Delivery
           Problem with Time Windows." *Transportation Science* 40 (4): 455
           - 472.
    .. [2] Roozbeh et al. 2018. "An Adaptive Large Neighbourhood Search for
           asset protection during escaped wildfires."
           *Computers & Operations Research* 97: 125 - 134.
    """
    if not (0 <= worse <= 1):
        raise ValueError("worse outside [0, 1] not understood.")

    if not (0 < accept_prob < 1):
        raise ValueError("accept_prob outside (0, 1) not understood.")

    if num_iters <= 0:
        raise ValueError("Non-positive num_iters not understood.")

    if method not in ["linear", "exponential"]:
        raise ValueError("Method must be one of ['linear', 'exponential']")

    start_temp = -worse * init_obj / np.log(accept_prob)

    if start_temp < 1:
        start_temp = 1.00001

    if method == "linear":
        step = (start_temp - 1) / num_iters
    else:
        step = (1 / start_temp) ** (1 / num_iters)

    return cls(start_temp, 1, step, method=method)
