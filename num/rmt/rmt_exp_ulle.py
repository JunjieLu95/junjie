# -*- coding: utf-8 -*-
"""
function to evaluate eigenvalues from experimental or numerical data (RMT):\n
Sigma_2, Delta_3, P(S), Weyl, ...

Created on Fri Mar 04 07:40:16 2016

@author: kuhl
"""


import sys
import argparse
import warnings
import collections

TestFlag=False#True
#TestFlag=True
import numpy as np
import matplotlib.pylab as plt

import mesopylib.extract.frq2 as extract_frq2
from mesopylib.num.rmt.rmt_th import th_sigma2, th_delta3, th_R2, th_Y2, num_Delta3,th_Ps

from mesopylib.utilities import symmetrize_values
import mesopylib.utilities.spectra as spectra

def weylGraph(k, length=None,const=None, text=False):
    """calculates the normalized k values\n
       if no Length and constant is supplied it makes a fit to obtain the total length of the graph\n
       k in 1/m; length in m; const m\n
       output should have <k_n+1-k_n>=1
    """
    if length is None:
        x=np.arange(0,len(k))+1
        #A = np.vstack([k, np.zeros(len(x))]).T
        #m_fit, c_fit = np.linalg.lstsq(A, x)[0]
        #length=m_fit*np.pi
        #const=-c_fit
        pfit=np.polyfit(x,k,1)
        length=pfit[0]*np.pi
        const=pfit[1]-pfit[0]*k[0]
        if text:
            print('Fitted Weyl law with length,constant:', length, const, pfit) #,m_fit, c_fit)
    result=length/np.pi*k+const
    return result


def __normalize_histogram_y2(hist_y, E_min, E_max, l_max):
    """
    helper function to normalize the y2 histogram correctly

    In order to obtain a normalization comparable to the
    theoretical curves for GOE, GUE, and GSE, we need to do the
    normalization differently from an ordinary histogram

    Parameters
    ----------
    hist_y : ndarray
        the histogram's y-values for Y_2
    E_min : scalar
        minimum energy of the spectrum under consideration
    E_max : scalar
        maximum energy of the spectrum under consideration
    l_max : scalar
        maximum value of l to calculate Y_2 for

    Returns
    -------
    normalized_hist_y : ndarray
        normalized values suitable to calculate Y_2 via
        Y_2 = 1 - `normalized_hist_y`.

    See Also
    --------
    mesopylib.num.rmt : The docstring has more info on this
                    (See also README.rst)
    """
    return hist_y*len(hist_y)/((E_max - E_min) * l_max)


def determine_truncation(weighting, E_max, lMax):
    """Determine the truncation in y2
    """
    def do_not_truncate(evals):
        """this is used if no truncation is done"""
        return None

    # TODO: One could use functools.partial here or a closure in order
    # to include E_min and E_max here and not have to do this explicitly
    # further down -- this would decouple the code a bit more.
    def truncation_index(evals):
        """this is used as an upper bound on the spectrum for truncation"""
        where_valid = evals < (E_max - lMax)
        if not where_valid.any():
            # all energies are too small
            first_invalid_index = 0
        else:
            first_invalid_index = np.nonzero(where_valid)[0][-1] + 1
        return first_invalid_index

    return (truncation_index
            if weighting == "truncate"
            else do_not_truncate)


def determine_weighting(weighting):
    """Determine the weighting in y2 and K
    """
    # determine what kind of weighting should be used
    if not isinstance(weighting, collections.Callable):
        if weighting is None or weighting is False:
            weighting = spectra.no_weight_occurrence_for_Y_2
        elif weighting is True:
            weighting = spectra.weight_occurrence_for_Y_2
        elif weighting == "truncate":
            weighting = spectra.no_weight_occurrence_for_Y_2
        else:
            raise ValueError("Unsupported value for weighting: %s" % (
                str(weighting)))
    return weighting


def calc_y2(ef, lAxis=None, lMax=10, nLAxis=None,
            plotFlag=False, text=False, save=False, overwrite=False,
            histFlag=False, hx=None, hy=None, Norm=True,
            NBins=100, BinSize=0.2, Minimum=0, Maximum=None,
            nDim=1, xscale=1.0,
            weighting=spectra.weight_occurrence_for_Y_2):
    """calculates Y2

    Calculates Y2 from stick spectrum which is already normalized (Weyl
    axis) the Y2 two point cxorrelation function.

    Note: As this is meant to be used with experimental data. In these,
    missing values are marked with `NaN`. These values are discarded
    here before the calculation is done.

    Parameters
    ----------
    ef : list of ndarrays
        Several input spectra to use for the calculation. Note that all
        NaN values are filtered out of these arrays in order to allow to
        experimental data have gaps in the determined spectra.
    weighting : function or string, optional
        Take a weighting for energies distances L > 0 into account to
        compensate the finiteness of the given spectrum. Defaults to
        utilities.spectra.weight_occruence_for_Y_2 Note: Due to the
        finite size of the sample of spectrum, larger distances are
        under-represented compared to an infinitely large spectrum. This
        is compensated here by scaling the count of longer distances
        with a weighting factor larger than one. See also the doc string
        of `mesopylib.num.rmt`.
        If you want to avoid this, have a look at
        no_weight_occrruence_for_Y_2.
        If the value is not a function, then these values are allowed:
          None, False : No weighting is done, corresponds to
                        no_weight_occrruence_for_Y_2.
          True : Corresponds to weight_occrruence_for_Y_2 (default)
          "truncate" : This will use `no_weight_occrruence_for_Y_2` AND
                       only take energies into account with
                       E < E_max - lmax in order to get the same
                       statistics on all `L` ranges.

    See Also
    --------
    tests/test_R2_with_missing_levels.py: Uses this flag
    """
    if nLAxis is not None:
        warnings.warn("You are using 'nLAxis' in calc_y2 but it is not used!" +
                      " Consider `NBins` instead")
    if Maximum is not None:
        warnings.warn("You are using 'Maximum' in calc_y2 but it is not used!" +
                      " Consider using `lMax` instead.")

    dShape= np.shape(ef)
    #dimension=len(dShape) # dimensionality of ef
    nEigenValues= int(dShape[-1]) #how many eigenvalues maximal
    if type(ef)==np.ndarray:
        ef=[ef]
    nDim=len(ef)
    #result= [[0.0]]
    result= []

    # calculate extend of spectrum (necessary for the weighting later)
    E_min = np.min([e.min() for e in ef])
    E_max = np.max([e.max() for e in ef])

    truncate_func = determine_truncation(weighting, E_max, lMax)
    weighting = determine_weighting(weighting)

    if text:
        print('nEigenValues= ', nEigenValues, ' nDim= ', nDim)
    for iDim in np.arange(nDim) :
        iData=ef[iDim]
        w = np.where(np.logical_not(np.isnan(iData)))[0]
        wCount = len(w)
        if wCount > 0:
            if text:
                print('Calculating Y2 in Dimension: ', iDim+1,' of ',nDim)
            iData= iData[w]

            # determine which values to take into account (Note: If no
            # truncation is done, then this is a no-op function
            # returning None which corresponds to taking the whole
            # array)
            trunc_idx = truncate_func(iData)

            for iDist in np.arange(1, len(iData)):
                DistData= iData[iDist:] - iData[0:-iDist]

                # only take energies into account which are smaller
                # than E_max - lMax, see docstring
                DistData = DistData[:trunc_idx]

                w= np.where( DistData <= lMax)[0]
                wCount = len(w)
                if wCount > 0:
                    result.append(DistData[w])
                else:
                    break #final value reached # iDist= len(iData)

    result = np.concatenate(result if len(result) > 0 else [[]])

    result= np.sort(result, axis=None) # sort flattend array
    count= len( result)
    if text:
        print('Distances= ', count, ' MaxDist= ', result[count-1])
    if histFlag or plotFlag :
        #hy= Make_H( Result, hx, NBins=NBins, BinSize=Binsize, Minimum=Minimum, Maximum=Maximum )
        weights = weighting(result, E_min, E_max)
        if np.isnan(weights).any():
            raise ValueError(
                "Weighting produced NaN values in y2 histogram!")
        if np.isinf(weights).any():
            new_max = (100 *
                       weights[np.logical_not(np.isinf(weights))].max())
            warnings.warn(
                "Setting Infinitiy values in weights to 100 x maximum: %s" % (
                    new_max))
            weights[np.isinf(weights)] = new_max

        # Calculate Y2 as a histogram
        # Note: we don't do any normalization here as we don't want the
        # histogram's area to be normalized but rather its behavior
        # should match the one from the theoretical curves on GOE, GUE,
        # and GSE
        hy, hx = np.histogram(result, bins=NBins,
                              range=[Minimum, lMax],
                              weights=weights)

        # In order to obtain a normalization comparable to the
        # theoretical curves for GOE, GUE, and GSE, we need to do the
        # normalization differently from an ordinary histogram
        hy = __normalize_histogram_y2(
            hy, E_min=E_min, E_max=E_max, l_max=lMax)

        # Also: Y2 is defined as decaying from 1
        hy = 1 - hy

        if plotFlag:
            plt.figure('exp. $Y_2$');plt.clf()
            plt.plot(hx[0:-1]*xscale, hy, label="Data",marker='x');
            plt.ylabel('$Y_2(L)$',size=16); plt.xlabel('$L$',size=16)
            tx=np.arange(min(hx),max(hx),0.01)
            plt.plot(tx, th_Y2(tx, Poisson=True), label="Poisson", linestyle='--')
            plt.plot(tx, th_Y2(tx, GOE=True), label="GOE", linestyle='-')
            plt.plot(tx, th_Y2(tx, GUE=True), label="GUE", linestyle='-')
            plt.plot(tx, th_Y2(tx, GSE=True), label="GSE", linestyle=':')
            plt.legend()
        return result, hx, hy
    return result


def calc_spacing(evals, NBins=200, plotFlag=False, **kwargs):
    """
    Calculate P(s) using np.histogram

    All parameters are passed to np.histogram.
    """
    # comply to the naming in histogram -- if there is a parameter
    # called `bins`, then we ignore NBins
    kwargs["bins"] = kwargs.get("bins", NBins)
    hist, edges = np.histogram(
        np.diff(evals.real), **kwargs)
    centers = edges[:-1] + edges.ptp() / (len(edges) - 1)
    if plotFlag:
        plt.figure('P(s)');plt.clf()
        plt.plot(edges[0:-1], hist)
        tx = np.arange(min(edges),max(edges),0.01)
        plt.plot(tx, th_Ps(tx, Poisson=True), label="Poisson", linestyle='--')
        plt.plot(tx, th_Ps(tx, GOE=True), label="GOE", linestyle='-')
        plt.plot(tx, th_Ps(tx, GUE=True), label="GUE", linestyle='-')
        plt.plot(tx, th_Ps(tx, GSE=True), label="GSE", linestyle=':')
        plt.legend()

    return centers, hist


def calc_r2(ef, lAxis=None, lMax=10,nLAxis=150, plotFlag=False, text=False, save=False, overwrite=False,
            histFlag=False, Norm=True,NBins=100, BinSize=0.2, Minimum=0, Maximum=None, nDim=1,
            weighting=spectra.weight_occurrence_for_Y_2):
    """ calculates from stick spectrum which is already normalized (Weyl axis) the R2 two point cxorrelation function
    For the list of parameters see `calc_y2`.
    """

    if plotFlag:
        histFlag=True
    result = calc_y2(ef, lAxis=lAxis, lMax=lMax,nLAxis=nLAxis, text=text,
                     histFlag=histFlag, Norm=Norm, NBins=NBins, BinSize=BinSize,
                     Minimum=Minimum, Maximum=Maximum, nDim=nDim,
                     weighting=weighting)
    if histFlag or plotFlag :
        result,hx,hy=result
        hy= 1 - hy
        if plotFlag:
            plt.figure('exp. $R_2$');plt.clf()
            plt.plot(hx[:-1], hy, label="Data",marker='x');
            plt.ylabel('$R_2(L)$',size=16); plt.xlabel('$L$',size=16)
            tx=np.arange(min(hx),max(hx),0.01)
            plt.plot(tx, th_R2(tx, Poisson=True), label="Poisson", linestyle='--')
            plt.plot(tx, th_R2(tx, GOE=True), label="GOE", linestyle='-')
            plt.plot(tx, th_R2(tx, GUE=True), label="GUE", linestyle='-')
            plt.plot(tx, th_R2(tx, GSE=True), label="GSE", linestyle=':')
            #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.legend()
        return 1-result, hx, hy
    return 1-result


def determine_smaller_values(E, ef):
    """
    Returns the number of values from `ef` smaller or equal to E

    For each value of `E`, E_i, determine the number of elements in
    `ef` which fulfills (E_i < ef[j:]).all()

    If E_i is smaller than `ef`.max() and `ef` is sorted, then this is
    also the index from which on `ef` is larger than the value in `E`.

    Parameters
    ----------
    E : ndarray of length N
        The values to check against the eigenvalues
    ef : ndarray of length M
        The eigenvalues to use

    Returns
    -------
    indices : ndarray, length N
        For each E_i the number of values in ef being smaller than E_i.

    Examples
    --------
    # Note that the value 6 is larger than the maximum in `ef`,
    # therefore the corresponding value is the length of `ef`, 4.0 in
    # this example.
    >>> determine_smaller_values(
    ...     [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 6.0],
    ...     [0, 1, 2, 3, 4])
    [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0]

    """
    return (ef[:, np.newaxis] <= E[np.newaxis, :]).sum(axis=0)


def _sigma2_SingelSpectrum(ef, lAxis):
    """use not directly use sigma2!!!
    Calculates the sigma2 from a single stick spectrum

    Note: Works for eigenvalues already properly normalized to constant
    density of 1

    ef : normalized eigenvalues
    returns sigma2
    sigma2 = sigma2()
    """
    nLAxis = len(lAxis)
    deltaL = (lAxis.ptp())/(nLAxis + 1)
    scale = 1 / deltaL
    # calculate the mean slope of the spectrum and use it as estimate on
    # the occuring steps in energy

    # In order to approximate the integral necessary for Sigma_2
    # correctly, we approximate the necessary E-interval to evaluate the
    # integral on, see comment below on the integration (*)
    maxdE = np.round(np.max(ef) - np.min(ef))*scale
    E = np.round(np.arange(maxdE) + scale*ef[0]) / scale

    # This is the integrated density of states up to the energies in E.
    # The above E range was choosen such that for a given value of L_i
    # the range of n values, n[i:] and n[:len(n) - i], correspond to the
    # integral over dE n(E)^2 when squared and summed up afterwards.
    n = determine_smaller_values(E, ef)
    result =  np.zeros(nLAxis, dtype=float)
    for i in range(nLAxis):
        # calculate the number of eigenvalues relevant for the current
        # value of L: Note, we want to average over [E-L/2, E+L/2]

        # Therefore, we take the count of all states with energies E
        # with E_i < E < E_max and substract from these number the count
        # of all energies E with E_min < E < E_(N-i)
        # This is element-wise exactly the calculation of the integral
        # over the density: n is the integral of the density and taking
        # n at the end of the integration range minus at the beginning
        # is exactly the fundamental theorem of calculus. We therefore
        # had to choose the E axis appropriately above (*)
        dn = n[i:len(E)] - n[:len(E) - i]
        # FIXME: Do here a weighting as for the distances in Y_2? Note:
        # There we weighted the distances (L = E_j - E_k) irrespectively
        # on the concrete value of the terms in the difference (E_j and
        # E_k). Here, however, we would have to weight energy levels
        # directly, not only their difference, I guess.
        qn = np.sum(dn ** 2, dtype=np.int64) / (len(E) - i)
        # 64-bit integer must be given explicitly
        # sum interprets its result the same type type the input is
        # old: 32-bit may result in overflow for high nLAxis
        L = np.sum(dn) / (len(E) - i)
        result[i] = qn - L**2

    return result


def sigma2(ef, lAxis=None, lMax=8, nLAxis=151, plotFlag=False, text=False, averageFlag=True, **kwargs):
    """Calculates the number variance sigma2 from eigenvalues already properly normaliezed to constant density of 1
       ef : normalized eignvalues
       lAxis: Output axis
       returns LAxis,sigma2
       LAxisOut,sigma2 = sigma2(ef)
       if averageFlag is set False the sigma2 for each spectrum is returned!
       default averageFlag=True
    """
    if lAxis is None:
        lAxis=np.linspace(0,lMax,num=nLAxis)
    else:
        lMax= max(lAxis)
    if type(ef) == np.ndarray:
        ef=ef[0]
    nDim=len(ef)
    result=[]
    for i in np.arange(nDim):
        tmp_sigma= _sigma2_SingelSpectrum(ef[i],lAxis)
        result.append(tmp_sigma)
    result=np.array(result)
    if averageFlag:
        result=np.average(result, axis=0)

    if plotFlag:
        plt.figure('exp. $\Sigma_2$')
        # plt.clf()
        plt.plot(lAxis, np.transpose(result)); plt.ylabel('$\Sigma_2$',size=16); plt.xlabel('$L$',size=16)
        tx=np.arange(min(lAxis),max(lAxis),0.01)
        plt.plot(tx, th_sigma2(tx, GOE=True), label="GOE", linestyle='-')
        plt.plot(tx, th_sigma2(tx, GUE=True), label="GUE", linestyle='--')
        plt.plot(tx, th_sigma2(tx, GSE=True), label="GSE", linestyle=':')
        plt.plot(tx, th_sigma2(tx, Poisson=True), label="Poisson", linestyle='-.')
        # plt.legend()
    return lAxis, result

def delta3(ef, lAxis=None, lMax=10, nLAxis=100, plotFlag=False, text=False, Sigma2=None):
    """Calculates the spectral rigidity Delta_3 from eigenvalues already properly normaliezed to constant density of 1
       ef : normalized eignvalues
       lAxis: Output axis
       returns LAxis,delta3
       LAxisOut,dela3 = delta3()
       uses sigma2 for he calculation if not directly sigma2 already given!
    """
    if Sigma2 is None : #or len(hSigma2) != len(LAxis)
        lAxis,Sigma2= sigma2( ef, lAxis=lAxis, lMax=lMax, nLAxis=nLAxis, text=text)
    lMax= max(lAxis)
    nLAxis= np.size(lAxis)
    result=np.zeros(nLAxis,dtype=float)
    dL= 1.*lMax/nLAxis
    iMin=0
    if lAxis[0] == 0:
        iMin=1
    #import pdb; pdb.set_trace()
    for i in np.arange(iMin, nLAxis):
        E = lAxis[iMin:i+1]
        L = lAxis[i]
        result[i]= 2 / L**4 * dL * np.sum(
            # this calculates the integral from 0 to L_i, therefore the
            # sum only runs from [iMin to i]
            (L**3 - 2 * L**2 * E + E**3)
            * Sigma2[iMin:i+1])
    if plotFlag:
        plt.figure('exp. $\Delta_3$');plt.clf()
        plt.plot(lAxis, result); plt.ylabel('$\Delta_3$',size=16); plt.xlabel('$L$',size=16)
        tx=np.arange(min(lAxis),max(lAxis),0.01)
        plt.plot(tx, th_delta3(tx, GOE=True), label="GOE", linestyle='-')
        plt.plot(tx, num_Delta3(tx, GUE=True), label="GUE", linestyle='--')
        plt.plot(tx, num_Delta3(tx, GSE=True), label="GSE", linestyle=':')
        plt.plot(tx, num_Delta3(tx, Poisson=True), label="Poisson", linestyle='-.')
        plt.plot(tx, th_delta3(tx, GSE=True), label="GSE", linestyle=':')
        plt.legend()
    return lAxis, result


def deltafunction_approx(tau, N_spectrum, hbar=1, cut_off=0.5):
    """Returns an approximation of the delta-peak of K(t)

    Based on the assumption that the average level density is given by
    :math:`\rho(E) = \sum_{n=1}^N \delta(t - n)`
    we can use the approximation given by

    .. math::

       \overline{\rho}^2\cdot\delta(t) =
       \frac{1}{N}\left(\frac{sin(N t / 2)}{sin(t/2)}\right)^2.

    Use multiple paragraphs if necessary

    Parameters
    ----------
    tau : ndarray
        The times array to calculate this function at
    N_spectrum : scalar, integer
        Size of the spectrum. Necessary for the approximation.
    hbar : float, optional
        Value of hbar to use
    cut_off : float, optional
        The cut-off in tau to use. As the approximation is periodic
        while the delta contribution is not, we usually don't want this
        approximation for all values of tau.
        You can override this by any value. Its default value
        corresponds to half the Heisenberg time :math:`T_H = hbar /
        <s>`. In order to obtain an independent parameter, the value
        cut_off * hbar is used when comparing against tau.

    See Also
    --------
    mesopylib.num.rmt : the module description has details on this

    Returns
    -------
    delta : ndarray
        Values of the approximation

    """
    result = np.zeros_like(tau)
    small_tau_idx = tau <= (
        cut_off * hbar if cut_off is not None else np.inf)
    small_tau = tau[small_tau_idx]
    result[small_tau_idx] = (
        np.sin(np.pi * small_tau / hbar * N_spectrum) /
        np.sin(np.pi * small_tau / hbar))**2 / N_spectrum

    return result


def calc_K(spectrum, tau, hbar=1, remove_delta=False):
    """
    Calculate the spectral form factor K(tau).

    This is based on the spectrum as follows

    .. math::
       K(t) = \int_{-\inf}^\inf \text{d}E
              C(E)\text{e}^{-\frac{\text{i}{\hbar}E t}}
            = \frac{1}{N}\sum_{m, n=1}^{N}
              \text{e}^{-\frac{\text{i}}{\hbar} (E_n - E_m) t} +
              \overline{\rho}^2\cdot\delta(t)
            = \frac{2}{N}\sum_{m < n}
              \cos(-\frac{1}{\hbar} (E_n - E_m) t) +
              \overline{\rho}^2\cdot\delta(t)
            = \frac{1}{N} \left|\sum_{m=1}^{N}
              \text{e}^{-\frac{\text{i}}{\hbar} (E_n) t}\right|^2 +
              \overline{\rho}^2\cdot\delta(t)
   \end{array}

    with the autocorrelation function :math:`C(E)`

    The removal of the delta peak can be handled using a picket-fence
    approximation for small values of tau, see `mesopylib.num.rmt`'s
    docstring.
    """
    # FIXME: Why do we have to divide by N, not N*(N-1)/2 or N**2?
    norm = len(spectrum)

    result = np.abs(np.exp(
        -2j*np.pi/hbar *
        spectrum[:, np.newaxis] *
        tau[np.newaxis, :]).sum(axis=0))**2  / norm

    if remove_delta:
        delta = deltafunction_approx(tau, len(spectrum), hbar)
        result += delta

    return result


def calc_b2(spectrum, tau, hbar=1,
            weighting=spectra.weight_occurrence_for_Y_2, **kwargs):
    """Calculate 1 - the spectral form factor.

    This is based on the Fourier transform of Y_2 according to:

    .. math::

       b_2(t) = 1 - K(t) \\
              = \int_{-\inf}^\inf \text{d}E
                Y_2(E)\text{e}^{-\frac{\text{i}{\hbar}E t}}

    wherein :math:`K(t)` is the spectral form factor.

    Parameters
    ----------
    spectrum: ndarray
        Spectrum to calculate b2 for. Must be sorted and
        fulfill `spectrum.mean() == 1`.
    tau : ndarray
        Time array to calculate b2 on
    hbar : float, optional
        The scaling constant hbar to use in the above formula
    weighting : function, optional
        Used to weight the distances in the spectrum to account for
        finite-size effects. See `calc_y2`.

    Returns
    -------
    b2 : ndarray
        Calculated values

    See Also
    --------
    calc_K : calculates the Spectral form factor
    th_b : Theoretical predictions in `rmt.rmt_th`.
    calc_y2 : The weighting is applied there using the given function

    Examples
    --------
    >>> calc_b2(np.arange(-10, 11), np.linspace(0, 2, 50))

    """
    return tau, 1 - calc_K(
        spectrum, tau, hbar, weighting, **kwargs)


#------------------------------------------------------------
if __name__ == "__main__":

    #--------------------------------------------------------
    # parse the command line to check which tests the user
    # wants
    parser = argparse.ArgumentParser(
        description="Execute some tests on the RMT data",
        add_help=True)

    parser.add_argument(
        '--data-path',
        default=r'c:\\e\data\tog\GSE-Graphs\eigenfreq\\',
        type=str,
        help="Specify where to read the test data from")

    parser.add_argument(
        '--execute-test', action="store_true", default=False,
        help="If this flag is set, then the tests are executed")

    parser.add_argument(
        '--show', action="store_true", default=False,
        help="Call plt.show() at the end of the script.")

    parser.add_argument(
        '--force-test', action="store_true", default=False,
        help="Override script-local TestFlag with 'True'")

    args = parser.parse_args()

    #--------------------------------------------------------
    if not TestFlag and not args.force_test:
        # Just skipping the test (In order not to break how the script
        # was working before)
        sys.exit(0)
    # FIXME: We might get rif of the test flag and use the commandline
    # flag instead -- this would allow us to trigger the tests from the
    # outside.
    #elif not args.execute_test:
    #    print("Not performing any tests as --execute-test not given")
    #    sys.exit(0)
    else:
        print("Exectuing tests for directory %s" % (args.data_path))

    path=args.data_path

    ids=[4, 5, 6, 7, 8, 9, 10, 11]
    name=[r'eigenfreq_%02d_S22_pi_adjust.csv' % (iid) for iid in ids]
    Length=[3.27, 3.175009, 3.208363, 3.168025, 3.149142, 3.277373, 3.128410, 3.191570]
    ef=[]
    for index in np.arange(len(ids)):
        fname=path+name[index]
        tmp_ef=np.genfromtxt(fname,delimiter='\n',dtype=float)
        #ef_index=np.array([ief[0] for ief in tmp_ef])
        ef_index=np.array(tmp_ef)
        ef.append(ef_index)
    #n=np.arange(len(ef))+1
    #plt.figure('ef'); plt.clf(); plt.plot(n,ef); plt.ylabel('frequency/GHz'); plt.xlabel('$n$')
    k=[extract_frq2.frq2k(tmp_ef, inGHz=True) for tmp_ef in ef]
    #plt.figure('k'); plt.clf(); plt.plot(n,k); plt.ylabel('$k$/m$**{-1}$'); plt.xlabel('$n$')
    #kn=weylGraph(k, length=None,const=None, text=True)
    kn=[weylGraph(k[index], length=Length[index],const=0) for index in np.arange(len(ids))]
    knexp=[k[index]/(max(k[index])-min(k[index]))*len(k[index]) for index in np.arange(len(ids))]
    #kn_fit=weylGraph(k, length=None,const=None, text=True)
    #plt.figure('norm $k$'); plt.clf(); plt.plot(n,kn); plt.ylabel('normalized $k$'); plt.xlabel('$n$')
    #plt.figure('norm k_fit'); plt.clf(); plt.plot(n,kn-n-kn[0]); plt.ylabel('normalized $k_{fit}$'); plt.xlabel('$n$')
    l=np.arange(0,10,.01)
    r2=calc_r2(kn,lAxis=l,plotFlag=True,NBins=30)
    r2exp=calc_r2(knexp,lAxis=l,plotFlag=True,NBins=30)

    l2,s2=sigma2(kn)
    l2exp,s2exp=sigma2(knexp)
    plt.figure('sigma2'); plt.clf(); plt.plot(l2,s2,label='exp'); plt.ylabel('$\Sigma_2$',size=16); plt.xlabel('$l$',size=16)
    plt.plot(l2exp,s2exp,label='exp_weyl')
    plt.plot(l,th_sigma2(l,beta=1),ls='-',label='GOE');
    plt.plot(l,th_sigma2(l,beta=2),ls='--',label='GUE');
    plt.plot(l,th_sigma2(l,beta=4),ls=':',label='GSE')
    plt.legend()
    l3,d3=delta3(kn)
    l3exp,d3exp=delta3(knexp)
    db1a=th_delta3(l,beta=1,numFlag=True)
    db2a=th_delta3(l,beta=2,numFlag=True)
    db4a=th_delta3(l,beta=4,numFlag=True)
    plt.figure('Delta3'); plt.clf(); plt.plot(l3,d3,label='exp_weyl'); plt.ylabel('$\Delta_3$',size=16); plt.xlabel('$L$',size=16)
    plt.plot(l3exp,d3exp,ls=':',label='exp');
    plt.plot(l,db1a,ls='--',label='GOE_int');plt.plot(l,db2a,ls='--',label='GUE_int');plt.plot(l,db4a,ls='--',label='GSE_int')
    plt.legend()

    if args.show:
        plt.show()
