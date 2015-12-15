from __future__ import division, print_function

from collections import namedtuple

import numpy as np
from scipy.special import erf, erfc

import emceemr
from astropy import units as u

MINF = -np.inf


class RGBModel(emceemr.Model):
    param_names = 'tipmag, alphargb, alphaother, fracother'.split(', ')

    AutoFuncmags = namedtuple('AutoFuncmags', ['startat', 'endat', 'uncspacing'])



    def __init__(self, magdata, magunc=None, priors=None,
                       uncfunc=None, biasfunc=None, complfunc=None,
                       funcmags=None):

        self.magdata = np.array(magdata)

        self.maxdata = np.max(magdata)
        self.mindata = np.min(magdata)

        self._magunc = magunc
        if isinstance(funcmags, self.AutoFuncmags):
            self._funcmags = self._auto_funcmags(uncfunc, *funcmags)
        else:
            self._funcmags = funcmags
        self._uncfunc = uncfunc
        self._biasfunc = biasfunc
        self._complfunc = complfunc
        self._validate_lnprob_func()

        super(RGBModel, self).__init__(priors)

    def _auto_funcmags(self, uncfunc, startat, endat, uncspacing):
        fmags = [startat]
        while fmags[-1] < endat:
            dmag = abs(uncfunc(fmags[-1])*uncspacing)
            if dmag==0:
                raise ValueError('auto_funcmags got stuck at an unc of 0. Might'
                                 ' your uncfunc be non-positive somewhere?')
            fmags.append(fmags[-1]+dmag)
            if fmags[-1] > endat:
                fmags[-1] = endat
        return np.array(fmags)


    @property
    def sorted_magdata(self):
        if self._sorted_magdata is None:
            self._sorted_magdata = np.sort(self.magdata)
        return self._sorted_magdata


    @property
    def magdata(self):
        return self._magdata
    @magdata.setter
    def magdata(self, value):
        self._sorted_magdata = None
        self._magdata = value


    def _validate_lnprob_func(self):
        """
        Checks that the various ways of giving uncertainties or not make sense
        """
        if self.magunc is None:
            if self.uncfunc is not None:
                self._lnprob_func = self._lnprob_uncfuncs
            else:
                self._lnprob_func = self._lnprob_no_unc
        elif self.funcmags is not None:
            raise ValueError('Cannot give both uncertainties and the various uncfuncs')
        else:
            self._lnprob_func = self._lnprob_w_unc

    # need to do getstate/setstate b/c _lnprob_func can't be pickled as a method
    def __getstate__(self):
        state = self.__dict__.copy()
        if state['_lnprob_func'] is not None:
            state['_lnprob_func'] = self._lnprob_func.im_func.__name__
        return state

    def __setstate__(self, state):
        meth = getattr(self, state['_lnprob_func'])
        state['_lnprob_func'] = meth
        self.__dict__ = state

    def lnprob(self, tipmag, alphargb, alphaother, fracother):
        """
        This does *not* sum up the lnprobs - that goes in __call__.  Instead it
        gives the lnprob per data point
        """
        return self._lnprob_func(self.magdata, tipmag, alphargb, alphaother, fracother)


    def _lnprob_no_unc(self, magdata, tipmag, alphargb, alphaother, fracother):
        dmags = magdata - tipmag
        rgbmsk = dmags > 0
        lnpall = np.zeros_like(dmags)

        lnpall[rgbmsk] = alphargb * dmags[rgbmsk]
        lnpall[~rgbmsk] = alphaother * dmags[~rgbmsk] + np.log(fracother)

        eterm1 = 1 - np.exp(alphaother*(self.mindata - tipmag))
        eterm2 = np.exp(alphargb*(self.maxdata - tipmag)) - 1
        lnN = np.log(fracother * eterm1 / alphaother + eterm2 / alphargb)

        return lnpall - lnN

    def _lnprob_w_unc(self, magdata, tipmag, alphargb, alphaother, fracother):
        dmag_upper = self.maxdata - tipmag
        dmag_lower = self.mindata - tipmag
        return np.log(self._exp_gauss_conv_normed(magdata - tipmag,
                                                  alphargb, alphaother,
                                                  fracother, self.magunc,
                                                  dmag_lower, dmag_upper))

    def _lnprob_uncfuncs(self, magdata, tipmag, alphargb, alphaother, fracother, _normalizationint=False):
        funcmags = self._funcmags.reshape(1, self._funcmags.size)

        if self._uncfunc is None:
            raise ValueError('Funcmags given but uncfunc is None')
        elif callable(self._uncfunc):
            uncs = self._uncfunc(funcmags)
        else:
            uncs = self._uncfunc

        if self._biasfunc is None:
            biasedmags = funcmags
        elif callable(self._biasfunc):
            biasedmags = self._biasfunc(funcmags)
        else:
            biasedmags = self._biasfunc.reshape(1, funcmags.size)

        if self._complfunc is None:
            compl = 1
        elif callable(self._complfunc):
            compl = self._complfunc(funcmags)
        else:
            compl = self._complfunc.reshape(1, funcmags.size)

        magdata_reshaped = magdata.reshape(magdata.size, 1)

        lf = self._lnprob_no_unc(biasedmags, tipmag, alphargb, alphaother, fracother)
        uncterm = (2*np.pi)**-0.5 * np.exp(-0.5*((magdata_reshaped - biasedmags)/uncs)**2)/uncs
        dataintegrand = compl*uncterm*np.exp(lf)

        Idata = np.trapz(y=dataintegrand, x=funcmags, axis=-1)

        if _normalizationint:
            return Idata
        else:
            intN =  self._lnprob_uncfuncs(self.sorted_magdata,tipmag,
                                          alphargb, alphaother, fracother,
                                          _normalizationint=True)
            N = np.trapz(y=intN, x=self.sorted_magdata)
            self.normed = intN,funcmags.ravel(), N
            return np.log(Idata) - np.log(N)


    def plot_lnprob(self, tipmag, alphargb, alphaother, fracother, magrng=100, doplot=True, delog=False, **plotkwargs):
        """
        Plots (optionally) and returns arrays suitable for plotting the pdf. If
        `magrng` is a scalar, it gives the number of samples over the data
        domain.  If an array, it's used as the x axis.
        """
        from copy import copy
        from astropy.utils import isiterable
        from matplotlib import pyplot as plt

        fakemod = copy(self)
        if isiterable(magrng):
            fakemod.magdata = np.sort(magrng)
        else:
            fakemod.magdata = np.linspace(self.mindata, self.maxdata, magrng)

        if fakemod.magunc is not None:
            sorti = np.argsort(self.magdata)
            fakemod.magunc = np.interp(fakemod.magdata, self.magdata[sorti], self.magunc[sorti])

        lnpb = fakemod.lnprob(tipmag, alphargb, alphaother, fracother)
        if delog:
            lnpb = np.exp(lnpb - np.min(lnpb))

        if doplot:
            plt.plot(fakemod.magdata, lnpb, **plotkwargs)

        return fakemod.magdata, lnpb

    def plot_data_and_model(self, samplerorparams, perc=50, datakwargs={}, lfkwargs={}):
        from astropy.utils import isiterable
        from matplotlib import pyplot as plt

        if isiterable(samplerorparams):
            ps = samplerorparams
        else:
            sampler = samplerorparams
            ps = np.percentile(sampler.flatchain, perc, axis=0)
        self.plot_lnprob(*ps, **lfkwargs)

        n, edges = np.histogram(self.magdata, bins=datakwargs.pop('bins', 100))
        cens = (edges[1:]+edges[:-1])/2
        N = np.trapz(x=cens, y=n)
        plt.scatter(cens, np.log(n/N), **datakwargs)

        plt.ylabel('log(lf/data)')



    @staticmethod
    def _exp_gauss_conv_normed(x, a, b, F, s, x_lower, x_upper):
        # from scipy.integrate import quad
        # N = quad(exp_gauss_conv, x_lower, x_upper, args=(a, b, F, np.mean(s)))[0]
        # return exp_gauss_conv(x, a, b, F, s)/N
        norm_term_a = RGBModel._exp_gauss_conv_int(x_upper, a, s, g=1) - RGBModel._exp_gauss_conv_int(x_lower, a, s, g=1)
        norm_term_b = RGBModel._exp_gauss_conv_int(x_upper, b, s, g=-1) - RGBModel._exp_gauss_conv_int(x_lower, b, s, g=-1)
        return RGBModel._exp_gauss_conv(x, a, b, F, s)/(norm_term_a + F * norm_term_b)

    @staticmethod
    def _exp_gauss_conv(x, a, b, F, s):
        """
        Convolution of broken power law w/ gaussian.
        """
        A = np.exp(a*x+a**2*s**2/2.)
        B = np.exp(b*x+b**2*s**2/2.)
        ua = (x+a*s**2)*2**-0.5/s
        ub = (x+b*s**2)*2**-0.5/s
        return (A*(1+erf(ua))+F*B*erfc(ub))

    @staticmethod
    def _exp_gauss_conv_int(x, ab, s, g=1):
        """
        Integral for a *single* term of exp_gauss_conv.
        g should be 1/-1
        """
        prefactor = np.exp(-ab**2*s**2 / 2.) / ab
        term1 = np.exp(ab*(ab*s**2 + x))*(1 + g * erf((ab*s**2 + x)*2**-0.5/s))
        term2 = np.exp(ab**2*s**2 / 2.)*g*erf(x * 2**-0.5 / s)
        return prefactor*(term1 - term2)

    #properties for the alternate uncertainty functions
    @property
    def funcmags(self):
        return self._funcmags
    @funcmags.setter
    def funcmags(self, value):
        oldval = self._funcmags
        self._funcmags = value
        try:
            self._validate_lnprob_func()
        except:
            self._funcmags = oldval
            raise
    @property
    def magunc(self):
        return self._magunc
    @magunc.setter
    def magunc(self, value):
        oldval = self._magunc
        self._magunc = value
        try:
            self._validate_lnprob_func()
        except:
            self._magunc = oldval
            raise
    @property
    def uncfunc(self):
        return self._uncfunc
    @uncfunc.setter
    def uncfunc(self, value):
        oldval = self._uncfunc
        self._uncfunc = value
        try:
            self._validate_lnprob_func()
        except:
            self._uncfunc = oldval
            raise
    @property
    def biasfunc(self):
        return self._biasfunc
    @biasfunc.setter
    def biasfunc(self, value):
        oldval = self._biasfunc
        self._biasfunc = value
        try:
            self._validate_lnprob_func()
        except:
            self._biasfunc = oldval
            raise
    @property
    def complfunc(self):
        return self._complfunc
    @complfunc.setter
    def complfunc(self, value):
        oldval = self._complfunc
        self._complfunc = value
        try:
            self._validate_lnprob_func()
        except:
            self._complfunc = oldval
            raise


class NormalColorModel(emceemr.Model):
    param_names = 'colorcen, colorsig, askew'.split(', ')
    has_blobs = True

    def __init__(self, magdata, tipdistr, colordata, colorunc, nstarsbelow=100,
                       priors=None):

        self.magdata = np.array(magdata)
        self.colordata = np.array(colordata)
        self.colorunc = None if colorunc is None else np.array(colorunc)

        self.tipdistr = np.array(tipdistr)
        self._len_tipdistr = self.tipdistr.size
        self.nstarsbelow = nstarsbelow

        super(NormalColorModel, self).__init__(priors)

    def lnprob(self, colorcen, colorsig, askew):
        tipmag = self.tipdistr[np.random.randint(self._len_tipdistr)]

        sorti = np.argsort(self.magdata)
        idxs = sorti[np.in1d(sorti, np.where(self.magdata > tipmag)[0])]
        msk = idxs[:self.nstarsbelow]
        assert len(self.magdata[msk]) == self.nstarsbelow
        assert np.all(self.magdata[msk] > tipmag)

        sig = np.hypot(self.colorunc[msk], colorsig)
        x = (self.colordata[msk]-colorcen)/sig
        lnpnorm = -0.5*(x**2 + np.log(sig))
        lnpskew = np.log1p(erf(askew*x*2**-0.5))
        return lnpnorm + lnpskew, tipmag

