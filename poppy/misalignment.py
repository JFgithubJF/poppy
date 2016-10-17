from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.units as u
import logging
import time

import poppy
from poppy.poppy_core import PlaneType, _FFTW_AVAILABLE, OpticalSystem, Wavefront
from . import utils

_log = logging.getLogger('poppy')


if _FFTW_AVAILABLE:
    import pyfftw

__all__ = ['MisalignedLens']


class MisalignedLens(poppy.optics.AnalyticOpticalElement):
    """
    Misaligned lens
    """

    @utils.quantity_input(z=u.m)
    def __init__(self,
                 R1=1.0*u.m,
                 R2=1.0*u.m,
                 n=1.515,
                 planetype=PlaneType.intermediate,
                 name='Misaligned Lens',
                 **kwargs):
        poppy.AnalyticOpticalElement.__init__(self, name=name, planetype=planetype, **kwargs)
        f_lens = R1*R2/((n-1.0)*(R2-R1))
        self.fl = f_lens.to(u.m)

    def get_phasor(self, wave):
        """ return complex phasor for the quadratic phase

        Parameters
        ----------
        wave : obj
            a Fresnel Wavefront object
        """
        Quadratic_Lens = poppy.QuadraticLens(self.fl, name='Quadratic_Lens')
        wave *= Quadratic_Lens
        y, x = wave.coordinates()
        rsqd = (x ** 2 + y ** 2) * u.m ** 2
        _log.debug("Misaligned lens focal length ={0:0.2e}".format(self.fl))

        k = 2 * np.pi / wave.wavelength
        #lens_phasor = np.exp(1.j * k * rsqd / (2.0 * self.fl))
        lens_phasor = np.exp(0.0 * 1.j * k * rsqd / (2.0 * self.fl))
        return lens_phasor
    
    