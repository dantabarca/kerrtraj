from kerrtraj import *
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt


def kepler(radius):
    ri = radius
    thi = np.pi / 2.
    phii = 0.
    pri = 0.
    pthi = 0.
    pphii = ri ** (-3 / 2)
    # one period
    tf = 1*2*np.pi*ri**(3./2)
    spin = 0.
    eom = EquationsOfMotions(spin)
    eom.initialize(ri, thi, phii, pri, pthi, 9.*pphii)
    solution = eom.do_integration(tf, 500)

    ts = solution[:,0]
    rs = solution[:,1]
    phs = solution[:,3]
    ths = solution[:,2]
    twophi = np.linspace(0.,2*np.pi,100)
    #plt.plot(ts,phs%(2.*np.pi))
    #plt.show()
    plt.plot(rs*np.cos(phs), rs*np.sin(phs),linewidth=1)
    plt.plot(6.*np.cos(twophi),6.*np.sin(twophi), linewidth=1)
    plt.plot(2.*np.cos(twophi),2.*np.sin(twophi), linewidth=1)
    plt.axis('image')
    plt.axis([-ri*1.1, ri*1.1, -ri*1.1, ri*1.1])
    plt.show()

if __name__ == '__main__':
    kepler(100.)