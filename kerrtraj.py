from __future__ import division
import numpy as np
from scipy.integrate import odeint
import sympy as sym
from matplotlib import pyplot as plt

sym.init_printing(use_unicode=True)


# metric class
class Metric:
    # coordinates
    t, r, th, ph = sym.symbols('t r th ph')

    # metric constants
    a = sym.symbols('a')

    # metric constants
    Sigma = r * r + a * a + sym.cos(th) ** 2
    Delta = r * r - 2 * r + a * a
    # metric components
    gtt = -(1 - 2 * r / Sigma)
    gtph = -4  * a * r * sym.sin(th) ** 2 / Sigma
    grr = Sigma / Delta
    gthth = Sigma
    gphph = (r * r + a * a + 2  * a * a * r * sym.sin(th) / Sigma) * sym.sin(th) ** 2

    def __init__(self, mya):
        self.mya = mya
        # initialize metric componenets as functions
        self.mygtt = sym.lambdify((self.r, self.th, self.ph), self.gtt.subs({self.a: mya}), 'numpy')
        self.mygtph = sym.lambdify((self.r, self.th, self.ph), self.gtph.subs({self.a: mya}), 'numpy')
        self.mygrr = sym.lambdify((self.r, self.th, self.ph), self.grr.subs({self.a: mya}), 'numpy')
        self.mygthth = sym.lambdify((self.r, self.th, self.ph), self.gthth.subs({self.a: mya}), 'numpy')
        self.mygphph = sym.lambdify((self.r, self.th, self.ph), self.gphph.subs({self.a: mya}), 'numpy')


class EquationsOfMotions(Metric):

    def __init__(self, mya):
        Metric.__init__(self, mya)

    # coordinates for convinience
    r = Metric.r
    th = Metric.th
    ph = Metric.ph

    # additional variables
    pt, pr, pth, pph = sym.symbols('pt pr pth pph')

    # constants of motion
    L, E, Q = sym.symbols('L E Q')
    # utility expressions
    TH = Q - (sym.cos(th)**2)*(Metric.a ** 2 * (1 - E ** 2) + (sym.sin(th) ** (-2)) * (L ** 2))
    R = ((E * (r ** 2 + Metric.a ** 2) - Metric.a * L) ** 2 -
         Metric.Delta * ((r ** 2) + (L - Metric.a * E) ** 2 + Q))

    # time derivatives
    dtdtau = 1 / (2 * Metric.Delta * Metric.Sigma) * sym.diff(R + Metric.Delta * TH, E)
    drdtau = Metric.Delta / Metric.Sigma * pr
    dthdtau = pth / Metric.Sigma
    dphdtau = -1 / (2 * Metric.Delta * Metric.Sigma) * sym.diff(R + Metric.Delta * TH, L)

    dptdtau = sym.Integer(0)
    dprdtau = (-sym.diff(Metric.Delta / 2 / Metric.Sigma, r) * (pr ** 2) -
               sym.diff(1 / 2 / Metric.Sigma, r) * (pth ** 2) +
               sym.diff((R + Metric.Delta * TH) / (2 * Metric.Delta * Metric.Sigma), r))
    dpthdtau = (-sym.diff(Metric.Delta / 2 / Metric.Sigma, th) * pr ** 2 -
                sym.diff(1 / 2 / Metric.Sigma, th) * pth ** 2 +
                sym.diff((R + Metric.Delta * TH) / (2 * Metric.Delta * Metric.Sigma), th))
    dpphdtau = sym.Integer(0)

    def convert_3to4vel(self, r, th, ph, vr, vth, vph):
        gtt = self.mygtt(r, th, ph)
        gtph = self.mygtph(r, th, ph)
        grr = self.mygrr(r, th, ph)
        gthth = self.mygthth(r, th, ph)
        gphph = self.mygphph(r, th, ph)
        ut = 1. / np.sqrt(-gtt - 2. * gtph * vph - gphph * vph * vph - grr * vr * vr - gthth * vth * vth)
        return ut, ut * vr, ut * vth, ut * vph

    def lower_4vel(self, r, th, ph, ut, ur, uth, uph):
        gtt = self.mygtt(r, th, ph)
        gtph = self.mygtph(r, th, ph)
        grr = self.mygrr(r, th, ph)
        gthth = self.mygthth(r, th, ph)
        gphph = self.mygphph(r, th, ph)

        u_t = gtt * ut + gtph * uph
        u_r = grr * ur
        u_th = gthth * uth
        u_ph = gtph * ut + gphph * uph

        return u_t, u_r, u_th, u_ph

    def initialize(self, r0, th0, ph0, vr, vth, vph):
        ut, ur, uth, uph = self.convert_3to4vel(r0, th0, ph0, vr, vth, vph)
        u_t, u_r, u_th, u_ph = self.lower_4vel(r0, th0, ph0, ut, ur, uth, uph)


        #TEST
        print ut, self.dtdtau.subs({Metric.r:r0,
                                    Metric.th:th0,
                                    Metric.ph:th0,
                                    Metric.a:0.,
                                    self.E:-u_t,
                                    self.L:u_ph})
        self.initial_state = np.array([0., r0, th0, ph0, u_t, u_r, u_th, u_ph])

        self.myE = -u_t
        self.myL = u_ph
        self.myQ = (u_th) ** 2 + (np.cos(th0) ** 2) * (
                (self.mya ** 2) * (1. - self.myE ** 2) + (self.myL ** 2) / np.square(np.sin(th0)))

        # lambdify equations of motion
        fields = ['dtdtau', 'drdtau', 'dthdtau', 'dphdtau',
                  'dptdtau', 'dprdtau', 'dpthdtau', 'dpphdtau']

        t, r, th, ph = Metric.t, Metric.r, Metric.th, Metric.ph
        pt, pr, pth, pph = self.pt, self.pr, self.pth, self.pph
        mya = self.mya
        for field in fields:
            newname = 'my' + field
            setattr(self, newname,
                    sym.lambdify((t, r, th, ph, pt, pr, pth, pph),
                                 getattr(self, field).subs({self.L: self.myL,
                                                            self.E: self.myE,
                                                            self.Q: self.myQ,
                                                            Metric.a: mya})))

    current_state = np.zeros(8)

    def time_derivative(self, state, tau):
        derivative = np.zeros(8)
        derivative[0] = self.mydtdtau(*state)
        derivative[1] = self.mydrdtau(*state)
        derivative[2] = self.mydthdtau(*state)
        derivative[3] = self.mydphdtau(*state)

        derivative[4] = self.mydptdtau(*state)
        derivative[5] = self.mydprdtau(*state)
        derivative[6] = self.mydpthdtau(*state)
        derivative[7] = self.mydpphdtau(*state)

        return derivative


    def do_integration(self, final_time, number_of_outputs):
        taus = np.linspace(0., final_time, number_of_outputs)
        solution = odeint(self.time_derivative, self.initial_state, taus,rtol=1.e-10)
        return solution







if __name__ == '__main__':
    spin = 0.
    eom = EquationsOfMotions(spin)
    #print eom.dpthdtau
    ti = 0.
    ri = 1e2
    thi = np.pi/2.
    phii = 0.
    pri = 0.
    pthi = 0.
    pphii = ri ** (-3 / 2)
    eom.initialize(ri, thi, phii, pri, pthi, .4712*pphii)
    tf = 10*2*np.pi*ri**(3./2)
    print eom.myE, eom.myL, ri*ri*(ri**(-3./2))
    solution = eom.do_integration(tf,5000)
    print -solution[-1,4], solution[-1,7]

    ts = solution[:,0]
    rs = solution[:,1]
    phs = solution[:,3]
    ths = solution[:,2]
    twophi = np.linspace(0.,2*np.pi,100)
    plt.plot(ts,phs)
    plt.show()
    plt.plot(rs*np.cos(phs), rs*np.sin(phs),linewidth=1)
    plt.plot(6.*np.cos(twophi),6.*np.sin(twophi), linewidth=1)
    plt.plot(2.*np.cos(twophi),2.*np.sin(twophi), linewidth=1)
    plt.axis('image')
    plt.axis([-ri*1.1, ri*1.1, -ri*1.1, ri*1.1])
    plt.show()
    #print eom.mu, eom.myE, eom.myL, eom.myQ
    #print eom.mydprdtau(ti, ri, thi, phii, -eom.myE, pri, pthi, pphii)
    # print eom.mydtdtau(0.,ri, thi,phii, eom.myE, pri, pthi, pphii)
