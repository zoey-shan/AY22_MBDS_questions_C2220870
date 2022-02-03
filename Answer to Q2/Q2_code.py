import numpy as np
import matplotlib.pyplot as plt


def diff_func(x, _t):
    # -- 8.1. Calculate rate of changes --
    # x = [C(E), C(S), C(ES), C(P)]
    # k1 = 100, k2 = 600, k3 = 150
    # Return: [dC(E)/dt, dC(S)/dt, dC(ES)/dt, dC(P)/dt]
    t1 = 100. / 60. * x[0] * x[1]
    t2 = 600. / 60. * x[2]
    t3 = 150. / 60. * x[2]
    return np.asarray([-t1 + t2 + t3, -t1 + t2, t1 - t2 - t3, t3])


def rk4(x, t, dt, f):
    # 4th-order Runge-Kutta
    k1 = dt * f(x, t)
    k2 = dt * f(x + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * f(x + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * f(x + k3, t + dt)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.


def main():
    x = np.asarray([1., 10., 0., 0.])
    dt = 0.001
    ts = np.arange(0, 30 + dt, step=dt)
    rcs = np.zeros((ts.size, 4))
    xs = np.zeros((ts.size, 4))
    ss = np.zeros(ts.size)
    vs = np.zeros(ts.size)
    for i, t in enumerate(ts):
        ss[i] = x[1]
        rc = diff_func(x, 0)
        rcs[i, :] = rc
        vs[i] = rc[3]
        xs[i, :] = x
        x = rk4(x, t, dt, diff_func)

    # -- 8.2. Plot rate of changes --
    plt.figure()
    plt.plot(ts, rcs)
    plt.legend(['E', 'S', 'ES', 'P'])
    plt.xlabel('Time (s)')
    plt.ylabel('Rate of changes (μM/s)')
    # plt.show()
    plt.savefig('8_2_roc.png')

    # -- Plot concentration --
    plt.figure()
    plt.plot(ts, xs)
    plt.legend(['E', 'S', 'ES', 'P'])
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (μM)')
    # plt.show()
    plt.savefig('8_2_c.png')

    # -- 8.3. Plot velocity w.r.t. concentration of S --
    plt.figure()
    plt.plot(ss, vs)
    maxv = vs.max()
    pos = ss[np.argmax(vs)]
    plt.annotate(f'Vm={maxv:.2f}', xy=(pos, maxv), xytext=(pos-1.7, maxv))
    plt.xlabel('Concentration of S (μM)')
    plt.ylabel('Velocity (μM/s)')
    # plt.show()
    plt.savefig('8_3.png')


if __name__ == '__main__':
    main()
