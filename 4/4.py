import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1
t = sp.symbols('t')
k, w, y0 = sp.symbols('k omega y0', positive=True)
y = sp.Function('y')

equation = sp.Eq(y(t).diff(t), -k * y(t) + sp.sin(w * t))
solution = sp.dsolve(equation, y(t), ics={y(0): y0})
y_analytic = sp.simplify(solution.rhs)
print("Аналитическое решение y(t) =", y_analytic)

# 2
K, W, Y0 = 0.5, 2.0, 1.0

def f(t, y):
    return -K * y[0] + np.sin(W * t)

t_eval = np.linspace(0, 15, 500)
solution_num = solve_ivp(f, (0, 15), [Y0], t_eval=t_eval)
t_num = solution_num.t
y_num = solution_num.y[0]

# 3
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_num, y_num, 'b-', lw=2, label=r'$y(t)$ — численное решение')
ax.plot(t_num, np.sin(W * t_num), 'r--', lw=1.5, label=r'$\sin(\omega t)$ — внешняя сила')
ax.set_title(r'$dy/dt = -ky + \sin(\omega t)$, $k=0.5$, $\omega=2$, $y_0=1$')
ax.set_xlabel('t')
ax.set_ylabel('y(t)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('relaxation.png', dpi=150)
plt.show()