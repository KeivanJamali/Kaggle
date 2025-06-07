import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sympy.core.relational import Relational
import numpy as np

class Golden_Section_Method:
    """Minimize a function."""
    r = (sp.sqrt(5) - 1) / 2
    decimal = 3

    def __init__(self, objective_func, domain):
        self.objective_func = objective_func
        self.a, self.b = domain
        self.n = sp.symbols("n")
        self.history = {}

    def how_many_iteration(self, error):
        n = sp.ln((2*error)/(self.b-self.a)) / sp.ln(self.r)
        n = n.evalf(self.decimal)
        return int(n) + 1

    def fit(self, error):
        limit = self.how_many_iteration(error)
        for i in range(limit):
            a_1, b_1 = self._find_new_x()
            res1, res2 = self.objective_func(a_1), self.objective_func(b_1)
            if res1 < res2:
                self.b = b_1.evalf(self.decimal)
            elif res2 < res1:
                self.a = a_1.evalf(self.decimal)
            self.history[i+1] = {f"z({a_1.evalf(self.decimal)})": res1.evalf(self.decimal),
                                 f"z({b_1.evalf(self.decimal)})": res2.evalf(self.decimal),
                                 "new domain": [self.a, self.b]}

    def _find_new_x(self):
        a_1 = self.b - self.r * (self.b - self.a)
        b_1 = self.a + self.r * (self.b - self.a)
        return a_1, b_1


class Bisection_Method:
    decimal = 3
    delta = 0.00001

    def __init__(self, objective_func, domain):
        self.objective_function = objective_func
        self.a, self.b = domain
        self.history = {}

    def how_many_iteration(self, error):
        n = (self.b-self.a) / error
        n = sp.log(n, 2)
        return int(n) + 1
    
    def fit(self, error):
        limit = self.how_many_iteration(error)
        for i in range(limit):
            m = (self.a + self.b) / 2
            a_1 = m - self.delta
            b_1 = m + self.delta
            res1, res2 = self.objective_function(a_1), self.objective_function(b_1)
            if res1 < res2:
                self.b = m
            else:
                self.a = m
            self.history[i+1] = {f"z({a_1})": res1,
                                 f"z({b_1})": res2,
                                 "new domain": [self.a, self.b]}


class Steepest_Descent_Method:
    def __init__(self, objective_function, parameters:list):
        self.objective_function = objective_function
        self.objective_gradiant = [sp.diff(objective_function, var) for var in parameters]
        self.params = parameters
        self.history = {}

    def _evaluate_gradient(self, point: list) -> list:
        subs = dict(zip(self.params, point))
        return [g.evalf(subs=subs) for g in self.objective_gradiant]
        
    def _find_negative_gradient(self, point: list) -> list:
        return [-g for g in self._evaluate_gradient(point)]
    
    def _find_alpha(self, point: list, direction: list) -> float:
        alpha = sp.symbols('alpha', real=True, positive=True)
        new_point = [x + alpha * d for x, d in zip(point, direction)]
        substituted = self.objective_function
        for var, val in zip(self.params, new_point):
            substituted = substituted.subs(var, val)
        derivative_wrt_alpha = sp.diff(substituted, alpha)
        alpha_solution = sp.solve(derivative_wrt_alpha, alpha)

        # Choose first valid (positive, real) solution
        for sol in alpha_solution:
            if sol.is_real and sol > 0:
                return float(sol.evalf())
        return 0.01  # fallback in case no good alpha is found

    def fit(self, x0: list, error=1e-6, max_iter=100):
        current_point = list(x0)
        for i in range(max_iter):
            grad = self._evaluate_gradient(current_point)
            grad_norm = sum(g**2 for g in grad)**0.5

            self.history[i] = current_point

            if grad_norm < error:
                print(f"Converged in {i} iterations.")
                return current_point

            direction = self._find_negative_gradient(current_point)
            alpha = self._find_alpha(current_point, direction)
            current_point = [x + alpha * d for x, d in zip(current_point, direction)]

        print("Reached maximum iterations.")
        return current_point
    
        
    def plot_contour(self, xlim=(-5, 5), ylim=(-5, 5), resolution=100):
        # Prepare meshgrid
        x_vals = np.linspace(xlim[0], xlim[1], resolution)
        y_vals = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Lambdify the symbolic function for NumPy use
        f_lambdified = sp.lambdify((self.params[0], self.params[1]), self.objective_function, 'numpy')
        Z = f_lambdified(X, Y)

        # Plot contour
        plt.figure(figsize=(5, 4))
        plt.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Function Value')
        
        # Extract history points
        xs = [float(p[0]) for p in self.history.values()]
        ys = [float(p[1]) for p in self.history.values()]
        
        # Plot points and path
        plt.plot(xs, ys, 'ro--', label='Descent Path')
        plt.scatter(xs[0], ys[0], color='blue', label='Start')
        plt.scatter(xs[-1], ys[-1], color='green', label='End (Min)')

        for i in range(len(xs)-1):
            plt.arrow(xs[i], ys[i],
                    xs[i+1] - xs[i], ys[i+1] - ys[i],
                    head_width=0.1, length_includes_head=True, color='red')

        plt.title('Steepest Descent Path')
        plt.xlabel(str(self.params[0]))
        plt.ylabel(str(self.params[1]))
        plt.legend()
        plt.grid(True)
        plt.savefig("figs\question4.png")
        plt.show()


