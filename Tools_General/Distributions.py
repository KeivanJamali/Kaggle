from IPython.display import display, Math
from math import factorial
import random
import numpy as np
import pandas as pd
from scipy.special import gamma as gm
import matplotlib.pyplot as plt


class DU:
    def __init__(self, i:int, j:int, size:int=100, print_:bool=False):
        self.print_ = print_
        self.sample = Distribution_Sample_Maker.discrete_uniform(i=i, j=j, size=size)
        self.type_ = "discrete"
        self.i = i
        self.j = j
        self.size = size
        self.mean = self._mean()
        self.var = self._var()

    def _mean(self):
        temp = (self.i+self.j)/2
        if self.print_:
            display(Math(r"\text{Mean of the Discrete Uniform is } \frac{i+j}{2} = " + f"{temp:.2f}"))
        return temp
    
    def _var(self):
        temp = ((self.j-self.i+1)**2 - 1) / 12
        if self.print_:
            display(Math(r"\text{Variance of the Discrete Uniform is } \frac{(j-i+1)^2 - 1}{12} = " + f"{temp:.2f}"))
        return temp
    
    def mass_function(self, x):
        if self.i <= x <= self.j:
            return (self.j-self.i+1)**(-1)
        else:
            return 0
        
    def distribution_function(self, x):
        if x < self.i:
            return 0
        elif x > self.j:
            return 1
        else:
            return (int(x)-self.i+1)/(self.j-self.i+1)
        

class Bernoulli:
    def __init__(self, p:float, size:int=100, print_:bool=False):
        self.print_= print_
        self.sample = Distribution_Sample_Maker.bernoulli(p=p, size=size)
        self.type_ = "discrete"
        self.p = p
        self.mean = self._mean()
        self.var = self._var()
        self.mode = self._mode()

    def _mean(self):
        temp = self.p
        if self.print_:
            display(Math(r"\text{Mean of the Bernoulli is } p = " + f"{temp:.2f}"))
        return temp
    
    def _var(self):
        temp = self.p*(1-self.p)
        if self.print_:
            display(Math(r"\text{Variance of the Bernoulli is } p\times(1-p) = " + f"{temp:.2f}"))
        return temp
    
    def _mode(self):
        if self.p < 0.5:
            temp = 0
        elif self.p == 0.5:
            temp = [0, 1]
        else:
            temp = 1
        if self.print_:
            display(Math(r"\text{Mode of the Bernoulli is } " + f"{temp}"))
        return temp
    
    def mass_function(self, x):
        if x==1:
            return self.p
        elif x==0:
            return 1-self.p
        else:
            return 0
        
    def distribution_function(self, x):
        if x < 0:
            return 0
        elif 0 <= x < 1:
            return 1-self.p
        else:
            return 1


class Bin:
    def __init__(self, t:int, p:float, size:int=100, print_:bool=False):
        self.print_= print_
        self.sample = Distribution_Sample_Maker.binomial(t=t, p=p, size=size)
        self.type_ = "discrete"
        self.t = t
        self.p = p
        self.mean = self._mean()
        self.var = self._var()
        self.mode = self._mode()

    def _mean(self):
        temp = self.t * self.p
        if self.print_:
            display(Math(r"\text{Mean of the Binomial is } np = " + f"{temp:.2f}"))
        return temp
    
    def _var(self):
        temp = self.t * self.p*(1-self.p)
        if self.print_:
            display(Math(r"\text{Variance of the Binomial is } n \times p \times (1-p) = " + f"{temp:.2f}"))
        return temp
    
    def _mode(self):
        if (self.p*(self.t+1)) % 1 < 0.00001:
            temp = [self.p*(self.t+1) - 1, self.p*(self.t+1)]
        else:
            temp = int(self.p*(self.t+1))
        if self.print_:
            display(Math(r"\text{Mode of the Binomial is } " + f"{temp}"))
        return temp
    
    def mass_function(self, x):
        if x in range(0, self.t+1):
            coef = factorial(self.t)/(factorial(x)*factorial(self.t-x))
            return coef * self.p**(x) * (1-self.p)**(self.t-x)
        else:
            return 0
        
    def distribution_function(self, x):
        if x < 0:
            return 0
        elif 0 <= x <= self.t:
            temp = sum([self.mass_function(i) for i in range(0, int(x))])
            return temp
        else:
            return 1


class Poisson:
    def __init__(self, lamb:int, size:int=100, print_:bool=False):
        self.print_= print_
        self.sample = Distribution_Sample_Maker.poisson(lamb=lamb, size=size)
        self.type_ = "discrete"
        self.lamb = lamb
        self.mean = self._mean()
        self.var = self._var()
        self.mode = self._mode()

    def _mean(self):
        temp = self.lamb
        if self.print_:
            display(Math(r"\text{Mean of the Poisson is } \lambda = " + f"{temp:.2f}"))
        return temp
    
    def _var(self):
        temp = self.lamb
        if self.print_:
            display(Math(r"\text{Variance of the Poisson is } \lambda = " + f"{temp:.2f}"))
        return temp
    
    def _mode(self):
        if self.lamb % 1 < 0.00001:
            temp = [self.lamb-1, self.lamb]
        else:
            temp = int(self.lamb)
        if self.print_:
            display(Math(r"\text{Mode of the Poisson is } " + f"{temp}"))
        return temp
    
    def mass_function(self, x):
        if x >= 0 and x%1 < 0.0001:
            temp = (np.e**(-self.lamb) * self.lamb**x) / factorial(x)
            return temp
        else:
            return 0
        
    def distribution_function(self, x):
        if x < 0:
            return 0
        elif 0 <= x:
            temp = np.e**(-self.lamb) * sum((self.lamb**i)/factorial(i) for i in range(0, int(x)))
            return temp


class Geom:
    def __init__(self, p:float, size:int=100, print_:bool=False):
        self.print_= print_
        self.sample = Distribution_Sample_Maker.geometry(p=p, size=size)
        self.type_ = "discrete"
        self.p = p
        self.mean = self._mean()
        self.var = self._var()
        self.mode = self._mode()

    def _mean(self):
        temp = (1-self.p)/self.p
        if self.print_:
            display(Math(r"\text{Mean of the Geometry is } \frac{1-p}{p} = " + f"{temp:.2f}"))
        return temp
    
    def _var(self):
        temp = (1-self.p)/self.p**2
        if self.print_:
            display(Math(r"\text{Variance of the Geometry is } \frac{1-p}{p^2} = " + f"{temp:.2f}"))
        return temp
    
    def _mode(self):
        temp = 0
        if self.print_:
            display(Math(r"\text{Mode of the Geometry is } " + f"{temp}"))
        return temp
    
    def mass_function(self, x):
        if x >= 0 and x%1 < 0.0001:
            temp = self.p*(1-self.p)**x
            return temp
        else:
            return 0
        
    def distribution_function(self, x):
        if 0 <= x:
            temp = 1 - (1-self.p)**(1+int(x))
        else:
            temp = 0
        return temp


class Negbin:
    def __init__(self, s:int, p:float, size:int=100, print_:bool=False):
        self.print_= print_
        self.sample = Distribution_Sample_Maker.negativebinomial(s=s, p=p, size=size)
        self.type_ = "discrete"
        self.p = p
        self.s = s
        self.mean = self._mean()
        self.var = self._var()
        self.mode = self._mode()

    def _mean(self):
        temp = self.s * ((1-self.p)/self.p)
        if self.print_:
            display(Math(r"\text{Mean of the Negative Binomial is } s \times \frac{1-p}{p} = " + f"{temp:.2f}"))
        return temp
    
    def _var(self):
        temp = self.s * ((1-self.p)/self.p**2)
        if self.print_:
            display(Math(r"\text{Variance of the Negative Binomial is } s \times \frac{1-p}{p^2} = " + f"{temp:.2f}"))
        return temp
    
    def _mode(self):
        temp = (self.s * (1-self.p) - 1)/self.p
        if temp % 1 < 0.0001:
            temp = [temp, temp + 1]
        else:
            temp = int(temp)
        if self.print_:
            display(Math(r"\text{Mode of the Negative Binomial is } " + f"{temp}"))
        return temp
    
    def mass_function(self, x):
        if x >= 0 and x%1 < 0.0001:
            coef = factorial(int(self.s+x-1)) / factorial(x) * factorial(self.s-1)
            temp = coef * self.p**self.s * (1-self.p)**x
            return temp
        else:
            return 0
        
    def distribution_function(self, x):
        if 0 <= x:
            temp = sum(self.mass_function(i) for i in range(0, int(x)+1))
        else:
            temp = 0
        return temp


class Uniform:
    def __init__(self, a:float, b:float, size:int=100, print_:bool=False):
        self.print_= print_
        self.sample = Distribution_Sample_Maker.uniform(a=a, b=b, size=size)
        self.type_ = "continuos"
        self.a = a
        self.b = b
        self.mean = self._mean()
        self.var = self._var()
        self.mode = self._mode()

    def _mean(self):
        temp = (self.a+self.b)/2
        if self.print_:
            display(Math(r"\text{Mean of the Uniform is } \frac{1}{b-a} = " + f"{temp:.2f}"))
        return temp
    
    def _var(self):
        temp = ((self.b-self.a)**2) / 12
        if self.print_:
            display(Math(r"\text{Variance of the Uniform is } \frac{(b-a)^2}{12} = " + f"{temp:.2f}"))
        return temp
    
    def _mode(self):
        if self.print_:
            display(Math(r"\text{Mode of the Uniform? } " + r"\text{You must be kidding!!!}"))
        return False
    
    def mass_function(self, x):
        if self.a <= x <= self.b:
            temp = 1/(self.b-self.a)
            return temp
        else:
            return 0
        
    def distribution_function(self, x):
        if x < self.a:
            return 0
        elif self.a <= x <= self.b:
            return (x-self.a)/(self.b-self.a)


class Expo:
    def __init__(self, beta:float, size:int=100, print_:bool=False):
        self.print_= print_
        self.sample = Distribution_Sample_Maker.exponential(beta=beta, size=size)
        self.type_ = "continuos"
        self.beta = beta
        self.mean = self._mean()
        self.var = self._var()
        self.mode = self._mode()

    def _mean(self):
        temp = self.beta
        if self.print_:
            display(Math(r"\text{Mean of the Exponential is } \beta = " + f"{temp:.2f}"))
        return temp
    
    def _var(self):
        temp = self.beta**2
        if self.print_:
            display(Math(r"\text{Variance of the Exponential is } \beta = " + f"{temp:.2f}"))
        return temp
    
    def _mode(self):
        if self.print_:
            display("Zero(0)")
        return 0
    
    def mass_function(self, x):
        if x >= 0:
            temp = (np.exp(-x/self.beta))/(self.beta)
            return temp
        else:
            return 0
        
    def distribution_function(self, x):
        if x >= 0:
            return 1 - np.exp(-x/self.beta)
        else:
            return 0


class Gamma:
    def __init__(self, alpha:float, beta:float, size:int=100, print_:bool=False):
        self.print_= print_
        self.sample = Distribution_Sample_Maker.exponential(beta=beta, size=size)
        self.type_ = "continuos"
        self.beta = beta
        self.alpha = alpha
        self.mean = self._mean()
        self.var = self._var()
        self.mode = self._mode()

    def _mean(self):
        temp = self.beta * self.alpha
        if self.print_:
            display(Math(r"\text{Mean of the Gamma is } \alpha \times \beta = " + f"{temp:.2f}"))
        return temp
    
    def _var(self):
        temp = self.alpha * self.beta**2
        if self.print_:
            display(Math(r"\text{Variance of the Gamma is } \alpha \times \beta ^ 2 = " + f"{temp:.2f}"))
        return temp
    
    def _mode(self):
        if self.alpha < 1:
            temp = 0
        else:
            temp = self.beta * (self.alpha-1)
        if self.print_:
            display(f"{temp}")
        return temp
    
    def mass_function(self, x):
        if x > 0:
            temp1 = (self.beta ** (-self.alpha)) * (x**(self.alpha-1)) * (np.exp(-x/self.beta))
            temp2 = gm(x)
            return temp1/temp2
        else:
            return 0
        
    def distribution_function(self, x):
        if x > 0:
            temp1 = sum(((x/self.beta)**j)/(factorial(j)) for j in range(0, self.alpha-1))
            temp2 = 1 - np.exp(-x/self.beta) * temp1
            return temp2
        else:
            return 0


class Distribution_Sample_Maker:
    @staticmethod
    def discrete_uniform(i:int, j:int, size:int) -> np.array:
        sample = np.random.randint(i, j+1, size=size)
        return sample
    
    @staticmethod
    def bernoulli(p:float, size:int) -> np.array:
        sample = np.random.choice([0, 1], size=size, p=[1 - p, p])
        return sample
    
    @staticmethod
    def binomial(t:int, p:float, size:int) -> np.array:
        sample = np.random.binomial(n=t, p=p, size=size)
        return sample
    
    @staticmethod
    def poisson(lamb:int, size:int) -> np.array:
        sample = np.random.poisson(lam=lamb, size=size)
        return sample

    @staticmethod
    def geometry(p:float, size:int) -> np.array:
        """The result shows the number of failure before success. Not the total trial which is n+1"""
        uniform_random = np.random.uniform(0, 1, size=size)
        sample = np.ceil(np.log(uniform_random) / np.log(1-p)) - 1
        return sample.astype(int)
    
    @staticmethod
    def negativebinomial(s:int, p:float, size:int) -> np.array:
        sample = np.random.negative_binomial(n=s, p=p, size=size)
        return sample
    
    @staticmethod
    def uniform(a:int, b:int, size:int) -> np.array:
        sample = np.random.uniform(a, b+1, size=size)
        return sample
    
    @staticmethod
    def exponential(beta:float, size:int) -> np.array:
        sample = np.random.exponential(scale=beta, size=size)
        return sample
    
    @staticmethod
    def gamma(alpha:float, beta:float, size:int) -> np.array:
        sample = np.random.gamma(shape=alpha, scale=beta, size=size)
        return sample
    


class Sample_Calculation:
    def __init__(self, sample):
        self.sample = sample.sample
        self.type_ = sample.type_
        self.mean = self._mean()
        self.var = self._var()
        self.data, self.distribution = self._make_data()
        self.max = self._max()
        self.n = len(self.sample)

    def plot_data(self) -> None:
        if self.type_=="discrete":
            plt.bar(self.data.keys(), self.data.values(), width=0.3, alpha=0.7, edgecolor="red")
            plt.xticks(list(self.data.keys()))
        elif self.type_=="continuos":
            plt.scatter(self.data.keys(), self.data.values(), marker=".")
        plt.xlabel("X")
        plt.ylabel("Repeate")
        plt.title("Function Plot")
        plt.show()

    def plot_distribution(self) -> None:
        plt.plot(self.distribution.keys(), self.distribution.values())
        if self.type_=="discrete":
            plt.xticks(list(self.data.keys()))
        plt.xlabel("X")
        plt.ylabel("Cumulative probability")
        plt.title("Distribution Plot")
        plt.show()

    def _make_data(self):
        data = {}
        distribution = {0:0}
        sorted_sample = sorted(self.sample)
        for i in sorted_sample:
            if self.type_=="descrete":
                i = int(i)
            elif self.type_=="continuos":
                i = float(i)
            if i in data.keys():
                data[i] += 1
                distribution[i] += 1
            else:
                data[i] = 1
                distribution[i] = max(distribution.values()) + 1
        return data, distribution

    def _mean(self):
        # m = sum(self.sample) / len(self.sample)
        m = np.mean(self.sample)
        return float(m)

    def _var(self, ddof=0):
        # var = sum((x - self.mean(print_=False)) ** 2 for x in self.sample) / (len(self.sample) - 1)
        var = np.var(self.sample, ddof=ddof)
        return float(var)
    
    def _max(self):
        return max(self.sample)

    def MLE(self, type_:str):
        if type_ == "discrete_uniform":
            print(f"We have a ''Discrete Uniform'' distribution\nfor the maximum Likelihood Estimator we have \nthese parameters: i={min(self.sample)} and j={max(self.sample)} with size of {len(self.sample)}")

        elif type_ == "bernoulli":
            temp = sum(self.sample)/len(self.sample)
            print(f"We have a ''Bernoulli'' distribution\nfor the maximum Likelihood Estimator we have \nthese parameters: p={temp} with size of {len(self.sample)}")

        elif "binomial" in type_ and not "negative_binomial" in type_:
            par = type_.split("_")
            if len(par) == 2:
                temp = sum(self.sample) / len(self.sample)
                temp = temp / int(par[1])
                print(f"We have a ''Binomial'' distribution with known parameters of t={par[1]}\nfor the maximum Likelihood Estimator we have \nthese parameters: p={temp:.2f} with size of {len(self.sample)}")
            elif len(par) == 1:
                f_k = np.array([np.sum(self.sample >= k) for k in range(1, self.max + 1)])
                def g(t, p):
                    term1 = np.sum(f_k * np.log(t - np.arange(1, len(f_k) + 1) + 1))
                    term2 = self.n * t * np.log(1 - p)
                    term3 = self.n * self.mean * np.log(p / (1-p))
                    return term1 + term2 + term3                
                
                t_values = np.arange(self.max, int((self.mean * (self.max-1)) / (1 - self.var / self.mean) + 1))
                best_t = 0
                best_p = 0
                max_likelihood = -np.inf

                for t in t_values:
                    p = self.mean / t
                    if 0 < p < 1:
                        likelihood = g(t, p)
                        if likelihood > max_likelihood:
                            max_likelihood = likelihood
                            best_t = t
                            best_p = p
                print(f"We have a ''Binomial'' distribution without known parameters\nfor the maximum Likelihood Estimator we have \nthese parameters: t={best_t} and p={best_p:.2f} with size of {len(self.sample)}")

        elif type_ == "poisson":
            temp = sum(self.sample)/len(self.sample)
            print(f"We have a ''Poisson'' distribution \nfor the maximum Likelihood Estimator we have \nthese parameters: lambda={temp:.2f} with size of {len(self.sample)}")

        elif type_ == "geometry":
            temp = 1/(sum(self.sample)/len(self.sample)+1)
            print(f"We have a ''Geometry'' distribution \nfor the maximum Likelihood Estimator we have \nthese parameters: lambda={temp:.2f} with size of {len(self.sample)}")

        elif "negative_binomial" in type_:
            par = type_.split("_")
            if len(par) == 3:
                s = int(par[2])
                temp = s / ((sum(self.sample)/len(self.sample)) + s)
                print(f"We have a ''Negative Binomial'' distribution with known parameter of s={s} \nfor the maximum Likelihood Estimator we have \nthese parameters: p={temp:.2f} with size of {len(self.sample)}")

            if len(par) == 2:
                f_k = np.array([np.sum(self.sample >= k) for k in range(1, self.max + 1)])
                def g(t, p):
                    term1 = np.sum(f_k * np.log(s + np.arange(1, len(f_k) + 1) - 1))
                    term2 = self.n * s * np.log(p)
                    term3 = self.n * self.mean * np.log(1-p)
                    return term1 + term2 + term3                
                s = 0
                best_s = 0
                best_p = 0
                max_likelihood = -np.inf
                while True:
                    s += 1
                    p = s / int(self.mean+s)
                    if 0 < p < 1:
                        likelihood = g(s, p)
                        if likelihood > max_likelihood:
                            max_likelihood = likelihood
                            best_s = s
                            best_p = p
                        else:
                            break
                print(f"We have a ''NegativeBinomial'' distribution without known parameters\nfor the maximum Likelihood Estimator we have \nthese parameters: s={best_s} and p={best_p:.2f} with size of {len(self.sample)}")

        elif type_ == "uniform":
            print(f"We have a ''Uniform'' distribution\nfor the maximum Likelihood Estimator we have \nthese parameters: a={min(self.sample)} and b={max(self.sample)} with size of {len(self.sample)}")

        elif type_ == "exponential":
            temp = self.mean()
            print(f"We have a ''Exponential'' distribution\nfor the maximum Likelihood Estimator we have \nthese parameters: beta={temp} with size of {len(self.sample)}")

        elif type_ == "gamma":
            temp = 0
        
