import numpy as np
import pandas as pd
from scipy import stats


class Validation_Tools:
    def __init__(self, data_list:list, name_list:list, decimal:int=4):
        if not len(data_list) == len(name_list):
            print(f"[ERROR] There should be a name for each data. We will use those names for refrence.")
        self.decimal = decimal
        self.data = {}
        self.analysis = {}
        self.comparisons = {}
        for i in range(len(data_list)):
            self.add_data(data_list[i], name_list[i])

    def add_data(self, data_values:np.array, name:str):
        if name in self.data.keys():
            print(f"[ERROR] We already have '{name}' data in the store. If you want to rewrite it, please first delete it. using this method: object.remove(name).")
            return None
        data = np.array(data_values)
        mean_data = self.calculate_mean(data=data)
        S2 = self.calculate_S2(data=data, mean_data=mean_data)
        n = len(data)
        self.data[name] = {"data":data,
                           "n":n,
                           "mean":mean_data,
                           "S2": S2,
                           "confidence_intervals":{}}
        print(f"[INFO] '{name}' data added to the storage with Mean:{mean_data} and Variance:{S2}.")
        
    def remove(self, name:str):
        if not name in self.data.keys():
            print(f"[ERROR] We don't have such data in storage.")
            return None
        else:
            self.data.pop(name)
            temp = self.analysis.keys()
            for an in temp:
                if "_"+name+"_" in an:
                    self.analysis.pop(an)
            print(f"[INFO] '{name}' was deleted succesfully. We also removed all the related analysis.")

    @staticmethod
    def calculate_mean(data:np.array):
        mean_data = data.mean()
        return mean_data

    @staticmethod
    def calculate_S2(data:np.array, mean_data:float):
        S2 = ((data-mean_data)**2).sum() / (len(data) - 1)
        return S2

    def confidence_interval(self, data_name:str, alpha:float):
        interval_name = f"{(100 - 100*alpha):.3f}%"
        if interval_name in self.data[data_name]["confidence_intervals"].keys():
            return self.data[data_name]["confidence_intervals"][interval_name]
        else:
            n = self.data[data_name]["n"]
            mean_data = self.data[data_name]["mean"]
            S2 = self.data[data_name]["S2"]
            if n > 50:
                z_value = stats.norm.ppf(1 - alpha/2)
                temp = z_value
            else:
                t_value = stats.t.ppf(1-alpha/2, n-1)
                temp = t_value
            self.data[data_name]["confidence_intervals"][interval_name] = [mean_data - temp * np.sqrt(S2/n), mean_data + temp * np.sqrt(S2/n)]

    def independent_sampling_with_equal_variance(self, data_name_1:str, data_name_2:str, alpha:float, print_:bool=True):
        if len(self.data) < 2:
            print(f"[ERROR] Please add the second data to the object first. Use this: object.add_data(data:list or list of data:list(list))")
            return None
        n1 = self.data[data_name_1]["n"]
        S1 = self.data[data_name_1]["S2"]
        mean1 = self.data[data_name_1]["mean"]
        n2 = self.data[data_name_2]["n"]
        S2 = self.data[data_name_2]["S2"]
        mean2 = self.data[data_name_2]["mean"]
        interval_name = f"{(100 - 100*alpha):.3f}%"
        analysis_name = f"ISEV_{data_name_1}_{data_name_2}"

        S2_pooled = ((n1 - 1)*S1 + (n2 - 1)*S2)/(n1 + n2 - 2)
        freedom = n1 + n2 - 2
        SE = S2_pooled * np.sqrt((1/n1) + (1/n2))
        confidence_interval = [mean1 - mean2 - stats.t.ppf(1-alpha/2, freedom) * SE, mean1 - mean2 + stats.t.ppf(1-alpha/2, freedom) * SE]

        if analysis_name in self.analysis.keys():
            if not interval_name in self.analysis[analysis_name]["confidence_intervals"].keys():
                self.analysis[analysis_name]["confidence_intervals"][interval_name] = confidence_interval
                return self.analysis[analysis_name]
            else:
                print(f"[INFO] There is already exactly a same analysis in the data.")
                return self.analysis[analysis_name]
        else:
            self.analysis[analysis_name] = {"Y1-Y2": mean1-mean2,
                                            "S2_pooled": S2_pooled,
                                            "freedom": freedom,
                                            "SE": SE,
                                            "half-lenght": stats.t.ppf(1-alpha/2, freedom) * SE,
                                            "confidence_intervals":{interval_name: confidence_interval}}
        
        if print_:
            print(f"[RESULT] [{float(confidence_interval[0]):4f}, {float(confidence_interval[1]):4f}]")
        return self.analysis[analysis_name]

    def independent_sampling(self, data_name_1:str, data_name_2:str, alpha:float, print_:bool=True):
        if len(self.data) < 2:
            print(f"[ERROR] Please add the second data to the object first. Use this: object.add_data(data:list or list of data:list(list))")
            return None
        n1 = self.data[data_name_1]["n"]
        S1 = self.data[data_name_1]["S2"]
        mean1 = self.data[data_name_1]["mean"]
        n2 = self.data[data_name_2]["n"]
        S2 = self.data[data_name_2]["S2"]
        mean2 = self.data[data_name_2]["mean"]
        interval_name = f"{(100 - 100*alpha):.3f}%"
        analysis_name = f"IS_{data_name_1}_{data_name_2}"
        SE = np.sqrt(S1/n1 + S2/n2)
        freedom = int((S1/n1 + S2/n2)**2 / (((S1/n1)**2)/(n1-1) + ((S2/n2)**2)/(n2-1)))
        confidence_interval = [mean1 - mean2 - stats.t.ppf(1-alpha/2, freedom) * SE, mean1 - mean2 + stats.t.ppf(1-alpha/2, freedom) * SE]

        if analysis_name in self.analysis.keys():
            if not interval_name in self.analysis[analysis_name]["confidence_intervals"].keys():
                self.analysis[analysis_name]["confidence_intervals"][interval_name] = confidence_interval
                return self.analysis[analysis_name]
            else:
                print(f"[INFO] There is already exactly a same analysis in the data.")
                return self.analysis[analysis_name]
        else:
            self.analysis[analysis_name] = {"Y1-Y2": mean1-mean2,
                                            "freedom": freedom,
                                            "SE": SE,
                                            "half-lenght": stats.t.ppf(1-alpha/2, freedom) * SE,
                                            "confidence_intervals":{interval_name: confidence_interval}}
            
        if print_:
            print(f"[RESULT] [{float(confidence_interval[0]):4f}, {float(confidence_interval[1]):4f}]")
        return self.analysis[analysis_name]

    def common_random_numbers(self, data_name_1:str, data_name_2:str, alpha:float, print_:bool=True):
        if not self.data[data_name_1]["n"] == self.data[data_name_2]["n"]:
            print(f"[ERROR] Two data with different size. Not possable to use this method with different size of sample.")
            return
        data1 = self.data[data_name_1]["data"]
        data2 = self.data[data_name_2]["data"]
        n = self.data[data_name_1]["n"]
        mean1 = self.data[data_name_1]["mean"]
        mean2 = self.data[data_name_2]["mean"]
        interval_name = f"{(100 - 100*alpha):.3f}%"
        analysis_name = f"CRN_{data_name_1}_{data_name_2}"

        delta = data1 - data2
        D = delta.sum() / n
        S2 = ((delta - D)**2).sum() / (n-1)
        SE = np.sqrt(S2)/np.sqrt(n)
        freedom = n-1
        confidence_interval = [mean1 - mean2 - stats.t.ppf(1-alpha/2, freedom) * SE, mean1 - mean2 + stats.t.ppf(1-alpha/2, freedom) * SE]

        if analysis_name in self.analysis.keys():
            if not interval_name in self.analysis[analysis_name]["confidence_intervals"].keys():
                self.analysis[analysis_name]["confidence_intervals"][interval_name] = confidence_interval
                return self.analysis[analysis_name]
            else:
                print(f"[INFO] There is already exactly a same analysis in the data.")
                return self.analysis[analysis_name]
        else:
            self.analysis[analysis_name] = {"Y1-Y2": mean1-mean2,
                                            "D_bar": D,
                                            "freedom": freedom,
                                            "SE": SE,
                                            "half-lenght": stats.t.ppf(1-alpha/2, freedom) * SE,
                                            "confidence_intervals":{interval_name: confidence_interval}}
            
        if print_:
            print(f"[RESULT] [{float(confidence_interval[0]):4f}, {float(confidence_interval[1]):4f}]")
        return self.analysis[analysis_name]

    def multiple_comparison_with_standard(self, data1:str, data_list:list, alpha:float, method:str):
        data = {}
        k = len(data_list)
        print(f"[INFO] We assume {100*(1-alpha/k):.2f}% confidens to each comparison to get the total of {100*(1-alpha):.2f} according to Bon Ferroni inequality.")
        columns = [f"{data1} - Yi", "Interval"]
        if method == "ISEV":
            for data_name in data_list:
                info_ = self.independent_sampling_with_equal_variance(data1, data_name, alpha=alpha/k, print_=False)
                interval_name = f"{(100 - 100*alpha/k):.3f}%"
                data[data_name] = [round(info_["Y1-Y2"], self.decimal), (round(float(info_["confidence_intervals"][interval_name][0]), self.decimal), round(float(info_["confidence_intervals"][interval_name][1]), self.decimal))]
            data_structure = pd.DataFrame.from_dict(data, orient="index", columns=columns)
            self.comparisons["M.C. with a standard_ISEV"] = data_structure
            return data_structure
        elif method == "IS":
            for data_name in data_list:
                info_ = self.independent_sampling(data1, data_name, alpha=alpha/k, print_=False)
                interval_name = f"{(100 - 100*alpha/k):.3f}%"
                data[data_name] = [round(info_["Y1-Y2"], self.decimal), (round(float(info_["confidence_intervals"][interval_name][0]), self.decimal), round(float(info_["confidence_intervals"][interval_name][1]), self.decimal))]
            data_structure = pd.DataFrame.from_dict(data, orient="index", columns=columns)
            self.comparisons["M.C. with a standard_IS"] = data_structure
            return data_structure
        elif method == "CRN":
            for data_name in data_list:
                info_ = self.common_random_numbers(data1, data_name, alpha=alpha/k, print_=False)
                interval_name = f"{(100 - 100*alpha/k):.3f}%"
                data[data_name] = [round(info_["Y1-Y2"], self.decimal), (round(float(info_["confidence_intervals"][interval_name][0]), self.decimal), round(float(info_["confidence_intervals"][interval_name][1]), self.decimal))]
            data_structure = pd.DataFrame.from_dict(data, orient="index", columns=columns)
            self.comparisons["M.C. with a standard_CRN"] = data_structure
            return data_structure

    def multiple_comparison_All(self, data_list:list, alpha:float, method:str):
        data_mean = {}
        data_intervals = {}
        k = len(data_list) * (len(data_list)-1) / 2
        print(f"[INFO] We assume {100*(1-alpha/k):.2f}% confidens to each comparison to get the total of {100*(1-alpha):.2f} according to Bon Ferroni inequality.")
        columns = data_list
        if method == "ISEV":
            for data1 in data_list:
                data_mean[data1] = []
                data_intervals[data1] = []
                for data2 in data_list:
                    if data1 != data2:
                        info_ = self.independent_sampling_with_equal_variance(data1, data2, alpha=alpha/k, print_=False)
                        interval_name = f"{(100 - 100*alpha/k):.3f}%"
                        data_mean[data1].append(round(info_["Y1-Y2"], self.decimal))
                        data_intervals[data1].append((round(float(info_["confidence_intervals"][interval_name][0]), self.decimal), round(float(info_["confidence_intervals"][interval_name][1]), self.decimal)))
                    else:
                        data_mean[data1].append(0)
                        data_intervals[data1].append(0)
            data_structure_mean = pd.DataFrame.from_dict(data_mean, orient="index", columns=columns)
            data_structure_intervals = pd.DataFrame.from_dict(data_intervals, orient="index", columns=columns)
            self.comparisons["M.C. All_ISEV"] = (data_structure_mean, data_structure_intervals)
            return (data_structure_mean, data_structure_intervals)
        elif method == "IS":
            for data1 in data_list:
                data_mean[data1] = []
                data_intervals[data1] = []
                for data2 in data_list:
                    if data1 != data2:
                        info_ = self.independent_sampling(data1, data2, alpha=alpha/k, print_=False)
                        interval_name = f"{(100 - 100*alpha/k):.3f}%"
                        data_mean[data1].append(round(info_["Y1-Y2"], self.decimal))
                        data_intervals[data1].append((round(float(info_["confidence_intervals"][interval_name][0]), self.decimal), round(float(info_["confidence_intervals"][interval_name][1]), self.decimal)))
                    else:
                        data_mean[data1].append(0)
                        data_intervals[data1].append(0)
            data_structure_mean = pd.DataFrame.from_dict(data_mean, orient="index", columns=columns)
            data_structure_intervals = pd.DataFrame.from_dict(data_intervals, orient="index", columns=columns)
            self.comparisons["M.C. All_IS"] = (data_structure_mean, data_structure_intervals)
            return (data_structure_mean, data_structure_intervals)
        elif method == "CRN":
            for data1 in data_list:
                data_mean[data1] = []
                data_intervals[data1] = []
                for data2 in data_list:
                    if data1 != data2:
                        info_ = self.common_random_numbers(data1, data2, alpha=alpha/k, print_=False)
                        interval_name = f"{(100 - 100*alpha/k):.3f}%"
                        data_mean[data1].append(round(info_["Y1-Y2"], self.decimal))
                        data_intervals[data1].append((round(float(info_["confidence_intervals"][interval_name][0]), self.decimal), round(float(info_["confidence_intervals"][interval_name][1]), self.decimal)))
                    else:
                        data_mean[data1].append(0)
                        data_intervals[data1].append(0)
            data_structure_mean = pd.DataFrame.from_dict(data_mean, orient="index", columns=columns)
            data_structure_intervals = pd.DataFrame.from_dict(data_intervals, orient="index", columns=columns)
            self.comparisons["M.C. All_CRN"] = (data_structure_mean, data_structure_intervals)
            return (data_structure_mean, data_structure_intervals)
        
    def multiple_comparison_with_best(self, data_list:list, alpha:float, epsilon:float, more_less:str, step:int):
        if step == 1:
            r0 = self.data[data_list[0]]["n"]
            r = r0
            for data in data_list:
                if not r0 == self.data[data]["n"]:
                    print(f"[ERROR] All data should have the same sample size. Correct them please.")
                    return None
            k = len(data_list)
            t_value = stats.t.ppf(1- alpha/(2*(k-1)), r0-1)
            for data1_name in data_list:
                data1 = self.data[data1_name]["data"]
                for data2_name in data_list:
                    data2 = self.data[data2_name]["data"]
                    delta = data1 - data2
                    D = delta.sum() / r0
                    S2 = ((delta - D)**2).sum() / (r0-1)
                    r1 = int(S2*(t_value**2)/(epsilon**2))
                    if r1 > r:
                        r = r1
            print(f"[INFO] Great. Now please add up to {r} sample data. So you need {r-r0} more runs. Refine the data and run it with 2ed step parameter :)")

        elif step == 2:
            data_s = {}
            columns = ["mean(Y_bar)", "Yi - Yi* + e"]
            if more_less == "more":
                best = -float("inf")
                best_data = 0
                for data in data_list:
                    if self.data[data]["mean"] > best:
                        best = self.data[data]["mean"]
                        best_data = data
            elif more_less == "less":
                best = float("inf")
                best_data = 0
                for data in data_list:
                    if self.data[data]["mean"] < best:
                        best = self.data[data]["mean"]
                        best_data = data
            print(best_data)

            if more_less == "more":
                print("We need to see all negative numbers.")
                for data in data_list:
                    data_s[data] = [round(self.data[data]["mean"], self.decimal), round(self.data[data]["mean"] - best + epsilon, self.decimal)]
                
            elif more_less == "less":
                print("We need to see all positive numbers.")
                for data in data_list:
                    data_s[data] = [round(self.data[data]["mean"], self.decimal), round(self.data[data]["mean"] - best - epsilon, self.decimal)]
            data_structure = pd.DataFrame.from_dict(data_s, orient="index", columns=columns)
            self.comparisons["M.C. with the best"] = data_structure
            return data_structure


