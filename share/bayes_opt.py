import GPy
import numpy as np
import pandas as pd
from GPy import kern as gp_kern
from scipy.stats import norm


class Share:
    def __init__(self, all_env, df_with_energy, df_without_energy):
        self.all_env = all_env
        self.log_path = (
            f"{all_env['LOCAL_DIR']}{all_env['WORK_DIR']}/calc/results/bo-py.log"
        )
        self.num_candidates = 4
        self.INIT_NUM = int(all_env["INIT_NUM"])
        self.BO_NUM = int(all_env["BO_NUM"])
        self.df_with_energy = df_with_energy
        self.df_without_energy = df_without_energy

    def log_message(self, message):
        with open(f"{self.log_path}", "a") as log_file:
            log_file.write(message + "\n")

    def acquisition_lcb(self, mean, std, kappa):
        return mean - kappa * std

    def acquisition_EI(self, mean, std, min_value, xi=0.001):
        imp = min_value - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        return ei

    # log_message("Starting Python script")

    def main(self):
        # print("w/ ene", self.df_with_energy, "\n", "w/o ene", self.df_without_energy)
        if len(self.df_with_energy) < self.INIT_NUM:
            calc_dir = "/init"

            if len(self.df_with_energy) > self.INIT_NUM - self.num_candidates:
                shortage_num = self.INIT_NUM - len(self.df_with_energy)
                head_ids = self.df_without_energy.head(shortage_num).index
                self.log_message("Python script finished")
            else:
                head_ids = self.df_without_energy.head(self.num_candidates).index
                self.log_message("Python script finished")

            return list(head_ids), calc_dir

        else:
            # 特徴量とターゲット変数の分割
            X = self.df_with_energy.iloc[:, :-1]
            y = self.df_with_energy.iloc[:, -1:]

            # ガウス過程回帰モデルの設定と訓練
            num = X.shape[1]
            kernel = gp_kern.RBF(num) * gp_kern.Bias(num) + gp_kern.Linear(
                num
            ) * gp_kern.Bias(num)
            model = GPy.models.GPRegression(
                X.values, y.values, kernel=kernel, normalizer=True
            )
            model.optimize()

            # 各特徴量での獲得関数の計算
            min_value = min(self.df_with_energy["result"])
            means = []
            stds = []
            acs_ei = []
            acs_lcb = []

            itera = self.df_without_energy.values
            for item in itera:
                mean, val = model.predict(np.array(item).reshape(1, -1))
                std = np.sqrt(val)

                ac_ei = self.acquisition_EI(mean, std, min_value)
                ac_lcb = self.acquisition_lcb(mean, std, 7)

                means.append(mean.flatten()[0])
                stds.append(std.flatten()[0])
                acs_ei.append(ac_ei.flatten()[0])
                acs_lcb.append(ac_lcb.flatten()[0])

            # 提案された特徴量を Pandas データフレームとして出力
            result_df = pd.DataFrame(self.df_without_energy)
            result_df = result_df.assign(mean=means, std=stds, EI=acs_ei, LCB=acs_lcb)

            if (
                len(self.df_with_energy)
                > self.INIT_NUM + self.BO_NUM - self.num_candidates
            ):
                shortage_num = self.INIT_NUM + self.BO_NUM - len(self.df_with_energy)
            else:
                shortage_num = self.num_candidates
            # 候補の ID を出力
            result_df_sort = result_df.sort_values("LCB", ascending=True)
            self.log_message(f"Sorted result: {result_df_sort.head(shortage_num*2)}")

            candidates = result_df_sort.head(shortage_num).index
            # print("\n".join(map(str, candidates)))
            self.log_message("Python script finished")
            calc_dir = "/BO"
            return list(candidates), calc_dir


def make_candidates(all_env, df_with_energy, df_without_energy):
    share_instance = Share(all_env, df_with_energy, df_without_energy)
    return share_instance.main()
