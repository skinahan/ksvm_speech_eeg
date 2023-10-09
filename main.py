# main.py
import seaborn as sns
from data_analysis import *

sns.set_theme(style="darkgrid")

if __name__ == "__main__":
    seed = 42
    set_seed(42)
    global c_POW
    c_POW = 3
    scaleTransform = True
    combined_pkl_name = f'./optimal_gam_Increasing_C_all_seeds_3_trimmed.pkl'
    plot_opt_gam_increasing_C(combined_pkl_name)

    trimmed_combined_name = f'./optimal_c_Increasing_Gamma_all_seeds_3_trimmed.pkl'
    plot_opt_c_increasing_Gamma(trimmed_combined_name)

    run_sanity_checks(patho=False)
    run_sanity_checks(patho=True)
    remove_subjects()
    run_proportion_tests()
    plot_all_seeds()
    c_pow_range = [1, 3, 5, 7]
    for c in c_pow_range:
        c_POW = c
        valid_out_name = f'./{c_POW}_nestedCV_all_seeds_trimmed.pkl'
        plot_params_alternate(valid_out_name)
    print("\n\ndone.")