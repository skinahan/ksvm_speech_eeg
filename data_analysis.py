# data_analysis.py
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.special
from model_training import *
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def set_scale_transform():
    global scaleTransform
    scaleTransform = True


# Data Analysis and Visualization Module


def run_sanity_checks(patho):
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    global c_POW
    if patho:
        index_file = f'{base_dir}\\S_index.json'

    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    for entry in index:
        subject = entry['Subject']
        sanity_test(subject)


def filter_and_combine(dataframes, threshold):
    filtered_dfs = []

    # Filter out entries smaller than the threshold for each dataframe
    for df in dataframes:
        filtered_df = df[df > threshold]
        filtered_dfs.append(filtered_df)

    # Combine the filtered dataframes into a single output dataframe
    combined_df = pd.concat(filtered_dfs).fillna(0.5)

    # Calculate the average of the equivalent entries in the input set
    output_df = combined_df.groupby(combined_df.index).mean()

    return output_df


def read_file_into_list(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def run_proportion_tests():
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    patho_index_file = f'{base_dir}\\S_index.json'

    f = open(index_file)
    index = json.load(f)['index']
    f.close()

    f2 = open(patho_index_file)
    index2 = json.load(f2)['index']
    f2.close()

    pkl1 = f'./{seed}_individual_nestedCV_results_{c_POW}_scaled_trimmed.pkl'
    pkl2 = f'./{seed}_individual_nestedCV_results_patho_{c_POW}_scaled_trimmed.pkl'

    non_patho_df = pd.read_pickle(pkl1)
    patho_df = pd.read_pickle(pkl2)

    # get the unique values of C and gamma:
    uniq_Cs = np.unique(non_patho_df.C.values)
    uniq_Gs = np.unique(non_patho_df.gam.values)

    final_df = pd.DataFrame()

    valid_threshold = 0.65

    non_patho_pvts = []
    file_path = 'rm_subjs.txt'  # Replace with the path to your .txt file
    subjects_to_remove = read_file_into_list(file_path)

    for entry in index:
        subject = entry['Subject']
        if subject in subjects_to_remove:
            continue
        df_loc = non_patho_df[non_patho_df['Subject'] == subject]
        pvt = df_loc.pivot("gam", "C", "Accuracy")
        non_patho_pvts.append(pvt)
        prop_exceed = calculate_proportion(pvt, valid_threshold)
        print(subject)
        print("Percentage of validation accuracies above threshold: " + str(prop_exceed))

        new_row = {'Subject': [subject],
                   'Subject_Type': ["Control"],
                   'Threshold_Prop': [prop_exceed]}
        df2 = pd.DataFrame.from_dict(new_row)
        final_df = pd.concat([final_df, df2], sort=False)

    patho_pvts = []

    for entry in index2:
        subject = entry['Subject']
        if subject in subjects_to_remove:
            continue
        df_loc = patho_df[patho_df['Subject'] == subject]
        pvt = df_loc.pivot("gam", "C", "Accuracy")
        patho_pvts.append(pvt)
        prop_exceed = calculate_proportion(pvt, valid_threshold)
        print(subject)
        print("Percentage of validation accuracies above threshold: " + str(prop_exceed))
        new_row = {'Subject': [subject],
                   'Subject_Type': ["Stutter"],
                   'Threshold_Prop': [prop_exceed]}
        df2 = pd.DataFrame.from_dict(new_row)
        final_df = pd.concat([final_df, df2], sort=False)

    avg_non_patho_pvt = filter_and_combine(non_patho_pvts, valid_threshold)  # .fillna(0.5)
    avg_patho_pvt = filter_and_combine(patho_pvts, valid_threshold)  # .fillna(0.5)

    pkl_name = f'./{seed}_prop_threshold_{c_POW}.pkl'
    final_df.to_pickle(pkl_name)
    remove_subjects()
    pkl_name = f'./{seed}_prop_threshold_{c_POW}_trimmed.pkl'
    test_mean_significance(pkl_name, 'Threshold_Prop')

    title = "ANS Hyperparameter Search"
    g = sns.heatmap(avg_non_patho_pvt, xticklabels=uniq_Cs, yticklabels=uniq_Gs)
    g.set_xticklabels([label_fmt(label.get_text()) for label in g.get_xticklabels()])
    g.set_yticklabels([label_fmt(label.get_text()) for label in g.get_yticklabels()])
    for label in g.get_xticklabels()[::2]:
        label.set_visible(False)
    for label in g.get_yticklabels()[::2]:
        label.set_visible(False)

    g.set_xlabel("C", fontsize=16)
    g.set_ylabel("Gamma", fontsize=16)
    g.set_title(title, fontsize=20)
    plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(left=0.20)
    plt.savefig(f'./results/figures/heatmaps/{seed}_{title}_{c_POW}.png', dpi=300)
    plt.close()

    patho_title = "AWS Hyperparameter Search"
    g2 = sns.heatmap(avg_patho_pvt, xticklabels=uniq_Cs, yticklabels=uniq_Gs)
    g2.set_xticklabels([label_fmt(label.get_text()) for label in g2.get_xticklabels()])
    g2.set_yticklabels([label_fmt(label.get_text()) for label in g2.get_yticklabels()])
    for label in g2.get_xticklabels()[::2]:
        label.set_visible(False)
    for label in g2.get_yticklabels()[::2]:
        label.set_visible(False)

    g2.set_xlabel("C", fontsize=16)
    g2.set_ylabel("Gamma", fontsize=16)
    g2.set_title(patho_title, fontsize=20)
    plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(left=0.20)
    plt.savefig(f'./results/figures/heatmaps/{seed}_{patho_title}_{c_POW}.png', dpi=300)
    plt.close()
    n = len(uniq_Gs)
    rev_gs = [uniq_Gs[n:-i - 2:-1] for i in range(n)]

    # Create contour lines to represent accuracy levels for set 1
    contours1 = plt.contour(uniq_Cs, rev_gs[-1], avg_non_patho_pvt, levels=5, colors='k', linestyles='dashed')
    plt.clabel(contours1, inline=True, fontsize=8, colors='k')

    # Create contour lines to represent accuracy levels for set 2
    contours2 = plt.contour(uniq_Cs, rev_gs[-1], avg_patho_pvt, levels=5, colors='r', linestyles='solid')
    plt.clabel(contours2, inline=True, fontsize=8, colors='r')

    # Set labels and title
    plt.xlabel('C', fontsize=16)
    plt.ylabel('Gamma', fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Validation Accuracies Contour Plot', fontsize=20)

    plt.savefig(f'./results/figures/heatmaps/{seed}_Validation Accuracies Contour Plot_{c_POW}.png', dpi=300)
    plt.close()

    print("Done.")


def extract_max_plot_results(patho=True):
    # load pickle
    pkl_name = f'./individual_nestedCV_results_{c_POW}.pkl'
    if patho:
        pkl_name = f'./individual_nestedCV_results_patho_{c_POW}.pkl'
    df = pd.read_pickle(pkl_name)
    # sort based on accuracy
    df = df.sort_index(1)
    # drop duplicates, specify Subject
    # keep first (last?)
    df = df.drop_duplicates('Subject', keep='last')
    # now, we have max accuracy achieved per-subject
    sns.relplot(data=df, x="Subject", y="Accuracy", kind="line")
    plt.show()


def combine_df_nestedCV(scaleTransform):
    pkl1 = f'./{seed}_individual_nestedCV_results_{c_POW}'
    pkl2 = f'./{seed}_individual_nestedCV_results_patho_{c_POW}'

    if scaleTransform:
        pkl1 = f'{pkl1}_scaled.pkl'
        pkl2 = f'{pkl2}_scaled.pkl'

    combine_df_results(pkl1, pkl2)


def combine_df_log_regression():
    combine_df_results(f'./individual_log_regression_results.pkl', f'./individual_log_regression_results_patho.pkl')


def test_mean_significance(pkl_name, attr_name="Accuracy"):
    df = pd.read_pickle(pkl_name)
    # get accuracies of people who stutter
    cond1 = (df['Subject_Type'] == "Stutter")
    pws_acc = df[cond1][attr_name].values
    # get accuracies of people who do not stutter
    cond2 = (df['Subject_Type'] == "Control")
    control_acc = df[cond2][attr_name].values
    mean_AWS = np.mean(pws_acc)
    mean_AWDS = np.mean(control_acc)
    std_dev_AWS = np.std(pws_acc)
    std_dev_AWDS = np.std(control_acc)
    global c_POW
    print(f"Testing if mean difference is statistically significant [10^-{c_POW}, 10^{c_POW}]")
    print("Attribute: %s" % attr_name)

    print(f"AWS Mean (Std): {mean_AWS} ({std_dev_AWS})")
    print(f"ANS Mean (Std): {mean_AWDS} ({std_dev_AWDS})")

    t_stat, p_val = stats.ttest_ind(control_acc, pws_acc, equal_var=False)
    print('t_stat (p_val): %.3f (%.3f)' % (t_stat, p_val))
    if p_val < 0.05:
        print("SIGNIFICANT RESULT")
    else:
        print("RESULT NOT SIGNIFICANT")


def box_plot(pkl_name, title=None, logit=False, remove_dupes=True):
    df = pd.read_pickle(pkl_name)

    if title is None:
        title = ""

    def return_singleton(x):
        return x[0]

    if isinstance(df["gam"].values[0], list):
        df["gam"] = df["gam"].map(return_singleton)

    def logit_transform(x):
        return scipy.special.logit(x)

    if logit:
        df['Accuracy'] = df['Accuracy'].apply(logit_transform)

    if remove_dupes:
        # sort based on accuracy, gam, C
        df = df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
        # drop duplicates, specify Subject
        df = df.drop_duplicates('Subject', keep='first')
    df = df.sort_values(by=['Subject'])
    sns.set(rc={"figure.figsize": (10, 4)})
    g = sns.boxplot(data=df, x="Subject", y="Accuracy", hue="Subject_Type").set(title=title)
    max_acc = np.max(df["Accuracy"].values)
    min_acc = np.min(df["Accuracy"].values)
    subjects = np.unique(df["Subject"].values)
    renamed_subjects = [get_new_subject_name(subj) for subj in subjects]

    g[0].axes.set_xticklabels(renamed_subjects, rotation=45)
    g[0].axes.figure.subplots_adjust(bottom=0.2)
    # g.despine(left=True)
    if max_acc == 1.0:
        max_acc = 0.9
    # plt.ylim(min_acc - 0.1, max_acc + 0.1)
    # plt.show()
    if remove_dupes:
        plt.savefig(f'./results/figures/{seed}_{title}_{c_POW}_box.png', dpi=300)
    else:
        plt.savefig(f'./results/figures/{title}_box.png', dpi=300)
    plt.close()


def bar_chart(pkl_name, title=None, logit=False, remove_dupes=True):
    df = pd.read_pickle(pkl_name)

    if title is None:
        title = ""

    def return_singleton(x):
        return x[0]

    if isinstance(df["gam"].values[0], list):
        df["gam"] = df["gam"].map(return_singleton)

    def logit_transform(x):
        return scipy.special.logit(x)

    if logit:
        df['Accuracy'] = df['Accuracy'].apply(logit_transform)

    if remove_dupes:
        # sort based on accuracy, gam, C
        df = df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
        # drop duplicates, specify Subject
        df = df.drop_duplicates('Subject', keep='first')
    df = df.sort_values(by=['Subject'])
    g = sns.catplot(data=df, kind="bar", alpha=.6,
                    x="Subject", y="Accuracy", hue="Subject_Type", height=6, aspect=2.5)
    max_acc = np.max(df["Accuracy"].values)
    min_acc = np.min(df["Accuracy"].values)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title(title)
    g.despine(left=True)
    if max_acc == 1.0:
        max_acc = 0.9
    plt.ylim(min_acc - 0.1, max_acc + 0.1)
    # plt.show()
    if remove_dupes:
        plt.savefig(f'./results/figures/{seed}_{title}_{c_POW}.png', dpi=300)
    else:
        plt.savefig(f'./results/figures/{title}.png', dpi=300)
    plt.close()


def extract_avg_plot_results(patho=False):
    # load pickle
    pkl_name = f'./individual_nestedCV_results_{c_POW}.pkl'
    if patho:
        pkl_name = f'./individual_nestedCV_results_patho_{c_POW}.pkl'
    df = pd.read_pickle(pkl_name)
    # get the unique subjects
    unique_subjs = df['Subject'].unique()
    # drop duplicates, specify Subject
    subj_avgs = []
    for subj in unique_subjs:
        subj_values = df.loc[df['Subject'] == subj]
        avg = np.mean(subj_values['Accuracy'])
        subj_avgs.append(avg)
    # now, we have avg accuracy per-subject
    data = {'Subject': unique_subjs,
            'Average': subj_avgs}
    df2 = pd.DataFrame(data)
    # sns.relplot(data=df2, x="Subject", y="Average", kind="line")
    sns.relplot(data=df, x="Subject", y="Accuracy", kind="line", height=6, aspect=2.5)
    plt.show()


def group_avg_bar_chart_nestedCV(pkl_name, logit=False):
    if logit:
        group_avg_bar_chart(pkl_name, title=f"Logit Group Average Comparison [10^-{c_POW}, 10^{c_POW}]", logit=True)
    else:
        group_avg_bar_chart(pkl_name, title=f"Group Average Comparison [10^-{c_POW}, 10^{c_POW}]", logit=False)


def group_avg_box_plot(pkl_name, title=None, logit=False, remove_dupes=True):
    df = pd.read_pickle(pkl_name)

    def logit_transform(x):
        return scipy.special.logit(x)

    def return_singleton(x):
        return x[0]

    if isinstance(df["gam"].values[0], list):
        df["gam"] = df["gam"].map(return_singleton)

    if title is None:
        title = ""

    if remove_dupes:
        # sort based on accuracy, gam, C
        df = df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
        # drop duplicates, specify Subject
        df = df.drop_duplicates('Subject', keep='first')
        df = df.sort_values(by=['Subject'])
    df = df.sort_values(by=['Subject'])
    sns.set(rc={"figure.figsize": (8, 4)})
    g = sns.boxplot(data=df, x="Subject_Type", y="Accuracy").set(title=title)
    g[0].axes.figure.subplots_adjust(bottom=0.2)
    max_acc = np.max(df["Accuracy"].values)
    min_acc = np.min(df["Accuracy"].values)
    if max_acc == 1.0:
        max_acc = 0.9
    # plt.ylim(min_acc - 0.1, max_acc + 0.1)
    # g.despine(left=True)
    # plt.show()
    if remove_dupes:
        plt.savefig(f'./results/figures/{seed}_{title}_{c_POW}_box.png', dpi=300)
    else:
        plt.savefig(f'./results/figures/{title}_box.png', dpi=300)
    plt.close()


def group_avg_bar_chart(pkl_name, title=None, logit=False, remove_dupes=True):
    df = pd.read_pickle(pkl_name)

    def logit_transform(x):
        return scipy.special.logit(x)

    def return_singleton(x):
        return x[0]

    if isinstance(df["gam"].values[0], list):
        df["gam"] = df["gam"].map(return_singleton)

    if title is None:
        title = ""

    if logit:
        # get accuracies of people who stutter
        cond1 = (df['Subject_Type'] == "Stutter")
        pws_acc = df[cond1].Accuracy.values
        # get accuracies of people who do not stutter
        cond2 = (df['Subject_Type'] == "Control")
        control_acc = df[cond2].Accuracy.values
        pws_avg = np.mean(pws_acc)
        control_avg = np.mean(control_acc)

        pws_avg = logit_transform(pws_avg)
        control_avg = logit_transform(control_avg)

        data = {'Subject_Type': ['Control', 'Stutter'],
                'Mean_Accuracy_logit': [control_avg, pws_avg]}
        df2 = pd.DataFrame(data)
        g = sns.catplot(data=df2, kind="bar", palette="bright", alpha=.6,
                        x="Subject_Type", y="Mean_Accuracy_logit", height=6, aspect=2.5)
        g.fig.subplots_adjust(top=.95)
        g.ax.set_title(title)
        g.despine(left=True)
        plt.ylim(2.0, 3.0)
        plt.show()
    else:
        if remove_dupes:
            # sort based on accuracy, gam, C
            df = df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
            # drop duplicates, specify Subject
            df = df.drop_duplicates('Subject', keep='first')
            df = df.sort_values(by=['Subject'])

        g = sns.catplot(data=df, kind="bar", palette="bright", alpha=.6,
                        x="Subject_Type", y="Accuracy", height=6, aspect=2.5)
        max_acc = np.max(df["Accuracy"].values)
        min_acc = np.min(df["Accuracy"].values)
        if max_acc == 1.0:
            max_acc = 0.9
        plt.ylim(min_acc - 0.1, max_acc + 0.1)
        # plt.ylim(min_acc - 0.1, 0.1 + max_acc)
        g.fig.subplots_adjust(top=.95)
        g.ax.set_title(title)
        g.despine(left=True)
        # plt.show()
        if remove_dupes:
            plt.savefig(f'./results/figures/{seed}_{title}_{c_POW}.png', dpi=300)
        else:
            plt.savefig(f'./results/figures/{title}.png', dpi=300)
        plt.close()


def plot_contour_with_threshold(param1_values, param2_values, accuracies1, accuracies2=None, threshold=0.65):
    """
    Plot a contour plot of validation accuracies with optional second set of data and a threshold.

    Parameters:
        param1_values (array-like): Values for hyperparameter 1.
        param2_values (array-like): Values for hyperparameter 2.
        accuracies1 (array-like): Validation accuracies for the first set of data.
        accuracies2 (array-like, optional): Validation accuracies for the second set of data (default: None).
        threshold (float, optional): Threshold for accuracy values. Values below this threshold will be set to 0.50 (default: 0.50).

    Returns:
        None (displays the plot).
    """
    # Apply the threshold to accuracies1
    accuracies1 = np.where(accuracies1 < threshold, 0.50, accuracies1)

    # Create contour lines to represent accuracy levels for set 1
    contours1 = plt.contour(param1_values, param2_values, accuracies1, levels=5, colors='k', linestyles='dashed')
    plt.clabel(contours1, inline=True, fontsize=8, colors='k')

    if accuracies2 is not None:
        # Apply the threshold to accuracies2 if provided
        accuracies2 = np.where(accuracies2 < threshold, 0.50, accuracies2)

        # Create contour lines to represent accuracy levels for set 2
        contours2 = plt.contour(param1_values, param2_values, accuracies2, levels=5, colors='r', linestyles='solid')
        plt.clabel(contours2, inline=True, fontsize=8, colors='r')

    # Set labels and title
    plt.xlabel('Hyperparameter 1', fontsize=16)
    plt.ylabel('Hyperparameter 2', fontsize=16)
    plt.title('Validation Accuracies Contour Plot', fontsize=20)

    # Show the plot
    plt.show()


def combine_df_results(pkl1_name, pkl2_name):
    df1 = pd.read_pickle(pkl1_name)
    non_stut_labels = ["Control" for row in df1['Subject']]
    df1["Subject_Type"] = non_stut_labels
    df2 = pd.read_pickle(pkl2_name)
    stut_labels = ["Stutter" for row in df2['Subject']]
    df2["Subject_Type"] = stut_labels
    df = pd.concat([df1, df2], sort=False)
    name_no_ext = os.path.splitext(pkl1_name)[0]
    final_pkl_name = name_no_ext + '_combined.pkl'
    df.to_pickle(final_pkl_name)


def combine_df_results_defaultParams(scaleTransform):
    if scaleTransform:
        combine_df_results(f'./{seed}_individual_results_scaled.pkl', f'./{seed}_individual_results_patho_scaled.pkl')
    else:
        combine_df_results('./individual_results.pkl', './individual_results_patho.pkl')


def combine_df_results_increasingC(scaleTransform):
    pkl1 = f'./{seed}_individual_variableC_results'
    pkl2 = f'./{seed}_individual_variableC_results_patho'
    if scaleTransform:
        pkl1 = f'{pkl1}_scaled'
        pkl2 = f'{pkl2}_scaled'
    pkl1 = f'{pkl1}.pkl'
    pkl2 = f'{pkl2}.pkl'
    combine_df_results(pkl1, pkl2)


def combine_df_results_increasingGamma(scaleTransform):
    pkl1 = f'./{seed}_individual_variableGamma_results'
    pkl2 = f'./{seed}_individual_variableGamma_results_patho'
    if scaleTransform:
        pkl1 = f'{pkl1}_scaled'
        pkl2 = f'{pkl2}_scaled'
    pkl1 = f'{pkl1}.pkl'
    pkl2 = f'{pkl2}.pkl'
    combine_df_results(pkl1, pkl2)


# calculate average params for both groups
def get_avg_params(pkl_name):
    df = pd.read_pickle(pkl_name)
    # get only people who stutter
    cond1 = (df['Subject_Type'] == "Stutter")
    pws_C = df[cond1].C.values
    pws_Gamma = df[cond1].gam.values
    avg_pws_C = np.mean(pws_C)
    avg_pws_gam = np.mean(pws_Gamma)
    # get only people who do not stutter
    cond2 = (df['Subject_Type'] == "Control")
    control_C = df[cond2].C.values
    pws_Gamma = df[cond2].gam.values
    avg_control_C = np.mean(control_C)
    avg_control_gamma = np.mean(pws_Gamma)

    print(f'Control params: C: {avg_control_C}, gamma: {avg_control_gamma}')
    print(f'PWS params: C: {avg_pws_C}, gamma: {avg_pws_gam}')


# plot 2D distribution of selected params across both groups
def plot_params_2D(pkl_name):
    df = pd.read_pickle(pkl_name)
    filt = False
    if filt:
        # sort based on accuracy, gam, C
        df = df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
        # df = df.sort_values(by=['Accuracy'], ascending=[False])
        # drop duplicates, specify Subject
        df = df.drop_duplicates('Subject', keep='first')
        unique_subjs = df['Subject'].unique()
        # path_to_csv = f'./results/best_params/params_{c_POW}_out.csv'
        # df.to_csv(path_to_csv, index=False)
    # min_C = df['C'].min()
    # max_C = df['C'].max()
    # min_gam = df['gam'].min()
    # max_gam = df['gam'].max()
    g = sns.relplot(data=df, x="C", y="gam", hue="Subject_Type", kind="scatter")
    title = f"Per-Subject Params [10^-{c_POW}, 10^{c_POW}]"
    g.fig.subplots_adjust(top=.9)
    g.fig.suptitle(title)
    plt.yscale('log')
    plt.xscale('log')
    # plt.show()
    plt.savefig(f'./results/figures/param_maps/{seed}_{title}_{c_POW}.png', dpi=300)
    plt.close()


def plot_params_alternate(pkl_name):
    df = pd.read_pickle(pkl_name)
    df = df.sort_values(by=['Subject'])
    title = f"(All Seeds) Per-Subject Params (gamma) [10^-{c_POW}, 10^{c_POW}]"
    g = sns.boxplot(data=df, x="Subject_Type", y="gam", hue="Subject_Type").set(title=title)
    plt.yscale('log')
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(f'./results/figures/param_maps/{seed}_{title}_{c_POW}_boxG.png', dpi=300)
    plt.close()

    title = f"(All Seeds) Per-Subject Params (C) [10^-{c_POW}, 10^{c_POW}]"
    g = sns.boxplot(data=df, x="Subject_Type", y="C", hue="Subject_Type").set(title=title)
    plt.yscale('log')
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(f'./results/figures/param_maps/{seed}_{title}_{c_POW}_boxC.png', dpi=300)
    plt.close()


# plot C and gamma average on same plot
def plot_param_avgs(pkl_name):
    df = pd.read_pickle(pkl_name)
    min_C = df['C'].min()
    max_C = df['C'].max()
    min_gam = df['gam'].min()
    max_gam = df['gam'].max()

    # print(f"C: [{min_C}, {max_C}]")
    # print(f"gamma: [{min_gam}, {max_gam}]")
    # g = sns.pairplot(data=df, hue="Subject_Type", palette="dark")
    g = sns.displot(data=df, x="C", y="gam", hue="Subject_Type", kind="kde", palette="bright")
    plt.show()


# plot validation accuracy as a function of C for both control and stutter subjects
def plot_valid_accuracy_vs_C(scaleTransform):
    pkl_name = f'./{seed}_individual_variableC_results_combined.pkl'
    if scaleTransform:
        pkl_name = f'./{seed}_individual_variableC_results_scaled_combined.pkl'
    df = pd.read_pickle(pkl_name)
    # select only the validation accuracy
    df = df[df['Accuracy_Type'] == 'validation']
    # plot the accuracy curve
    g = sns.relplot(data=df, kind="line", x="C", y="Accuracy", hue="Subject_Type", palette="bright", height=6,
                    aspect=2.5)
    g.fig.subplots_adjust(top=.95)
    title = "Validation Accuracy as F(C)"
    g.ax.set_title(title)
    plt.xscale('log')
    # plt.show()
    plt.savefig(f'./results/figures/{seed}_{title}_{c_POW}.png', dpi=300)
    plt.close()


# Plot the trendline for increasing values of C paired with the gamma value returned by the param. search
def plot_opt_gam_increasing_C(pkl_name):
    df = pd.read_pickle(pkl_name)
    # select only the test accuracy
    df2 = df[df['Accuracy_Type'] == 'test']

    # Group by 'Subject', 'C', and 'Gamma', calculate mean accuracy
    result_df = df2.groupby(['Subject', 'Subject_Type', 'C', 'gam'], as_index=False)['Accuracy'].mean()

    title = f"Test Accuracy as F(C) with Gamma 0.005"
    save_path = f'./results/figures/{seed}_{title}_{c_POW}.png'
    create_and_save_plot(result_df, x_col="C", y_col="Accuracy", hue_col="Subject_Type", title=title,
                         save_path=save_path, x_label="C", y_label="Accuracy")

    df3 = df[df['Accuracy_Type'] == 'validation']

    # what is the mean value of gam for each group
    cond1 = (df3['Subject_Type'] == "Stutter")
    pws_gam = df3[cond1]['gam'].values
    mean_pws_gam = np.mean(pws_gam)
    # get accuracies of people who do not stutter
    cond2 = (df3['Subject_Type'] == "Control")
    control_gam = df3[cond2]['gam'].values
    mean_control_gam = np.mean(control_gam)

    result_df2 = df3.groupby(['Subject', 'Subject_Type', 'C', 'gam'], as_index=False)['Accuracy'].mean()
    title = f"Validation Accuracy as F(C) with Gamma 0.005"
    save_path = f'./results/figures/{seed}_{title}_{c_POW}.png'
    create_and_save_plot(result_df2, x_col="C", y_col="Accuracy", hue_col="Subject_Type", title=title,
                         save_path=save_path, x_label="C", y_label="Accuracy")


# Define a function to create and save plots with larger fonts
def create_and_save_plot(data, x_col, y_col, hue_col, title, save_path, x_label=None, y_label=None):
    sns.set(style="whitegrid", font_scale=1.5)  # Increase font size
    g = sns.relplot(data=data, kind="line", x=x_col, y=y_col, hue=hue_col, palette="bright", height=6, aspect=2.5)
    g.fig.subplots_adjust(top=0.90)
    g.ax.set_title(title, fontsize=20)  # Increase title font size
    if x_label:
        g.ax.set_xlabel(x_label, fontsize=16)  # Increase x-axis label font size
    if y_label:
        g.ax.set_ylabel(y_label, fontsize=16)  # Increase y-axis label font size
    g.ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label font size
    plt.xscale('log')
    plt.tight_layout(pad=1.2)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_opt_c_increasing_Gamma(pkl_name):
    df = pd.read_pickle(pkl_name)
    # select only the test accuracy
    df2 = df[df['Accuracy_Type'] == 'test']
    # Group by 'Subject', 'C', and 'Gamma', calculate mean accuracy
    result_df = df2.groupby(['Subject', 'Subject_Type', 'C', 'gam'], as_index=False)['Accuracy'].mean()

    title = f"Test Accuracy as F(gamma) with C 1.0"
    save_path = f'./results/figures/{seed}_{title}_{c_POW}.png'
    create_and_save_plot(result_df, x_col="gam", y_col="Accuracy", hue_col="Subject_Type", title=title,
                         save_path=save_path, x_label="Gamma", y_label="Accuracy")

    df3 = df[df['Accuracy_Type'] == 'validation']
    result_df2 = df3.groupby(['Subject', 'Subject_Type', 'C', 'gam'], as_index=False)['Accuracy'].mean()
    title = f"Validation Accuracy as F(gamma) with C 1.0"
    save_path = f'./results/figures/{seed}_{title}_{c_POW}.png'
    create_and_save_plot(result_df2, x_col="gam", y_col="Accuracy", hue_col="Subject_Type", title=title,
                         save_path=save_path, x_label="Gamma", y_label="Accuracy")


# plot test accuracy as a function of C for both control and stutter subjects
def plot_test_accuracy_vs_C(scaleTransform):
    pkl_name = f'./{seed}_individual_variableC_results_combined.pkl'
    if scaleTransform:
        pkl_name = f'./{seed}_individual_variableC_results_scaled_combined.pkl'
    df = pd.read_pickle(pkl_name)
    # select only the test accuracy
    df = df[df['Accuracy_Type'] == 'test']
    # plot the accuracy curve
    g = sns.relplot(data=df, kind="line", x="C", y="Accuracy", hue="Subject_Type", palette="bright", height=6,
                    aspect=2.5)
    g.fig.subplots_adjust(top=.95)
    title = "Test Accuracy as F(C)"
    g.ax.set_title(title)
    plt.xscale('log')
    # plt.show()
    plt.savefig(f'./results/figures/{seed}_{title}_{c_POW}.png', dpi=300)
    plt.close()


# plot validation accuracy as a function of gamma for both control and stutter subjects
def plot_valid_accuracy_vs_gamma(scaleTransform):
    pkl_name = f'./{seed}_individual_variableGamma_results_combined.pkl'
    if scaleTransform:
        pkl_name = f'./{seed}_individual_variableGamma_results_scaled_combined.pkl'
    df = pd.read_pickle(pkl_name)
    # select only the validation accuracy
    df = df[df['Accuracy_Type'] == 'validation']
    # plot the accuracy curve
    g = sns.relplot(data=df, kind="line", x="gam", y="Accuracy", hue="Subject_Type", palette="bright", height=6,
                    aspect=2.5)
    title = "Validation Accuracy as F(gamma)"
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title(title)
    plt.xscale('log')
    # plt.show()
    plt.savefig(f'./results/figures/{seed}_{title}_{c_POW}.png', dpi=300)
    plt.close()


# plot test accuracy as a function of gamma for both control and stutter subjects
def plot_test_accuracy_vs_gamma(scaleTransform):
    pkl_name = f'./{seed}_individual_variableGamma_results_combined.pkl'
    if scaleTransform:
        pkl_name = f'./{seed}_individual_variableGamma_results_scaled_combined.pkl'
    df = pd.read_pickle(pkl_name)
    # select only the validation accuracy
    df = df[df['Accuracy_Type'] == 'test']
    # plot the accuracy curve
    g = sns.relplot(data=df, kind="line", x="gam", y="Accuracy", hue="Subject_Type", palette="bright", height=6,
                    aspect=2.5)
    g.fig.subplots_adjust(top=.95)
    title = "Test Accuracy as F(gamma)"
    g.ax.set_title(title)
    plt.xscale('log')
    # plt.show()
    plt.savefig(f'./results/figures/{seed}_{title}_{c_POW}.png', dpi=300)
    plt.close()


def visualize_decision_boundary(subject, scaleTransform):
    (X_train, y_train), (X_test, y_test) = get_data(subject, scaleTransform)

    # define the dataset for decision function visualization: keep only the first two features in X
    # 7:9
    X_2d = X_train[:, 9:11]
    y_2d = y_train
    C_2d_range = [1e-3, 1, 1e3]
    gamma_2d_range = [1e-3, 1, 1e3]

    # lower_C = c_POW * -1
    # upper_C = c_POW
    # C_2d_range = np.logspace(lower_C, upper_C, 3)
    # gamma_2d_range = np.logspace(lower_C, upper_C, 3)

    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(kernel='rbf', C=C, gamma=gamma)
            clf.fit(X_2d, y_2d)
            classifiers.append((C, gamma, clf))
    plt.figure(figsize=(8, 6))
    fig_span = np.abs(np.max(X_2d) - np.min(X_2d))
    quart_span = fig_span / 8
    xx, yy = np.meshgrid(np.linspace(np.min(X_2d) - quart_span, np.max(X_2d), 200) + quart_span,
                         np.linspace(np.min(X_2d) - quart_span, np.max(X_2d) + quart_span, 200))
    for k, (C, gamma, clf) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # visualize decision function for these parameters
        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)), size="medium")

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors="k")
        plt.xticks(())
        plt.yticks(())
        plt.axis("tight")

    subject = get_new_subject_name(subject)
    title = f"{subject} 2D Hyperparam. Visualization"
    plt.suptitle(title)
    plt.savefig(f'./results/figures/2DVis/{seed}_{title}.png', dpi=300)
    plt.close()


def plot_single_param_increasing(scaleTransform):
    # Plot accuracy as a function of C parameter
    plot_valid_accuracy_vs_C(scaleTransform)
    plot_test_accuracy_vs_C(scaleTransform)
    # Plot accuracy as a function of gamma parameter
    plot_valid_accuracy_vs_gamma(scaleTransform)
    plot_test_accuracy_vs_gamma(scaleTransform)


def plot_indiv_and_group_test_acc(scaleTransform=True):
    # plot average test accuracy for all with default params
    indiv_results_pkl = f'./{seed}_individual_results_combined.pkl'
    if scaleTransform:
        indiv_results_pkl = f'./{seed}_individual_results_scaled_combined.pkl'
    bar_chart(indiv_results_pkl, title="Individual Test Accuracy Default Params", logit=False)
    group_avg_bar_chart(indiv_results_pkl, title="Group Mean Test Accuracy Default Params", logit=False)
    test_mean_significance(indiv_results_pkl)


def plot_current_cPOW(scaleTransform):
    pkl_name = f'./{seed}_individual_nestedCV_results_{c_POW}_combined.pkl'
    if scaleTransform:
        pkl_name = f'./{seed}_individual_nestedCV_results_{c_POW}_scaled_combined.pkl'
    bar_chart(pkl_name,
              title=f"Validation Accuracy Comparison [10^-{c_POW}, 10^{c_POW}]", logit=False)
    group_avg_bar_chart(pkl_name, title=f"Group Mean Validation Accuracy Comparison [10^-{c_POW}, 10^{c_POW}]",
                        logit=False)
    # group_avg_bar_chart(pkl_name, logit=True)
    plot_params_2D(pkl_name)
    # print("Average Parameters")
    # get_avg_params(pkl_name)
    # print(f"Significance testing for c_POW: {c_POW}")
    # test_mean_significance(pkl_name)
    # test_mean_significance(pkl_name, 'C')
    # test_mean_significance(pkl_name, 'gam')
    # plot_param_avgs(pkl_name)


def plot_cPOWs_only(scaleTransform):
    c_pow_range = [1, 3, 5, 7]
    for c in c_pow_range:
        global c_POW
        c_POW = c
        plot_current_cPOW(scaleTransform)


# Test the best performing hyperparameters on the held-out set
def hyp_validation(scaleTransform):
    # plot_indiv_and_group_test_acc(scaleTransform)
    # plot_single_param_increasing(scaleTransform)
    c_pow_range = [1, 3, 5, 7]
    for c in c_pow_range:
        global c_POW
        c_POW = c
        evaluate_gamma(scaleTransform)
        # test_evaluation(scaleTransform)
        # pkl_name = f'./{seed}_hyperparam_testing_{c_POW}.pkl'
        # if scaleTransform:
        #     pkl_name = f'./{seed}_hyperparam_testing_{c_POW}_scaled.pkl'
        # bar_chart(pkl_name, title=f"Test Evaluation with Opt. Params [10^-{c_POW}, 10^{c_POW}]")
        # group_avg_bar_chart(pkl_name, title=f"Grouped Test Evaluation with Opt. Params [10^-{c_POW}, 10^{c_POW}]")
        # test_mean_significance(pkl_name)
        # test_mean_significance(pkl_name, 'C')
        # test_mean_significance(pkl_name, 'gam')
        # plot_current_cPOW(scaleTransform)


# generate combined results for each seed, c_POW combo.
def comb_test_results():
    c_pow_range = [1, 3, 5, 7]
    seeds = [0, 42, 3407, 5186, 77987]
    # For each hyperparameter search bounds...
    global c_POW
    global seed
    for c in c_pow_range:
        c_POW = c
        # collect the results from the different seeds into two dataframes
        test_df = pd.DataFrame()
        valid_df = pd.DataFrame()
        for s in seeds:
            seed = s
            validation_pkl_name = f'./{seed}_individual_nestedCV_results_{c_POW}_scaled_combined.pkl'
            seed_df = pd.read_pickle(validation_pkl_name)
            # sort based on accuracy, gam, C
            seed_df = seed_df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
            # drop duplicates, specify Subject
            seed_df = seed_df.drop_duplicates('Subject', keep='first')
            # add a column for the current seed
            seed_df["seed"] = seed
            valid_df = pd.concat([valid_df, seed_df])

            test_pkl_name = f'./{seed}_hyperparam_testing_{c_POW}_scaled.pkl'
            test_seed_df = pd.read_pickle(test_pkl_name)
            test_seed_df["seed"] = seed
            test_df = pd.concat([test_df, test_seed_df])
        # Now, we have collected all validation and test data for the given hyp param search space
        valid_out_name = f'./{c_POW}_nestedCV_all_seeds_trimmed.pkl'
        test_out_name = f'./{c_POW}_test_results_all_seeds_trimmed.pkl'
        valid_df.to_pickle(valid_out_name)
        test_df.to_pickle(test_out_name)


# generate combined results for (each seed) all the default parameter trials
def comb_default_results():
    seeds = [0, 42, 3407, 5186, 77987]
    global seed
    all_df = pd.DataFrame()
    for s in seeds:
        seed = s
        test_pkl_name = f'./{seed}_individual_results_scaled_combined.pkl'
        test_df = pd.read_pickle(test_pkl_name)
        test_df["seed"] = seed
        all_df = pd.concat([all_df, test_df])
    out_name = f'./default_params_all_seeds.pkl'
    all_df.to_pickle(out_name)


def calculate_proportion(df, threshold):
    # Count the number of entries that exceed the threshold
    count_exceed_threshold = np.sum(df.values > threshold)

    # Calculate the proportion
    total_entries = df.size
    proportion = count_exceed_threshold / total_entries

    return proportion


def decision_boundary_tests():
    # pkl_name = f'./individual_results_combined.pkl'
    # bar_chart(pkl_name, title='SVC Individual Accuracy')
    # group_avg_bar_chart(pkl_name, title='SVC Group Mean Accuracy')
    # run_classification_all_groups_default_params(scaleTransform=True)
    # pkl_name = f'./individual_results_scaled_combined.pkl'
    # bar_chart(pkl_name, title='SVC Individual Accuracy - Alternate Scaling')
    # group_avg_bar_chart(pkl_name, title='SVC Group Mean Accuracy - Alternate Scaling')
    subject = 'CM112'
    visualize_decision_boundary(subject, scaleTransform=True)
    # scaleTransform = True
    # visualize_decision_boundary(subject, scaleTransform)


def log_regression_tests():
    scaleTransform = True
    run_indiv_log_regression(patho=False)
    run_indiv_log_regression(patho=True)
    combine_df_log_regression()
    pkl_name = './individual_log_regression_results_combined.pkl'
    bar_chart(pkl_name, title="Logistic Regression Accuracy")
    group_avg_bar_chart(pkl_name, title="Logistic Regression Group Mean Accuracy")


def plot_all_seeds():
    # comb_test_results()
    c_pow_range = [1, 3, 5, 7]
    # For each hyperparameter search bounds...
    global c_POW
    for c in c_pow_range:
        c_POW = c
        valid_pkl_name = f'./{c_POW}_nestedCV_all_seeds_trimmed.pkl'
        test_pkl_name = f'./{c_POW}_test_results_all_seeds_trimmed.pkl'

        plot_params_2D(test_pkl_name)

        # df = pd.read_pickle(test_pkl_name)

        indiv_valid_acc_title = f"(All Seeds) Validation Acc. with Opt. Params [10^-{c_POW}, 10^{c_POW}]"
        group_valid_acc_title = f"(All Seeds) Grouped Validation Acc. with Opt. Params [10^-{c_POW}, 10^{c_POW}]"
        box_plot(valid_pkl_name, title=indiv_valid_acc_title, remove_dupes=False)
        group_avg_box_plot(valid_pkl_name, title=group_valid_acc_title, remove_dupes=False)

        indiv_test_acc_title = f"(All Seeds) Test Evaluation with Opt. Params [10^-{c_POW}, 10^{c_POW}]"
        group_test_acc_title = f"(All Seeds) Grouped Test Evaluation with Opt. Params [10^-{c_POW}, 10^{c_POW}]"
        box_plot(test_pkl_name, title=indiv_test_acc_title, remove_dupes=False)
        group_avg_box_plot(test_pkl_name, title=group_test_acc_title, remove_dupes=False)


def label_fmt(s):
    try:
        n = "{:.3f}".format(float(s))
    except:
        n = ""
    return n


# takes an (unfiltered) dataframe of validation accuracies for an entire hyperparameter grid.
# plots a heatmap of the results
def plot_param_heatmap(valid_df, title):
    # valid_df["C"] = valid_df["C"].apply(lambda x: decimal.Decimal(x))
    # valid_df["gam"] = valid_df["gam"].apply(lambda x: decimal.Decimal(x))
    #
    # get the unique values of C and gamma:
    uniq_Cs = np.unique(valid_df.C.values)
    uniq_Gs = np.unique(valid_df.gam.values)
    #
    # x_tick_loc = uniq_Cs[0::2]
    # y_tick_loc = uniq_Gs[0::2]

    pvt = valid_df.pivot("gam", "C", "Accuracy")
    #
    # prop_exceed = calculate_proportion(pvt, 0.70)
    #
    # print("Percentage of validation accuracies above 70%: " + str(prop_exceed))

    g = sns.heatmap(pvt, xticklabels=uniq_Cs, yticklabels=uniq_Gs)
    g.set_xticklabels([label_fmt(label.get_text()) for label in g.get_xticklabels()])
    g.set_yticklabels([label_fmt(label.get_text()) for label in g.get_yticklabels()])
    for label in g.get_xticklabels()[::2]:
        label.set_visible(False)
    for label in g.get_yticklabels()[::2]:
        label.set_visible(False)

    # g.set_xscale('symlog')
    # g.set_yscale('symlog')
    # g.set(xlabel="C", ylabel="Gamma")
    # g.subplots_adjust(top=.95)
    # g.set_xticks(x_tick_loc)
    # g.set_yticks(y_tick_loc)

    # g.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
    # g.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
    g.set_xlabel("C", fontsize=16)
    g.set_ylabel("Gamma", fontsize=16)
    g.set_title(title, fontsize=20)
    plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(left=0.20)
    plt.savefig(f'./results/figures/heatmaps/{seed}_{title}_{c_POW}.png', dpi=300)
    plt.close()
    # plt.show()


def plot_gamma_slices(valid_df, title):
    # get the unique values of C and gamma:
    uniq_Cs = np.unique(valid_df.C.values)
    uniq_Gs = np.unique(valid_df.gam.values)
    gamma_vals = uniq_Gs
    subject = title.split(" ")[0]
    g_ind = 0
    for g in gamma_vals:
        y_value = g
        gamma_only = valid_df[valid_df["gam"] == g]
        plot_title = f"{title} - {y_value}"
        g = sns.relplot(data=gamma_only, kind="line", x="C", y="Accuracy",
                        height=6,
                        aspect=2.5)
        g.fig.subplots_adjust(top=.95)
        plt.xscale('log')

        g.ax.set_xlabel("C")
        g.ax.set_ylabel("Validation Accuracy")
        g.ax.set_title(plot_title)
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.20)
        plt.savefig(f'./results/figures/heatmaps/slices/horizontal/validation/{seed}_{subject}_G{g_ind}_{c_POW}.png',
                    dpi=300)
        plt.close()

        plot_title = f"{subject} Training Accuracy as f(C) - {y_value}"
        g = sns.relplot(data=gamma_only, kind="line", x="C", y="Train Accuracy",
                        height=6,
                        aspect=2.5)
        g.fig.subplots_adjust(top=.95)
        plt.xscale('log')
        g.fig.subplots_adjust(top=.95)
        g.ax.set_xlabel("C")
        g.ax.set_ylabel("Training Accuracy")
        g.ax.set_title(plot_title)
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.20)
        plt.savefig(f'./results/figures/heatmaps/slices/horizontal/training/{seed}_{subject}_G{g_ind}_{c_POW}.png',
                    dpi=300)
        plt.close()

        g_ind += 1


def plot_C_slices(valid_df, title):
    # get the unique values of C and gamma:
    uniq_Cs = np.unique(valid_df.C.values)
    subject = title.split(" ")[0]
    g_ind = 0
    for C in uniq_Cs:
        y_value = C
        C_only = valid_df[valid_df["C"] == C]
        plot_title = f"{title} - {y_value}"
        g = sns.relplot(data=C_only, kind="line", x="gam", y="Accuracy",
                        height=6,
                        aspect=2.5)
        g.fig.subplots_adjust(top=.95)
        plt.xscale('log')

        g.ax.set_xlabel("Gamma")
        g.ax.set_ylabel("Validation Accuracy")
        g.ax.set_title(plot_title)
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.20)
        plt.savefig(f'./results/figures/heatmaps/slices/vertical/validation/{seed}_{subject}_G{g_ind}_{c_POW}.png',
                    dpi=300)
        plt.close()

        plot_title = f"{subject} Training Accuracy as f(gamma) - {y_value}"
        g = sns.relplot(data=C_only, kind="line", x="gam", y="Train Accuracy",
                        height=6,
                        aspect=2.5)
        g.fig.subplots_adjust(top=.95)
        plt.xscale('log')
        g.fig.subplots_adjust(top=.95)
        g.ax.set_xlabel("Gamma")
        g.ax.set_ylabel("Training Accuracy")
        g.ax.set_title(plot_title)
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.20)
        plt.savefig(f'./results/figures/heatmaps/slices/vertical/training/{seed}_{subject}_G{g_ind}_{c_POW}.png',
                    dpi=300)
        plt.close()

        g_ind += 1


def sanity_test(subject):
    global seed
    seed = 42
    global c_POW
    c_POW = 3
    scaleTransform = True

    pub_subj = get_new_subject_name(subject)

    if pub_subj is None:
        # This subject is excluded from the study
        return

    print("Sanity Check")
    print("Seed:")
    print(seed)
    print(f"Search range: 10^{c_POW}")
    valid_df = pd.DataFrame()
    valid_df = nested_CV_Intra_Subj3(subject, valid_df, scaleTransform)

    # plot_C_slices(valid_df, f'{pub_subj} Validation Accuracy as f(Gamma)')
    # plot_gamma_slices(valid_df, f'{pub_subj} Validation Accuracy as f(C)')
    #
    heatmap_title = f'{pub_subj} Hyperparameter Search'
    plot_param_heatmap(valid_df, heatmap_title)
    #
    # # sort based on accuracy, gam, C
    # valid_df = valid_df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
    # # drop duplicates, specify Subject
    # valid_df = valid_df.drop_duplicates('Subject', keep='first')
    # valid_df = valid_df.sort_values(by=['Subject'])
    #
    # subject_C = valid_df.C[0]
    # subj_gam = valid_df.gam[0]
    #
    # print(subject)
    # print("======== Tuned Params ==========")
    # print(f"C: {subject_C}")
    # print(f"Gamma: {subj_gam}")
    #
    # test_df = pd.DataFrame()
    # test_df = chosen_param_classification(subject, test_df, scaleTransform, C=subject_C, gamma=subj_gam)
    #
    # print(f"Validation Accuracy: {valid_df.Accuracy[0]}")
    # print(f"Test Accuracy (80% train, 20% test): {test_df.Accuracy[0]}")
    #
    # small_test_df = pd.DataFrame()
    # small_test_df = small_data_classification(subject, small_test_df, scaleTransform, C=subject_C, gamma=subj_gam)
    # print(f"Test Accuracy (60% train, 20% test): {small_test_df.Accuracy[0]}")
    #
    # print("======== Standard Params ==========")
    # default_C = 1.0
    # print(f"C: {default_C}")
    #
    # default_gamma = 'scale'
    #
    # test_df = pd.DataFrame()
    # test_df = chosen_param_classification(subject, test_df, scaleTransform, C=default_C, gamma=default_gamma)
    # print(f"Test Accuracy (80% train, 20% test): {test_df.Accuracy[0]}")
    #
    # small_test_df = pd.DataFrame()
    # small_test_df = small_data_classification(subject, small_test_df, scaleTransform, C=default_C, gamma=default_gamma)
    # print(f"Test Accuracy (60% train, 20% test): {small_test_df.Accuracy[0]}")


def plot_default_params_all_seeds():
    pkl_default_prms = f'./default_params_all_seeds.pkl'
    indiv_title = "(All Seeds) Test Accuracy with Default K-SVC Params."
    group_title = "(All Seeds) Group Mean Test Accuracy with Default K-SVC Params."
    bar_chart(pkl_default_prms, title=indiv_title, remove_dupes=False)
    group_avg_bar_chart(pkl_default_prms, title=group_title, remove_dupes=False)


def leakage_testing():
    (X_train, y_train), (X_test, y_test) = get_data("CF60", True)
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    score = f1_score(y_true=y_test, y_pred=y_pred)
    print('F1 Score: ' + str(np.round(score, 2)))
    plot_feature_importance(X_train, rf)


def plot_feature_importance(X_train, model):
    features = range(0, len(X_train[0]))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.xlabel('Relative Importance')
    plt.show()


def split_tester():
    subj = "CM119"
    # for each subject, there are three LARGE feature sets f0-f2
    X1_train, X1_true_test, \
    train1_labels, test1_labels = train_test_split_helper(0, subj)

    X2_train, X2_true_test, \
    train2_labels, test2_labels = train_test_split_helper(1, subj)

    X3_train, X3_true_test, \
    train3_labels, test3_labels = train_test_split_helper(2, subj)

    for x_samp in X1_true_test:
        for idx, x_samp2 in enumerate(X1_train):
            assert (not (np.array_equal(x_samp, x_samp2)))

    for x_samp in X2_true_test:
        for idx, x_samp2 in enumerate(X2_train):
            assert (not (np.array_equal(x_samp, x_samp2)))

    for x_samp in X3_true_test:
        for idx, x_samp2 in enumerate(X3_train):
            assert (not (np.array_equal(x_samp, x_samp2)))


def optimal_gamma_increasing_C_all_seeds():
    global seed
    seed = 42
    global c_POW
    c_POW = 3
    c_POW_Pkls = {}
    seeds = [0, 42, 3407, 5186, 77987]
    for s in seeds:
        seed = s
        c_pow_range = [1, 3, 5, 7]
        for c in c_pow_range:
            c_POW = c
            evaluate_gamma(scaleTransform)
            seed_pow_pkl = f'./{seed}_optimal_gam_Increasing_C_scaled_{c_POW}.pkl'
            if c in c_POW_Pkls:
                c_POW_Pkls[c].append(seed_pow_pkl)
            else:
                c_POW_Pkls[c] = [seed_pow_pkl]
    for c in c_POW_Pkls:
        c_POW = c
        # load all the pkl files into dataframes:
        pkls = c_POW_Pkls[c]
        df = pd.DataFrame()
        for pkl in pkls:
            df2 = pd.read_pickle(pkl)
            df2["seed"] = pkl.split('_')[0].lstrip('./')
            df = pd.concat([df, df2], sort=False)
        print(df)
        combined_pkl_name = f'./optimal_gam_Increasing_C_all_seeds_{c_POW}.pkl'
        df.to_pickle(combined_pkl_name)
        plot_opt_gam_increasing_C(combined_pkl_name)


def optimal_C_increasing_gamma_all_seeds():
    global seed
    seed = 42
    global c_POW
    c_POW = 3
    c_POW_Pkls = {}
    seeds = [0, 42, 3407, 5186, 77987]
    for s in seeds:
        seed = s
        c_pow_range = [1, 3, 5, 7]
        for c in c_pow_range:
            c_POW = c
            evaluate_C(scaleTransform)
            seed_pow_pkl = f'./{seed}_optimal_c_Increasing_Gamma_scaled_{c_POW}.pkl'
            if c in c_POW_Pkls:
                c_POW_Pkls[c].append(seed_pow_pkl)
            else:
                c_POW_Pkls[c] = [seed_pow_pkl]
    for c in c_POW_Pkls:
        c_POW = c
        # load all the pkl files into dataframes:
        pkls = c_POW_Pkls[c]
        df = pd.DataFrame()
        for pkl in pkls:
            df2 = pd.read_pickle(pkl)
            df2["seed"] = pkl.split('_')[0].lstrip('./')
            df = pd.concat([df, df2], sort=False)
        # print(df)
        combined_pkl_name = f'./optimal_c_Increasing_Gamma_all_seeds_{c_POW}.pkl'
        df.to_pickle(combined_pkl_name)
        remove_subjects()
        trimmed_combined_name = f'./optimal_c_Increasing_Gamma_all_seeds_{c_POW}_trimmed.pkl'
        plot_opt_c_increasing_Gamma(trimmed_combined_name)


def combine_fixed_gamma_trials():
    global seed
    seed = 42
    global c_POW
    c_POW = 3
    scaleTransform = True
    seeds = [0, 42, 3407, 5186, 77987]
    dfs = []
    for s in seeds:
        seed = s
        filename = f'./{seed}_individual_nestedCV_Fixed_Gamma_3_scaled.pkl'
        patho_fname = f'./{seed}_individual_nestedCV_Fixed_Gamma_patho_3_scaled.pkl'
        if os.path.exists(filename):
            df = pd.read_pickle(filename)
            df['seed'] = seed
            dfs.append(df)
        else:
            print(f"File '{filename}' not found.")

        if os.path.exists(patho_fname):
            df2 = pd.read_pickle(patho_fname)
            df2['seed'] = seed
            dfs.append(df2)
        else:
            print(f"File '{filename}' not found.")

    # Concatenate the list of dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)

    out_fname = f'./combined_nestedCV_Fixed_Gamma_C5.pkl'
    combined_df.to_pickle(out_fname)


# Obfuscate subject names
def get_new_subject_name(subject_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv('Subject_CodeMap.csv')

    if subject_name.startswith('C'):
        row = df[df['ANS'] == subject_name]
        if not row.empty:
            return row.iloc[0]['ANS_New']
    elif subject_name.startswith('S'):
        row = df[df['AWS'] == subject_name]
        if not row.empty:
            return row.iloc[0]['AWS_New']

    return None  # Return None if subject_name is not found


# Delete specified subjects out of the results data.
def remove_subjects():
    # List of subjects to be removed
    subjects_to_remove = ["CF60", "CM118", "SF14", "SM51"]
    # Get a list of all .pkl files in the current directory
    pkl_files = [file for file in os.listdir() if file.endswith('.pkl')]

    # Iterate through each .pkl file
    for pkl_file in pkl_files:
        try:
            if not pkl_file.endswith('_trimmed.pkl'):
                # Load data into a pandas DataFrame
                df = pd.read_pickle(pkl_file)

                # Remove rows with specified subjects
                df = df[~df['Subject'].isin(subjects_to_remove)]

                # Save the trimmed DataFrame to a new .pkl file
                trimmed_file_name = pkl_file.replace('.pkl', '_trimmed.pkl')
                df.to_pickle(trimmed_file_name)

                print(f"Processed {pkl_file} and saved as {trimmed_file_name}")

        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
