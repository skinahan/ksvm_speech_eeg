import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def load_data(file_path):
    """Load CSV data into a pandas DataFrame with manual headers."""
    return pd.read_csv(file_path, header=None,
                       names=['File Path', 'Number of Components', 'Mean Component Probability'])


def extract_subject_code(file_path):
    """Extract subject code from the file path."""
    match = re.search(r'\\([A-Z]+[0-9]+)\\', file_path)
    if match:
        return match.group(1)
    return None


def map_private_to_public(file_path, group, df_CodeMap):
    """Map private subject code to public subject code using the CodeMap."""
    subject_code = extract_subject_code(file_path)
    if subject_code:
        # print(f"Subject Code: {subject_code}")
        match_row = None
        if group == 'AWS' and subject_code.startswith('S'):
            # print("Mapping to AWS")
            match_row = df_CodeMap[df_CodeMap['AWS'] == subject_code]
        elif group == 'ANS' and subject_code.startswith('C'):
            # print("Mapping to ANS")
            match_row = df_CodeMap[df_CodeMap['ANS'] == subject_code]

        if match_row is not None and not match_row.empty:
            return match_row.iloc[0]['AWS_New'] if group == 'AWS' else match_row.iloc[0]['ANS_New']

    return None


def preprocess_data(df, group, df_CodeMap):
    """Preprocess the data by mapping private to public subject codes."""
    df['Public_Code'] = df.apply(lambda row: map_private_to_public(row['File Path'], group, df_CodeMap), axis=1)
    df = df.dropna(subset=['Public_Code'])  # Drop rows where Public_Code is None
    return df


def create_seaborn_plot(df):
    """Create a seaborn bar plot comparing the number of components across groups."""
    sns.set(style="whitegrid")

    # Assuming your columns are named 'Public_Code', 'Number of Components', and 'Group'
    sns.barplot(x='Public_Code', y='Number of Components', hue='Group', data=df)

    plt.title('Number of Non-Artifactual Components')
    plt.xlabel('Subject')
    plt.ylabel('Number of Components')
    plt.legend(title='Group')

    # Rotate X axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Save the plot with informative name
    plt.savefig('number_of_components_comparison.png', dpi=300)

    # Show the plot
    plt.show()


def create_mean_probability_plot(df):
    """Create a seaborn bar plot comparing the mean probability across groups."""
    sns.set(style="whitegrid")

    # Remove percentage sign and convert to decimal
    df['Mean Component Probability'] = df['Mean Component Probability'].str.rstrip('%').astype('float') / 100.0

    # Assuming your columns are named 'Public_Code', 'Mean Component Probability', and 'Group'
    sns.barplot(x='Public_Code', y='Mean Component Probability', hue='Group', data=df)

    plt.title('Mean Component Probability Comparison')
    plt.xlabel('Subject')
    plt.ylabel('Mean Component Probability')
    plt.legend(title='Group')

    # Rotate X axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Save the plot with an informative name
    plt.savefig('mean_component_probability_comparison.png', dpi=300)

    # Show the plot
    plt.show()


def conduct_t_tests(df):
    """Conduct independent t-tests and save results to a CSV file."""
    ans_data = df[df['Group'] == 'ANS']['Number of Components']
    aws_data = df[df['Group'] == 'AWS']['Number of Components']

    # Independent t-test for Number of Components
    t_stat_comp, p_val_comp = ttest_ind(ans_data, aws_data, equal_var=False)

    ans_prob = df[df['Group'] == 'ANS']['Mean Component Probability']
    aws_prob = df[df['Group'] == 'AWS']['Mean Component Probability']

    # Independent t-test for Mean Component Probability
    t_stat_prob, p_val_prob = ttest_ind(ans_prob, aws_prob, equal_var=False)

    # Create a DataFrame to save the results
    results_df = pd.DataFrame({
        'Metric': ['Number of Components', 'Mean Component Probability'],
        'T-Statistic': [t_stat_comp, t_stat_prob],
        'P-Value': [p_val_comp, p_val_prob]
    })

    # Save the results to a CSV file
    results_df.to_csv('t_test_results.csv', index=False)

    return results_df


def main():
    # Step 1: Load the data
    file_path_ANS = r'D:\Research\EEG-DIVA\SpeakingDAF\SpeakingDAF\ICA_summary_stats_ANS.csv'
    file_path_AWS = r'D:\Research\EEG-DIVA\SpeakingDAF_S\ICA_summary_stats_AWS.csv'
    file_path_CodeMap = r'H:\My Drive\Fall 2022\eeg_proj\eeglab2021.1\Subject_CodeMap.csv'

    df_ANS = load_data(file_path_ANS)
    df_AWS = load_data(file_path_AWS)
    # Assign 'Group' column to df_ANS and df_AWS
    df_ANS['Group'] = 'ANS'
    df_AWS['Group'] = 'AWS'

    # Concatenate dataframes
    df_combined = pd.concat([df_ANS, df_AWS], ignore_index=True)

    # Preprocess data and create 'Public_Code' column
    df_CodeMap = pd.read_csv(file_path_CodeMap)
    df_ANS = preprocess_data(df_ANS, 'ANS', df_CodeMap)
    df_AWS = preprocess_data(df_AWS, 'AWS', df_CodeMap)

    # Combine dataframes
    df_combined = pd.concat([df_ANS, df_AWS], ignore_index=True)
    # Remove percentage sign and convert to decimal
    df_combined['Mean Component Probability'] = df_combined['Mean Component Probability'].str.rstrip('%').astype('float') / 100.0

    # Create seaborn plots
    # create_seaborn_plot(df_combined)
    # create_mean_probability_plot(df_combined)
    # Usage
    t_test_results = conduct_t_tests(df_combined)

    # print(t_test_results)


if __name__ == "__main__":
    main()
