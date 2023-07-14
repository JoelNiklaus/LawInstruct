import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('aggregate_stats.csv')


# Define a function to safely parse dictionary strings into actual dictionaries
def safe_parse_dictionary(dict_str):
    if isinstance(dict_str, str):
        # Replace 'nan' with 'None' before parsing
        dict_str = dict_str.replace('nan', 'None')
        try:
            # Try to parse as Python literal structure
            return ast.literal_eval(dict_str)
        except ValueError:
            # If parsing fails, return an empty dictionary
            return {}
    else:
        # If the input is not a string (probably NaN), return an empty dictionary
        return {}


# Apply the function to the relevant columns
one_entry_cols = ['jurisdiction', 'task_type']
pd_describe_cols = ['instruction_length', 'prompt_length', 'answer_length']
columns_to_parse = one_entry_cols + pd_describe_cols
for col in columns_to_parse:
    df[col] = df[col].apply(safe_parse_dictionary)

# Convert dictionary columns into pandas Series
instruction_length_stats = df['instruction_length'].apply(pd.Series)
prompt_length_stats = df['prompt_length'].apply(pd.Series)
answer_length_stats = df['answer_length'].apply(pd.Series)


# Define a function to plot a histogram and save it to a file
def plot_and_save_histogram(data, title, filename, cap=None):
    if cap is not None:
        data = np.minimum(data, cap)
        title = f"{title}, Capped at {cap}"
        filename = f"plots/{filename}_capped_at_{cap}.png"
    else:
        filename = f"plots/{filename}.png"
    fig, ax = plt.subplots()
    ax.hist(data, bins=100)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename)


# Plot and save the histograms
plot_and_save_histogram(instruction_length_stats['mean'].dropna(), 'Distribution of Instruction Lengths (Mean)',
                        'instruction_length_histogram')
plot_and_save_histogram(prompt_length_stats['mean'].dropna(), 'Distribution of Prompt Lengths (Mean)',
                        'prompt_length_histogram', cap=15000)
plot_and_save_histogram(prompt_length_stats['mean'].dropna(), 'Distribution of Prompt Lengths (Mean)',
                        'prompt_length_histogram', cap=5000)
plot_and_save_histogram(answer_length_stats['mean'], 'Distribution of Answer Lengths (Mean)', 'answer_length_histogram',
                        cap=1000)

# Flatten the columns
df_flattened = df.explode(one_entry_cols)


def plot_pie_chart(data, title, filename, threshold=0.03):
    # Compute percentages
    percentages = data / data.sum()

    # Create a mask for values greater than the threshold
    mask = percentages > threshold

    # Create a new series with the values above the threshold
    large_values = data[mask].copy()

    # Add an "Other" entry for the values below the threshold
    if mask.sum() < len(data):
        large_values.loc['Other'] = data[~mask].sum()

    # Plot a pie chart
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(large_values, labels=large_values.index, autopct='%1.1f%%', startangle=140)
    ax.set_title(title)
    plt.tight_layout()

    # Save the pie chart to a file
    plt.savefig(filename)


# Define a function to plot pie charts for a given column
def plot_pie_charts_for_column(df, column, name):
    # Aggregate the column counts across the entire dataset and plot a pie chart
    column_counts = df[column].value_counts()
    plot_pie_chart(column_counts, f'{name} by Number of Examples', f'plots/{column}_by_examples_pie_chart.png')

    # Count the number of unique datasets for each column value and plot a pie chart
    column_dataset_counts = df.groupby(column)['dataset'].nunique()
    plot_pie_chart(column_dataset_counts, f'{name} by Number of Datasets', f'plots/{column}_by_datasets_pie_chart.png')


# Plot and save the pie charts
plot_pie_charts_for_column(df_flattened, "task_type", "Task Type")
plot_pie_charts_for_column(df_flattened, "jurisdiction", "Jurisdiction")
