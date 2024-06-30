import pandas as pd
import ast
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import os
import constants
import seaborn as sns


TASK_TYPE_NAMES = {
    'MULTIPLE_CHOICE': 'Multiple Choice',
    'NATURAL_LANGUAGE_INFERENCE': 'Natural Language\nInference',
    'QUESTION_ANSWERING': 'Question Answering',
    'QUESTION_GENERATION': 'Question Generation',
    'SUMMARIZATION': 'Summarization',
    'TEXT_CLASSIFICATION': 'Text Classification',
    'TEXT_GENERATION': 'Text Generation',
    'OTHER': 'Other',
}

JURISDICTION_NAMES = {
    'BRAZIL': 'Brazil',
    'CHINA': 'China',
    'EU': 'EU',
    'GERMANY': 'Germany',
    'GREECE': 'Greece',
    'INDIA': 'India',
    'INTERNATIONAL': 'International',
    'JAPAN': 'Japan',
    'SPAIN': 'Spain',
    'SOUTH_KOREA': 'South Korea',
    'SWITZERLAND': 'Switzerland',
    'TURKEY': 'Turkey',
    'UK': 'UK',
    'UNKNOWN': 'Unknown',
    'US': 'US',
    'OTHER': 'Other',
}


# Function to filter based on subset
def filter_datasets(df, lang="en", license="commercial"):
    if lang == "en" and license == "commercial":
        intersection = constants.ENGLISH.intersection(constants.COMMERCIAL)
        to_filter = constants.ALL - intersection
    elif lang == "en":
        to_filter = constants.ALL - constants.ENGLISH
    elif license == "commercial":
        to_filter = constants.ALL - constants.COMMERCIAL
    else: # lang == "multi" and license="research"
        to_filter = set()

    for config in to_filter:
        config_parts = config.split("-")
        dataset, subset = config_parts[0], config_parts[1]
        df = df[~((df['dataset'] == dataset) & (df['subset'] == subset))]

    for config in constants.REMOVED:
        config_parts = config.split("-")
        dataset, subset = config_parts[0], config_parts[1]
        df = df[~((df['dataset'] == dataset) & (df['subset'] == subset))]
    
    return df


def safe_parse_dictionary(dict_str):
    """Function to safely parse dictionary strings into actual dictionaries"""
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


def plot_and_save_histogram(data, title, file_prefix, filename, cap=None):
    """Function to plot a histogram and save it to a file"""
    if cap is not None:
        data = np.minimum(data, cap)
        # title = f"{title}, Capped at {cap}"
        filename = f"plots/{file_prefix}/{filename}_capped_at_{cap}.pdf"
    else:
        filename = f"plots/{file_prefix}/{filename}.pdf"
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('axes', titlesize=30)
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)
    cmap = sns.color_palette("Set2", as_cmap=True)

    fig, ax = plt.subplots()
    ax.hist(data, color=cmap.colors[0], bins=100)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_pie_chart(data, column, ax, title, threshold=0.03): 
    # Compute percentages
    percentages = data / data.sum()

    # Create a mask for values greater than the threshold
    mask = percentages > threshold

    # Create a new series with the values above the threshold
    large_values = data[mask].copy()

    # Add an "OTHER" entry for the values below the threshold
    if mask.sum() < len(data):
        large_values.loc['OTHER'] = data[~mask].sum()

    labels = large_values.index
    if column == "task_type":
        labels = [TASK_TYPE_NAMES[label] for label in labels]
    else:
        labels = [JURISDICTION_NAMES[label] for label in labels]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('axes', titlesize=40) 
    plt.rc('axes', labelsize=20)
    cmap = sns.color_palette("Set2", as_cmap=True)
    colors = cmap.colors + ((238/255, 191/255, 212/255), (177/255, 201/255, 240/255))

    # Plot a pie chart
    pie = ax.pie(large_values, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 20})
    ax.set_title(title, x=0.85)
    box = ax.get_position()
    ax.legend(labels=labels, fontsize=30, loc='center left', bbox_to_anchor=(1, 0.5))


# Define a function to plot pie charts for a given column
def plot_pie_charts_for_column(df, name, column, by, ax):
    # Sort column alphabetically
    df = df.sort_values(by=column, axis=0)

    # Aggregate the column counts across the entire dataset and plot a pie chart
    # Don't sort by frequencies, retain sorted order (alphabetical) of column
    if by == "by_examples":
        column_counts = df[column].value_counts(sort=False)
        plot_pie_chart(column_counts, column, ax, f'{name} by Number of Examples')
    elif by == "by_datasets":
        # Count the number of unique datasets for each column value and plot a pie chart
        column_dataset_counts = df.groupby(column)['dataset'].nunique()
        plot_pie_chart(column_dataset_counts, column, ax, f'{name} by Number of Datasets')


def make_plots(df, file_prefix):
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

    # Plot and save the histograms
    plot_and_save_histogram(instruction_length_stats['mean'].dropna(), 'Instruction Lengths',
                            file_prefix, 'instruction_length_histogram')
    plot_and_save_histogram(prompt_length_stats['mean'].dropna(), 'Prompt Lengths',
                            file_prefix, 'prompt_length_histogram', cap=15000)
    plot_and_save_histogram(prompt_length_stats['mean'].dropna(), 'Prompt Lengths',
                            file_prefix, 'prompt_length_histogram', cap=5000)
    plot_and_save_histogram(answer_length_stats['mean'], 'Answer Lengths', 
                            file_prefix, 'answer_length_histogram', cap=1000)

    # Flatten the columns
    df_flattened = df.explode(one_entry_cols)

    # Plot and save the pie charts
    by = "by_examples"
    filename = f"plots/{file_prefix}/{by}_pie_chart.pdf"
    # Plot both charts on one figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(27, 9))
    plot_pie_charts_for_column(df_flattened, "Task Type", "task_type", by, ax0)
    plot_pie_charts_for_column(df_flattened, "Jurisdiction", "jurisdiction", by, ax1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    by = "by_datasets"
    filename = f"plots/{file_prefix}/{by}_pie_chart.pdf"
    # Plot both charts on one figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(27, 9))
    plot_pie_charts_for_column(df_flattened, file_prefix, "Task Type", "task_type", by, ax0)
    plot_pie_charts_for_column(df_flattened, file_prefix, "Jurisdiction", "jurisdiction", by, ax1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('aggregate_stats.csv')

    # "en_commercial"
    file_prefix = "en_commercial"
    print(file_prefix)
    os.makedirs(f"plots/{file_prefix}", exist_ok=True)
    df_en_commercial = filter_datasets(df, lang=file_prefix.split("_")[0], license=file_prefix.split("_")[1])
    make_plots(df_en_commercial, file_prefix)

    # "en_research"
    file_prefix = "en_research"
    print(file_prefix)
    os.makedirs(f"plots/{file_prefix}", exist_ok=True)
    df_en_research = filter_datasets(df, lang=file_prefix.split("_")[0], license=file_prefix.split("_")[1])
    make_plots(df_en_research, file_prefix)

    # "multi_commercial"
    file_prefix = "multi_commercial"
    print(file_prefix)
    os.makedirs(f"plots/{file_prefix}", exist_ok=True)
    df_multi_commercial = filter_datasets(df, lang=file_prefix.split("_")[0], license=file_prefix.split("_")[1])
    make_plots(df_multi_commercial, file_prefix)

    # "multi_research"
    file_prefix = "multi_research"
    print(file_prefix)
    os.makedirs(f"plots/{file_prefix}", exist_ok=True)
    df_multi_research = filter_datasets(df, lang=file_prefix.split("_")[0], license=file_prefix.split("_")[1])
    make_plots(df_multi_research, file_prefix)


if __name__ == "__main__":
    main()