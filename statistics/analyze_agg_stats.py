import pandas as pd
import ast
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import constants

TASK_TYPE_COLOR_INDICES = {
    'MULTIPLE_CHOICE': 0,
    'NATURAL_LANGUAGE_INFERENCE': 1,
    'QUESTION_ANSWERING': 2,
    'QUESTION_GENERATION': 3,
    'SUMMARIZATION': 4,
    'TEXT_CLASSIFICATION': 5,
    'TEXT_GENERATION': 6,
    'OTHER': 7,
}

JURISDICTION_COLOR_INDICES = {
    'BRAZIL': 0,
    'CHINA': 1,
    'EU': 2,
    'GERMANY': 3,
    'GREECE': 4,
    'INDIA': 5,
    'INTERNATIONAL': 6,
    'JAPAN': 7,
    'SPAIN': 8,
    'SOUTH_KOREA': 9,
    'SWITZERLAND': 10,
    'TURKEY': 11,
    'UK': 12,
    'UNKNOWN': 13,
    'US': 14,
    'OTHER': 15,
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
        title = f"{title}, Capped at {cap}"
        filename = f"plots/{file_prefix}/{filename}_capped_at_{cap}.png"
    else:
        filename = f"plots/{file_prefix}/{filename}.png"
    fig, ax = plt.subplots()
    ax.hist(data, bins=100)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_pie_chart(data, column, title, filename, threshold=0.03):
    if column == "task_type":
        cmap = matplotlib.colormaps.get_cmap("tab10")
        color_indices = TASK_TYPE_COLOR_INDICES
    elif column == "jurisdiction":
        cmap = matplotlib.colormaps.get_cmap("tab20")
        color_indices = JURISDICTION_COLOR_INDICES
    
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
    colors = [cmap.colors[color_indices[label]] for label in labels]

    # Plot a pie chart
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(large_values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.set_title(title)
    plt.tight_layout()

    # Save the pie chart to a file
    plt.savefig(filename)
    plt.close()


# Define a function to plot pie charts for a given column
def plot_pie_charts_for_column(df, file_prefix, column, name):
    # Sort column alphabetically
    df = df.sort_values(by=column, axis=0)

    # Aggregate the column counts across the entire dataset and plot a pie chart
    # Don't sort by frequencies, retain sorted order (alphabetical) of column
    column_counts = df[column].value_counts(sort=False)
    plot_pie_chart(column_counts, column, f'{name} by Number of Examples', f'plots/{file_prefix}/{column}_by_examples_pie_chart.png')

    # Count the number of unique datasets for each column value and plot a pie chart
    column_dataset_counts = df.groupby(column)['dataset'].nunique()
    plot_pie_chart(column_dataset_counts, column, f'{name} by Number of Datasets', f'plots/{file_prefix}/{column}_by_datasets_pie_chart.png')


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
    plot_and_save_histogram(instruction_length_stats['mean'].dropna(), 'Distribution of Instruction Lengths (Mean)',
                            file_prefix, 'instruction_length_histogram')
    plot_and_save_histogram(prompt_length_stats['mean'].dropna(), 'Distribution of Prompt Lengths (Mean)',
                            file_prefix, 'prompt_length_histogram', cap=15000)
    plot_and_save_histogram(prompt_length_stats['mean'].dropna(), 'Distribution of Prompt Lengths (Mean)',
                            file_prefix, 'prompt_length_histogram', cap=5000)
    plot_and_save_histogram(answer_length_stats['mean'], 'Distribution of Answer Lengths (Mean)', 
                            file_prefix, 'answer_length_histogram', cap=1000)

    # Flatten the columns
    df_flattened = df.explode(one_entry_cols)

    # Plot and save the pie charts
    plot_pie_charts_for_column(df_flattened, file_prefix, "task_type", "Task Type")
    plot_pie_charts_for_column(df_flattened, file_prefix, "jurisdiction", "Jurisdiction")


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