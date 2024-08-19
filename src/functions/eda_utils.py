import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Check the normality of a feature's ditribution across different categories using the Shapiro-Wilk test and Q-Q plots
def normality_check(ver_plots, hor_plots, splitter, feature, df):
    """
    This function creates subplots to display Q-Q plots for each category defined by the 'splitter' variable.
    It also prints the results of the Shapiro-Wilk test for each category.

    Parameters:
    -----------
    ver_plots : int
        The number of vertical plots (rows) in the subplot grid.
    
    hor_plots : int
        The number of horizontal plots (columns) in the subplot grid.
    
    splitter : str
        The column name used to split the data into categories. Q-Q plots and Shapiro-Wilk tests are performed for each unique value in this column.
    
    feature : str
        The name of the feature/column for which the normality is being checked.
    
    df : pandas.DataFrame
       The DataFrame containing the data to be analyzed.

    Returns:
    --------
    None
        The function does not return any values. It prints the results of the Shapiro-Wilk test and displays the Q-Q plots.

    """
    # Creating the subplots
    fig, axes = plt.subplots(ver_plots, hor_plots, figsize=(10, ver_plots * 3))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, split in enumerate(df[splitter].unique()):
        # Shapiro-Wilk Test
        shapiro_test = stats.shapiro(df[df[splitter] == split][feature])
        print(f"Shapiro-Wilk Test for {feature} for {split}: Statistic={shapiro_test[0]:.3f}, p-value={shapiro_test[1]:.3f}")
        
        # Q-Q Plots
        stats.probplot(df[df[splitter] == split][feature], dist="norm", plot=axes[i])
        axes[i].set_title(f'Q-Q plot for {split}')
    
    plt.tight_layout()
    plt.show()



def kruskal(splitter, feature, df):
    """
    This function performs Kruskal-Wallis H test to check statistical differences on a specific feature for three or 
    more different groups defined by the splitter.

    Parameters:
    -----------
    splitter : str
        The column name used to split the data into groups. Kruskal-Wallis H test is performed for each unique value in this column.
    
    feature : str
        The name of the feature/column for which the difference is being checked.
    
    df : pandas.DataFrame
       The DataFrame containing the data to be analyzed.

    Returns:
    --------
    None
        The function does not return any values. It prints the results of the Kruskal-Wallis H test.

    """
    # Extract feature values for each group
    overall_ratings = [df[df[splitter] == split][feature] for split in df[splitter].unique()]

    # Perform Kruskal-Wallis H Test
    kruskal_stat, kruskal_p = stats.kruskal(*overall_ratings)
    print(f"Kruskal-Wallis Test Results:\nStatistic={kruskal_stat:.3f}, p-value={kruskal_p:.3f}")


def correlation_coef(feature_a, feature_b, df):
    """
    This function calculates Pearson's correlation coefficient between two features.

    Parameters:
    -----------
     feature_a, feature_b : str
        The name of the features/columns for which the correlation coeffiecient is being calculated.
    
    df : pandas.DataFrame
       The DataFrame containing the data to be analyzed.

    Returns:
    --------
    None
        The function does not return any values. It prints Pearson's correlation coefficient for the specified features.

    """
    # Calculating the correlation coefficient
    correlation = df[[feature_a, feature_b]].corr().iloc[0, 1]
    print(f"Correlation between '{feature_a}' and '{feature_b}': {correlation:.3f}")


def plot_percentage(splitter, feature, df, colour = 'skyblue'):
    """
    This function creates a bar chart where every bar represents the percentage of a specific feature within each value of 
    a category defined by the splitter.
    It allows to specify a determined colour list.

    Parameters:
    -----------
    splitter : str
        The column name used to split the data into groups. Percentages are displayed for each unique value in this column.
    
    feature : str
        The name of the feature/column for which the percentages are calculated.
    
    df : pandas.DataFrame
       The DataFrame containing the data to be analyzed.

    colour : list
        The list of colours generated from a colour map (dictionary).

    Returns:
    --------
    None
        The function does not return any values. It dislays the percentage bar charts.

    """
    # Calculating percentage of 'feature' reviews by a certain 'splitter'
    percent = (df.groupby(splitter)[feature].mean()*100).sort_values(ascending=False)

    # Plotting the percentage
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = percent.plot(kind='bar', ax=ax, color=colour)
    ax.set_title(f"Percentage of {feature} Reviews by {splitter}")
    ax.set_xlabel(splitter)
    ax.set_ylabel(f'Percentage of {feature} Reviews')

    # Add percentage labels on top of each bar
    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.1f}%', ha='center', va='bottom')

    plt.show()


def correlation_matrix(features, df):
    """
    This function displays the correlation matrix of the specified features as a heatmap.

    Parameters:
    -----------
    features : list
        List containing the names of the features to include in the correlation matrix.
    
    df : pandas.DataFrame
       The DataFrame containing the data to be analyzed.

    Returns:
    --------
    None
        The function does not return any values. It dislays the correlation matrix.

    """
    # Obtaining correlation coefficients of specific features
    correlation_matrix = df[features].corr()

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Rating Aspects')
    plt.show()


# Combined Box Plot of numerical features
def plot_combined_boxplot(split, features, df):
    """
    This function displays Box plots for the specified features in a specific group(split).
    This function will be used within the eda function described next.

    Parameters:
    -----------
    split : str
    The specific group that is being analyzed.

    features : list
        List containing the names of the features for which the Box plots are displayed.
    
    df : pandas.DataFrame
       The DataFrame containing the data to be analyzed.

    Returns:
    --------
    None
        The function does not return any values. It dislays the Box plots of the specified features.

    """
    # Plotting the Box Plots
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[features])
    plt.title(f'Box Plot of Numerical Features for {split}')
    plt.xlabel('Features')
    plt.ylabel('Ratings')
    plt.show()


# Function to perform some EDA on different features for specific subsets
def eda(splitter, split, features, df):
    """
    This function performs some EDA on specific features for specified subsets.
    It extracts the subset for a specific group (split) within a specific category (splitter).
    It displays the descriptive statistics, Box plots, and correlation matrix for the specified features.

    Parameters:
    -----------
    splitter : str
        The name of the column that is being analyzed.
    
    split : str
        The specific value within the column that is being analyzed.
    
    features : list
        List containing the names of the features for which the EDA is performed.

    df : pandas.DataFrame
       The DataFrame containing the data to be analyzed.

    Returns:
    --------
    None
        The function does not return any values. It dislays the descriptive statistics, Box plots, and
        correlation matrix of the specified features.

    """
    # Extracting the subset
    df_eda = df[df[splitter]==split]

    # Displaying descriptive statistics
    print(df_eda[features].describe())

    # Box Plot of numerical features
    plot_combined_boxplot(split, features, df_eda)
    
    # Correlation matrix
    correlation_matrix(features, df_eda)


def traveller_type(split, df):
    """
    This function performs some EDA on different types of travellers for a specific cabin/class.
    It displays the descriptive statistics for each type of traveller in the specific class regarding 'Overall Rating'.
    It displays then a pie chart with the percentage of each type of traveller within the class.
    It finally displays a bar chart with the percentage of recommended reviews for each type of traveller within the class.

    Parameters:
    -----------
    split : str
        The specific Class that is being analyzed.

    df : pandas.DataFrame
       The DataFrame containing the data to be analyzed.

    Returns:
    --------
    None
        The function does not return any values. It dislays the descriptive statistics, pie chart, and percentages bar chart.

    """
    # Summary statistics for Overall Rating depending on the type of Traveller
    summary_stats = df.groupby('Type of Traveller')['Overall Rating'].describe()
    print(f'Descriptive statistics of Type of traveller in {split}')
    print('\n',summary_stats)
    
    # Calculate the percentage of each type of traveller
    traveller_counts = df['Type of Traveller'].value_counts()
    traveller_percentages = traveller_counts / traveller_counts.sum() * 100

    # Define a color map
    color_map = {
    'Solo Leisure': 'lightblue',
    'Business': 'red',
    'Family Leisure': 'green',
    'Couple Leisure': 'orange',
    }

    # Traveler types for pie chart
    traveller_types = traveller_percentages.index.tolist()
    
    # Generate the color list for the pie chart
    colors = [color_map[traveller] for traveller in traveller_types]

    # Traveler types for percentage bar chart
    type_percent = (df.groupby('Type of Traveller')['Recommended'].mean() * 100).sort_values(ascending=False)
    
    # Generate the color list for the bar chart
    colors2 = [color_map.get(traveller) for traveller in type_percent.index]

    # Plotting the pie chart
    plt.figure(figsize=(5, 5))
    plt.pie(traveller_percentages, labels=traveller_percentages.index, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f'Distribution of Types of Travellers in {split}')
    plt.show()

    # Plot percentage of Recommended Reviews
    plot_percentage('Type of Traveller','Recommended',df,colors2)