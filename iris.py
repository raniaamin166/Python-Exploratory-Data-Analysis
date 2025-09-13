IRIS DATASET - EXPLORATORY DATA 
ANALYSIS (EDA)
 STEP 1: LOAD THE IRIS DATASET
 Load the Iris Dataset
 The Iris dataset is a classic dataset in machine learning and statistics. It contains 150 samples 
from three species of Iris flowers (setosa, versicolor, virginica), with four features: sepal 
length, sepal width, petal length, and petal width.
 import seaborn as sns
 import pandas as pd
 Load dataset from seaborn
 df = sns.load_dataset('iris')
 Display first 5 rows
 df.head()
  
 STEP 2: BASIC DATASET INFORMATION
 Basic Dataset Information
 This step provides a general overview of the dataset including shape, missing values, and 
data types.
 print("Dataset Shape:", df.shape)
 print("\nData Types and Info:")
 print(df.info())
 print("\nMissing Values:")
 print(df.isnull().sum())

 STEP 3: DESCRIPTIVE STATISTICS
 Descriptive Statistics
 Summary statistics help us understand the central tendency, dispersion, and shape of the 
distribution of the datasets features.
 Descriptive statistics summarize and describe the main features of a dataset using numerical 
values. These values provide insights into:-Central tendency (where data tends to cluster)-Dispersion (how spread out the data is)-Shape of the distribution (symmetry, skewness, etc.)
 Why Is This Important?-Helps compare species across features.-Identifies outliers or unexpected values.-Prepares you for visualization and feature engineering.-Enables better data-driven decisions before modeling.
 Insights:
-Setosa has significantly smaller petal lengths and widths compared to 
others.-Virginica tends to have the largest measurements overall.-Versicolor shows intermediate measurements for all features, with 
petal dimensions larger than Setosa but smaller than Virginica-The standard deviation tells us how consistent the measurements are 
within each species.
 # Basic statistics
 df.describe()
 # Group-wise mean
 print("\nGroup-wise Mean:")
 print(df.groupby('species').mean())
 # Group-wise standard deviation
 print("\nGroup-wise Standard Deviation:")
 print(df.groupby('species').std())
 
 STEP 4: VISUALIZE FEATURE DISTRIBUTIONS
 Visualize Feature Distributions
 Histograms and boxplots help visualize the distribution and spread of features.
 Histogram
 A histogram is a plot that shows the distribution of a single numeric feature by dividing the data 
into intervals (bins) and counting how many data points fall into each bin.
 In the Iris Dataset:
 When we plot histograms for features like sepal_length, petal_length, etc., we are visualizing:
-How the values are distributed (e.g., normal, skewed)-Where the concentration of values lies
 Example Insight:
 In a histogram of petal_length:-Setosa values are concentrated around 1,2 cm-Versicolor values appear in the 3,5 cm range-Virginica values fall around 5,7 cm
 This clearly shows class separation in petal length.
 Boxplot
 A boxplot (or box-and-whisker plot) is a statistical graph that summarizes the distribution of a 
numeric feature using 5 key values
 It also highlights outliers 
 In the Iris Dataset:
 When we plot a boxplot of petal_length across species:-We can compare the central tendency and spread between species.-It visually shows how distinct the species are based on the feature.-Outliers, if any, will be clearly visible.
 Example Insight:-Setosa has a very small and tight box (low variation).-Versicolor shows moderate spread.-Virginica has the largest values and some variability.

Histogram vs Boxplot | Feature | Histogram | Boxplot | | -------------- | ------------------------------------------- | -------------------------------------------- | | Focus | 
Frequency distribution | Summary statistics (median, quartiles, etc.) | | Best for | Seeing overall 
shape (normal, skewed, etc.) | Comparing distributions across groups | | Shows outliers |  No |  
Yes | | Shows median |  No (only visually estimated) |  Yes |
 import
 matplotlib.pyplot 
# Histograms
 plt.figure(figsize=(
 as plt
 10, 8
 ))
 df.hist(figsize=(
 10, 8
 plt.tight_layout()
 plt.show()
 ),color=
 'pink')
 # Boxplot for one feature by species
 sns.boxplot(x=
 'species'
 , y=
 'petal_length'
 , data=
 df)
 plt.title(
 "Boxplot of Petal Length by Species")
 plt.show()

STEP 5: EXPLORE PAIRWISE RELATIONSHIPS
 Explore Pairwise Relationships
 Pairplots and scatter plots help identify patterns and separations between species.
 Pairplot
 A pairplot is a grid of scatterplots that shows the pairwise relationships between multiple 
features in the dataset. It’s a very handy way to explore how features interact and whether 
patterns emerge by class.
 Key Features:-It shows scatterplots for all possible feature pairs.-Diagonal plots often show histograms or KDEs (distributions) of each 
feature.-It can be color-coded by a categorical variable like species using 
hue.
 In the Iris Dataset:
 This helps us:
-Visualize class separation (e.g., petal features clearly separate 
species)-Detect correlations between features-Spot potential outliers
 Example Insight:-Petal length and petal width show strong linear separation between 
species.-Setosa is completely separated in multiple pairwise plots — very 
distinguishable.-Versicolor and Virginica overlap more but still show some separation.
 Scatterplot
 A scatterplot shows the relationship between two numeric variables. Each point represents a 
single data observation.
 Example Insight:-Setosa forms a tight cluster at the bottom-left.-Virginica spreads widely in the top-right.-Versicolor appears in-between, sometimes overlapping both.
 Pairplot vs Scatterplot
 Feature Pairplot Scatterplot
 Compares All pairs of features A specific pair of features
 Best for Overall relationships & class separation Detailed analysis of a selected pair
 Shows 
diagonal
 Histograms/KDE plots  No
 Hue 
supported
 Yes Yes
 # Pairplot
 custom_palette_pairplot = {"setosa": "pink", "versicolor": "purple", 
"virginica": "yellow"}
 sns.pairplot(df, hue='species', palette=custom_palette_pairplot)
 plt.show()
 # Scatterplot
 custom_palette = {"setosa": "green", "versicolor": "grey", 
"virginica": "orange"}
sns.scatterplot(x=
 'petal_length'
 data=
 df, palette=
 , y=
 'petal_width'
 , hue=
 custom_palette)
 plt.title(
 "Petal Length vs Petal Width")
 plt.show()
 'species', 
STEP 6: CORRELATION ANALYSIS
 Correlation Analysis
 Correlation heatmaps display the linear relationships between numeric features.
 corr =
 df.corr(numeric_only=
 sns.heatmap(corr, annot=
 True)
 True
 , cmap=
 'coolwarm')
 plt.title(
 "Correlation Heatmap")
 plt.show()

 1) Petal length & petal width (correlation = 0.96)
 Strong positive correlation
 Strong positive correlation
 Very weak negative correlation
 Moderate negative correlation
 Moderate negative correlation
 Very strong positive correlation-These two features are almost perfectly linearly related.-This indicates they convey similar information and are excellent 
predictors for species classification.
 2) Sepal length & petal length (correlation = 0.87)-Also highly correlated, meaning longer sepals are often accompanied 
by longer petals.
 3) Sepal width has weak or moderate negative correlations with other features:
-It tends to slightly decrease as the other features increase.-This suggests sepal width behaves differently from the other 
measurements.
 Color Interpretation (with coolwarm colormap):-Red indicates strong positive correlation (closer to +1)-Blue indicates strong negative correlation (closer to -1)-White/light indicates no correlation (around 0)
 The reddest square is between petal_length and petal_width (≈ 0.96).
 The bluest ones are the correlations involving sepal_width with petal features (≈ -0.4).
 STEP 7: GROUPED VISUALIZATIONS
 Grouped Visualizations
 Violin plots and swarm plots show feature distribution and individual data points grouped by 
species.
 Violin Plot
 A violin plot combines aspects of a boxplot and a kernel density plot. It shows the distribution 
shape of a numeric variable for different categories, along with summary statistics.
 Insights:-Wider parts of the violin indicate more data points at that value.-You can see if the data is skewed or bimodal within a species.-It gives more detail than a boxplot about distribution shape.
 Swarm Plot
 A swarm plot is a scatter plot where data points are adjusted (jittered) along the categorical axis 
so they don’t overlap. It shows all individual data points and their distribution.
 Features:-Each point represents one observation.-Points are spread out (like bees in a swarm) to avoid overlap.-Useful for visualizing data density and outliers.
 Insights:
-Shows exact data points (not summarized).-Reveals clusters and potential outliers clearly.-Complements violin or boxplots for granular understanding.
 Violin Plot vs Swarm Plot
 Feature
 Violin Plot
 Swarm Plot
 Shows 
distribution
 Shows individual 
points
 Good for
 Yes (density estimate shape)
 Often includes median & quartiles 
(boxplot inside)
 Understanding shape & spread of data
 Visual complexity Medium (smooth shapes)
 # Violin plot
 sns.violinplot(x=
 'species'
 , y=
 No (individual points only)
 Shows every data point
 Seeing exact data point 
distribution
 Higher (many points, can be 
dense)
 'sepal_length'
 , data=
 df)
 plt.title(
 "Violin Plot of Sepal Length by Species")
 plt.show()
 # Swarm plot
 sns.swarmplot(x=
 plt.show()
 'species'
 , y=
 'sepal_length'
 , data=
 plt.title(
 "Swarm Plot of Sepal Length by Species")
 df)

Step 8: Summary of Key Findings
 Feature Separation: Petal length and petal width show the most separation between species, 
especially between setosa and the other two.
 Best Visualizations: Pairplots and scatter plots provided the clearest insights into the 
relationship between features and species separation.
 Interesting Patterns: Setosa species shows distinct characteristics in both sepal and petal 
features, while versicolor and virginica slightly overlap but are still distinguishable.
 These insights can help build classification models to predict species using features like petal 
length and width.
 BONUS CHALLENGE: COMBINED VISUALIZATIONS
 Combined Visualizations
 # Create a FacetGrid with multiple subplots by species
 Spec1=
 sns.FacetGrid(df, col=
 "species")
 Spec1.map_dataframe(sns.histplot, x=
 plt.show()
 Spec2=
 "petal_length"
 sns.FacetGrid(df, col=
 "species")
 Spec2.map_dataframe(sns.histplot, x=
 , color=
 'pink')
 "sepal_length"
 , color=
 'purple')
plt.show()
 Spec3=
 sns.FacetGrid(df, col=
 "species")
 Spec3.map_dataframe(sns.histplot, x=
 plt.show()
 Spec4=
 plt.show()
 "petal_width"
 , color=
 'red')
 sns.FacetGrid(df, col=
 "species")
 Spec4.map_dataframe(sns.histplot, x=
 "sepal_width"
 , color=
 'yellow')
# Custom subplot combining histogram and scatterplot
 fig, axes =
 plt.subplots(1
 df[df[
 'species'
 , 2
 , figsize=(
 12, 5
 ] == 
'setosa'
 ))
 ].hist(column=
 'petal_length'
 color=
 'purple')
 sns.scatterplot(x=
 data=
 df, ax=
 'sepal_length'
 axes[1
 ])
 axes[0
 axes[1
 , y=
 'petal_width'
 , ax=
 axes[0
 ], 
, hue=
 'species', 
].set_title(
 "Histogram: Petal Length (Setosa)")
 ].set_title(
 "Scatter: Petal Length vs Width")
 plt.tight_layout()
 plt.show()
