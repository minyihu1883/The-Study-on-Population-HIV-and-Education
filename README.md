# The-Study-on-Population-HIV-and-Education
We divided the study into three parts. In Part I, we visualize the change of population, growth rate, death rate, population age structure and the distribution of population across the world. Then we use moving average, weighted moving average, ARIMA and Prophet to predict the growth of population. Comparing these time series methods, Prophet is proved to be the optimal option for forecasting population. In part 2, we are trying to find the factors that will affect the the percentage prevalence of HIV for the main labor force, age group 15–49 using unsupervised techniques and building classifiers. These factors may include population demographic factors, environmental factors, education health awareness, etc., such as mortality rate, water source quality, knowledge of immunization, respectively. For the education model, almost all the primary school enrollment rate is higher than the secondary school enrollment rate and the tertiary school enrollment rate from year 1960 to 2015. And there are lots of factors will influent the rate of school enrollment. The most significant factors are mortality and birth rates, which means the education and the population are closely related.
Our data set: Health Nutrition and Population dataset

Code Link : https://colab.research.google.com/drive/1BFT8ooRc86L9NMYYzB4Wf4MwJj9ti0Eu#scrollTo=zXvi0kWZoj14

Part 1 Dive into Demographic Features![Uploading 1_GkMhKOigcZBTfaWxoq9G-Q.png…]()


Source: https://www.business-standard.com/article/international/covid-19-impact-us-population-growth-smallest-in-at-least-120-years-120122300187_1.html
Population has grown at such a rapid speed ever since the World War II that we will reach Earth’s carrying capacity in decades. The spectacular growth of world population has been closely associated with economic progress and growing prosperity while it also has some negative effect on development. In this part, we will explore features of population, including population age structure, population distribution, etc..

First of all, let us take a look at population growth since 1960.


Population grows at a stable speed from 1960. Male population exceeds female population in recent years though there is not much difference between them.

Given the fact that population grows stably, we want to see two rates which have impact on growth rate. That is birth rate and death rate.


Noticing that birth rate is higher than death rate, which is the reason why population keeps growing. Also, birth rate and death rate gradually declines. Mortality reductions have been associated with two factors: the frequent conquest of cardiovascular disease in the elderly and the prevention of death caused by low birth weight in infants. Traditional killers such as pneumonia in the young also have continued to decline, but mortality from these causes was already so low that further improvements did not add greatly to overall longevity.

Besides birth rate and death rate of the world, we explore birth rate and death rate in different continents.



Based on the plot above, we find that the death rate and birth rate is highest in Africa and the birth rate is lowest in Europe. This conclusion accords with our common sense.

Then we want to explore the age structure of population. A simple and direct way to exhibit population age and gender structure is demographic pyramid.




Then we want to see the spacial distribution of population around the world.


After visualization, we want to predict population growth in next a few years.

Moving Average
Moving average is a simple method, which can be a baseline of our prediction. In this problem, we choose a window size of 3, which means the prediction of next year is the mean of 3 years’ population before it.

Basically, we intend to use 2 statistics to evaluate our prediction: Accuracy and Bias. The way we compute accuracy and bias is as follows:


We use data after 2000 as our test set. Then let us take a look at the result of moving average method:


Basically, the prediction is not so bad. But there is not much variation in population data, we can absolutely do better.

2. Weighted Moving Average

It is not hard to notice that the population continuously grows at a relatively stable speed, so we can multiply a factor and make the prediction larger than simply moving average. The factor is learned by past few years growth rate.

Let us see the result of this method.


This method successfully captures the growth trend of population.

3. ARIMA

In this part, we follow the steps of ARIMA. Even though we are confident that the time series data is non-stationary, we still follow the steps of ARIMA. Firstly, we check whether the data is stationary.


As we expect, with a p-value of 0.997, the data is non-stationary.

Now let us take a look at the result of ARIMA.



The result of ARIMA is pretty good while it tends to underestimate the growth of population.

4.Prophet

Now we come to our last method: Prophet. Let us directly see its result.


Both bias and accuracy is desirable for our forecast.

Now we summarize our model as below:


So we choose Prophet to predict population.


Part 2: Exploring HIV
What kind of aggregate indicators of a country would possibly affect the prevalence of HIV? Can we actually predict one region’s prevalence of HIV based on the forcasted value of these indicators?


image source: link

Part 2 outline:

We selected a data set from the world bank with worldwide chronologic HIV data with social and demographic features from 1960 to 2015. When it comes to data analysis, the foremost and very important part will be EDA and data wrangling in the first section.
In this part, we are trying to find the factors that will affect the the percentage prevalence of HIV for the main labor force, age group 15–49. These factors may include population demographical factors, environmental factors, education health awareness, etc., such as mortality rate, water source quality, knowledge of immunization, respectively.
We would conduct heuristic data mining to extract patterns and also build a classifier for HIV prevalence prediction based on the values of the selected features.
First, let’s take a look at the HIV prevalence trend:


As seen from the graph, the worldwide HIV prevalence reamains steady in the past decade.


For the past decades, the prevalence of HIV for the age group of 15–49 has a wide gap between male and female with female prevalence highly above male’s. The prevalence rate has drop down in the recent years for both genders.

2.1 EDA, Data wrangling and Feature selection for analyzing HIV

step 1: Having a look at our dataset — what features do we have? how many na values are there?

Let’s see the structure of the original data.


The original dataset has multiindex: year and country. We want our features to be in the form of columns. In the previous part, we re-organized the dataframe into the following form with features being pivoted:


Before doing feature selection, we wanna see what features do we have.This will be good for applying domain knowledge. The following graph shows a section of our features set.


By looking at all the features we have, we want to explore what things will possibly affect the prevalance of HIV. Let’s take a closer look on the data for our y variable(label) that we are interested in, namely ‘Prevalence of HIV, total (% of population ages 15–49)’. As we can see, there are many NA values.


step 2: Dealing with NA values: row and column wise dropping, inputation

Found out we can’t drop with any NA. Let’s focus on the NA values of our labels and drop all rows without labels.


Luckily, we have 3198 data points with labels.

By removing the columns with over 90% missing data, we obtain:


There are still 113 feature columns with over 90% data being filled in.

step 3: inputation using K-nearest neighbors

Which inputation method should we use on our dataframe? Let’s have a closer look for a specific country.


As seen above, the example of Zimbabwe’s feature — “life expectancy at birth” only misses the data entry of year 2015. So for choosing the imputation method, KNN is a good fit. By choosing KNN, we are able to find the closes year of the feature values and do a mean approximation to fill in the missing data, which is reasonable because if we know year 1 and year 3’s data for specific feature, year 2’s data will possibly lie in between.

Step 4: feature extraction using correlation matrix

We plot out the Pearson correlation matrix against our interested variable y and have a look at the correlation between each feature variable X’i and y.


The graph might not be clear to see. Let’s find out the top most negative and positive correlated features by ranking our correlation matrix.


The negative correlated features make sense based on our common sense. Since we are interested in the prevalence of HIV for the age group of 15–49. Survival to 65 is likely to be the most negatively correlated feature because it’s not likely that a region with higher HIV prevalence for age15–49 reuslt in a higher survival ratio to age 65. The victims probably can’t make it till his/her 65.

The results also indicate a negative correlation for the environmental and social factors like water source(hygine), sanitation facilities, immunization, urban population and life expectancy.


For the positive correlated features, The result indicates a positive correlation between the prevalence of HIV to the motality rate, unemployment rate, birth rate and rural population, etc. We will later select these features for learning.

Noticed that we set out objective label specificly for the age group of 15–49, the first several rows of other age groups’ HIV prevalence (highly correlated) should be excluded for controlling factors. When doing prediction, we expect we have no other knowledges of HIV prevalence.

2.2 Unsupervised learning: PCA, Clustering
We can see that some of the features are represented sepereately by sex and age group while they are actually representing the same aspect of this feature. For example, the feature suvival to age are seperated into many groups and thus expanding the number of features. In this situation, PCA really comes into handy for dimentionality reduction by extracting the useful information while negating the redundant. PCA also takes into account for multicolinearity between features.

Let’s find out the optimal number of principle components.


Plot the explained variance ratio against number of components:


We set out number of components to be 20 and redo the PCA. Plot out the first 2 components:


Let’s perform Agglomerative Clustering on our first 2 principle components to see if there are any natural groupings. The code in the next block is refering to https://scikitlearn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py.




The clustering is heuristic in a sense that it is not completely necessary to cluster the 2 components into groups with all our data comes with labels.

2.3 Classification
The whole setting of our classifier: our dataset presents the data samples as the feature values of a specific country in a specific year. The feature values thus deliver an aggregate representation of the regional situation in a specific time. By using classification, we are able to obtain a classifier which takes all the feature values to generate the percentage HIV prevalence of age group 15–49 (the main labor force). If the accuracy is promising, we could use it to fill out the missing data for other countries’ label or possibly predict a new region’s HIV % prevalence based on the features obtained.

2.3.0 Defining labels

Before digging into the model, we notice that the label % prevalence is a floating number instead of categorial. we need to decide a threshold for seperating these percentage values(0–1) into two groups with labels ‘0’ and ‘1’. We first explore the statistics of our y values and draw a boxplot.


We could see the distribution of the HIV prevalence across region and time are very scattered with lots of high HIV exposure outliers. The mean is descently high because of these high value outliers while the mode is low.

We will adopt the Low Precision/High Recall principle: we want to reduce the number of false negatives without necessarily reducing the number of false positives, we choose a decision value that has a low value of Precision or a high value of Recall. This is because in our case, we do not want any region with high HIV exposure to be classified as low level exposure without giving much heed to if the region is being wrongfully classified as high level. This is because a region being viewed as high exposure and be further investigated to exclude the true exposure while if a the real high prevalance region is not captured by the model, it is likely that the situation will be neglected with bad consequences.

2.3.1 Model validation using k-fold cross-validation

For this part, we will use k-fold cross-validation on our training set to find out the optimal classifier on our data. With python sklearn package, we are able to initiate the classifier instances, perform fit and transform on our dataset, using GridSearchCV for choosing hyperparameters and get value scores using k-fold cross validation.

K-nearest neighbors


Logistic regression


Random forest

Let’s find out the best hyperparameters for our model


Linear Discriminant Analysis


2.3.2 Model selection and testing by test set

Choosing method: based on the results of our k-fold cv, we will choose random forest as our classifier. Noticed that random forest itself is model using much computation power. Nevertheless, our data set is not very large with only 2684 entries so we will use it anyways. Let’s build a new random forest for our whole training set and see it’s performance on the test set.


The results indicate that the RF classifier is well-built!!! Let’s use it for prediction.

2.3.3 Further application of our classifier

Recall that from the previous part, we deliberately select those rows with missing values being filled as **df_hiv_inpute**. We did not use this set for unsupervised learning and classification because of its missing values. Here, we assume these data entries are potential cases collected in the future with labels unknown. We wanna use our classifier to predict the prevalence of HIV based on the collected feature variable values.

Standardizing and dimentionality reduction using the previous PCA instance.


Using the classifier built for predicting HIV prevalence:


The classifier yield a 72% accuracy rate to predict the potential HIV prevalence based on the 48 features collected. Noted that the many data entry are filled with inputation, thus, shrinking the accuracy to roughly 72%, which is acceptable from our perspective.

So far we have covered a mini example of the whole process of building a simple classifier for a predicting a region’s HIV prevalence based on the feature values.

Part III: Education

◉ Introduction:


https://www.cuemath.com/learn/3-types-of-education/

Education has always been a popular topic in modern social studies. There is a lot of information about school enrolment and literacy rates in the original data. Intuitively, we think that enrollment (education) is related to tuition, hardware equipment, quality of teaching, quality of instructors, etc. But a data-based display would be more convincing. According to the observation of the original data and the explanation of some literatures, we find that the influencing factors of education are also related to the birth rate and death rate of the population, the female pregnancy rate and employment rate. For this module, we will use a linear regression model to analyze factors affecting enrolment (education).

◉ Heat Map for the school of enroll enrollments:


Heatmap

As the heat map shows, population growth and decline, expenditure on health, and incidence of disease are all important factors influencing overall school attendance (unrestricted phase).

◉ The school enrollment trends in some certain countries:



The school enrollment of UK and US

There are many empty values in the original data. But we can still see a very clear trend(Take the US and UK for example). In each country, primary schools have the highest enrollment rate, followed by secondary schools and tertiary schools. From the data visualization, it can be found that no matter the primary school enrollment rate, secondary school enrollment rate or tertiary school enrollment rate generally shows an increase with time(The year from 1960–2015). Among more than 300 features, the primary school enrollment rate has less relevant features than middle school enrollment rate and higher education enrollment rate. This may have something to do with social security and welfare in various countries.

◉ The impacts for the school of enrollments:

Primary:


PCA for the Primary Part

The total most correlated features’ number: 44

Secondary:


PCA for the Secondary Part

The total most correlated features’ number: 114

Tertiary:


PCA for the tertiary Part

The total most correlated features’ number: 100

From these data:

It is concluded (after processing with Corr() function and PCA) that the characteristics most correlated (from strong to weak) with primary school enrolment rate are mortality rate, birth rate, medical expenditure, vaccination rate, and urban population. And the characteristics that are most correlated (from strong to weak) with secondary school enrolment rates are total population, health facilities, mortality, personal health expenditure and vaccination rates. And the characteristics most correlated with tertiary school enrolment (from strong to weak) are the total population, health expenditure, mortality, and incidence of diseases such as tuberculosis and non-pregnant women anaemia.

