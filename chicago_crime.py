
# coding: utf-8

# # STA141C Final Project: Crime in Chicago
# ### A clustering analysis and visualization for crime in Chicago from 2005-2017
# By: Joe Akanesuvan, Navid Al Nadvi, Sailesh Patnala
# 
# This analysis will determine the the components that can be use to determine the crime type in Chicago. We hope that through a thorough analysis of the dataset, we will able to cluster the observations into different crime types. By doing so we wish that we can determine important features, such as crime location, which can be use to prevent certain types of crime from happening.
# 
# In addition, since we are dealing with categorical features, we would also like to analyze the effectiveness of different clustering algorithm on such a data. Since we were already given the type of crimes within the dataset, we could easily check the accuracy of each clustering algorithm. Although this is an unconventional method, since clustering is supposed to be an unsupervised learning method, it would provide us with best accuracy measurement for the different algorithms.
# 
# The clustering algorithm that we will take into account are:
# * KMean
# * KMode
# * DBSCAN

# In[62]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from kmodes import kmodes
from kmodes import kprototypes
get_ipython().magic('matplotlib inline')


# In[63]:

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[64]:

init_notebook_mode(connected=True)


# In[65]:

cf.go_offline()


# Since there are a lot invalid observations in the CSV file from 2001 to 2004, we will omit the data from the analysis. We will conduct an analysis on the crimes from 2005 to 2017, while dropping any duplicate observations, determine by the ID and Case Number, or any observations with missing data.

# In[66]:

crimes1 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)
crimes2 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)
crimes3 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)
crimes = pd.concat([crimes1, crimes2, crimes3], ignore_index=False, axis=0)

crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)


# ## Understanding the Dataset

# Most of the features are specific to a given observation, it is highly unlikely for different crimes to happen at the exact same Block, and Location. Furthermore, features such Community Area and Ward are very similar to each other in nature, it is possible to drop many of the features to help ensure that the clustering algorithm will only contain useful features which can be use to accurately group the observations into a cluster.

# In[67]:

crimes.head()


# In[68]:

crimes.info()


# There are unnecessary columns which would not be able to use for any relevant visualization nor any clustering algorithm. Remove any columns which may be too specific, or might be too similar to one another.

# In[69]:

crimes.drop(['Unnamed: 0', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate',
             'Updated On','FBI Code', 'Beat','Ward', 'Location', 'District', 'Block',
             'Latitude', 'Longitude', 'Year'], inplace=True, axis=1)


# Since there is Date column, the Date format should be changed to its corresponding Pandas format

# In[70]:

crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p')
crimes.index = pd.DatetimeIndex(crimes.Date)


# In[71]:

crimes.head()


# ## Visualizing the Data

# To find out whether some of these features are relevant or not, we will plot the different features against to see whether a trend, or distribution can be found. In addition, we hope that we might find some relationship between the features which can be use to come up with any interesting, and unforeseen conclusion.

# Firstly, just go get an overall understanding of the big picture, we will plot the total amount of crime against the year, just to give up an overall understanding of the trend of hte data.

# In[72]:

crime_years = pd.DataFrame(crimes.groupby([crimes.index.year]).size().reset_index(name="count"))


# In[73]:

crime_years


# In[74]:

crime_years.iplot(kind="line", x='index', y='count',
                 xTitle='Year', yTitle='Total Crimes',title='Total of Crimes per Year')


# There is a decreasing linear relationship between the total amount of crimes and year. Although the data for 2017 is included in the plot, it should not be taken into consideration. Since the features are all categorical, it would be difficult to conduct a regression analysis on the dataset, in addition, since we do not have any additional features related to the total crimes, it will prove difficult to create multiple regression models for evaluation.

# With an understanding of the big picture, we will then move on to see whether any crime is more likely to happen than other. If so, there may be some relationship between such a crime type and some other features, which may be use to efficiently cluster the observations.

# In[75]:

crime_types_count = pd.DataFrame(crimes.groupby(["Primary Type"]).size().reset_index(name="count"))
crime_types_count.sort_values('count', ascending=True, inplace=True)


# In[76]:

crime_types_count.tail()


# In[77]:

data = [
    go.Bar(
        x=crime_types_count['count'],
        y=crime_types_count['Primary Type'],
        orientation='h',
    )
]

layout = go.Layout(
    title='Total Crime Types',
    yaxis=dict(
        title='Crime Types',
        tickfont=dict(
        size=8,
        color='black',
        ),
    ),
    xaxis=dict(
        title='Count',
    ),
    margin=go.Margin(
        l=200,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='total-crime-types')


# The plot shows that there are crime types which are more likely to occur than others. Due to this observation, our initial hypothesis that there exist features which can be use to determine the cluster is strengthen. Hopefully, through other visualization we would be able to find any other relevant features. 

# To avoid a huge number of clusters, we will only take the top five crime types into account. Fortunately, there is already an "Other Crimes" category which can be use to group up all the observations which do not fit into the given crime type. By reducing the clusters and the observation, we are hoping that the runtime for the clustering algorithm will also decrease.

# In[78]:

topCrimes  = list(crimes['Primary Type'].value_counts()[0:5].index)


# In[79]:

crimes = crimes[crimes['Primary Type'].isin(topCrimes)]
crimes.shape


# In[80]:

crime_location_count = pd.DataFrame(crimes.groupby(["Location Description"]).size().reset_index(name="count"))
crime_location_count.sort_values('count', ascending=True, inplace=True)
crime_location_count.shape


# In[81]:

crime_location_count.tail()


# In[82]:

crime_location_count.head()


# Since there are many features with low counts, we will only take the features with a high number of occurences into account.

# In[83]:

crime_location_count.drop(crime_location_count.index[0:82], inplace=True)


# In[84]:

data = [
    go.Bar(
        x=crime_location_count['count'],
        y=crime_location_count['Location Description'],
        orientation='h',
    )
]

layout = go.Layout(
    title='Total Crimes by Location',
    yaxis=dict(
        title='Location Description',
        tickfont=dict(
        size=8,
        color='black',
        ),
    ),
    xaxis=dict(
        title='Count',
    ),
    margin=go.Margin(
        l=200,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='total-crime-location')


# Like the crime type visualization, we can see that crimes are more likely to happen at some specific location. This might show that some crime may tend to happen at a certain location, and the location may be a useful feature to be use for further analysis.

# Since we have decided that only the 5 top location description will be taken into account during the analysis, we will need to update the crimes dataset to reflect the change.

# In[85]:

topLocations  = list(crimes['Location Description'].value_counts()[0:5].index)


# In[86]:

crimes = crimes[crimes['Location Description'].isin(topLocations)]
crimes.shape


# We hope that our finding on the location may also be extended to the date. We will plot the total amount of crimes against the day of the week to see if crimes may be more likely to happen at a certain day.

# In[87]:

crime_day_count = pd.DataFrame(crimes.groupby([crimes.index.dayofweek]).size().reset_index(name="count"))
days = ["Mon", "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun"]
crime_day_count['index'] = days
crime_day_count


# In[88]:

crime_day_count.iplot(kind="bar", x='index', y='count',
                 xTitle='Day of the Week', yTitle='Total Crimes',title='Total of Crimes per Day')


# From the histogram, we can see that there is no relationship between the day of the week and the total amount of crime. The histogram does not show any specific distribution, and could most likely be omitted from the clustering algorithm

# Although the plot of the day of the week against the total amount of crime did not show any distribution, we will extend this analysis to the month, in hope that there may be some sort of relation between a crime and when it may happen.

# In[89]:

crime_month_count = pd.DataFrame(crimes.groupby([crimes.index.month]).size().reset_index(name="count"))
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
crime_month_count['index'] = months
crime_month_count


# In[90]:

crime_month_count.iplot(kind="bar", x='index', y='count',
                 xTitle='Month', yTitle='Total Crimes',title='Total of Crimes per Month')


# Unlike the initial plot, the plot of the month against the total amount of crime, to our surprises, shows a Gaussian distribution. With crimes more likely to happen over the summer, and less likely to happen during the the end and the beginning of the year, with the exception of January. We hypothesize that there may be an increase in crime over the summer, and January, as they are travel seasons, which may lead to more crimes being committed against tourists. With additional data, this may lead to an interesting research topic. Since there exist a relationship between the month and crime, we decided that it may be useful to keep this feature for further analysis.

# In[91]:

crime_count = pd.DataFrame(crimes.groupby(["Primary Type", crimes.index]).size().reset_index(name="count"))
crime_count.sort_values('Primary Type', ascending=True, inplace=True)
crime_count.reset_index(drop=True, inplace=True)
type_count = crime_count.pivot_table(index='Date' ,columns='Primary Type', values='count').reset_index()
type_count.fillna(0,inplace=True)
type_count.head()


# ### Preprocessing: We will encode the label so they can be normalized

# To run the clustering algorithms, we will first preprocess the data into indicator variables. Any object/string types data will be converted. Any observation with missing data will be fill in with a 0, as they cannot be reliably use during the analysis.

# In[43]:

le = preprocessing.LabelEncoder()
le.fit(crimes['Primary Type'].unique())
crimes['Primary Type'] = le.transform(crimes['Primary Type'])
locationEncoder = preprocessing.LabelEncoder()
locationEncoder.fit(crimes['Location Description'].fillna('0'))
crimes['Location Description'] = locationEncoder.transform(crimes['Location Description'].fillna('0'))


# In[44]:

crimes['Arrest'] = crimes.Arrest.fillna(0).astype(int)
crimes['Domestic'] = crimes.Domestic.astype(int)
crimes['Month'] = crimes.index.month
crimes['Day'] = crimes.index.day
crimes['Hour'] = crimes.index.hour
crimes.reset_index(drop=True, inplace=True)


# In[45]:

crimes.head()


# In addition, we will also need to drop any irrelevant features determine from the visualization process. We will also remove the Primary Type, and the Description feature from the dataset as they will not be use for the analysis.

# In[48]:

X = crimes.filter(['Month', 'Day', 'Hour', 'Location Description', 'Community Area', 'Arrest', 'Domestic']).fillna(0)


# In[49]:

X.head()


# In[50]:

X.shape


# # Clustering Algorithm
# 
# As mentioned before, the clustering algorithms that we will use are:
# 
# * KMean
# * KMode
# * DBSCAN
# 
# Since the dataset are all categorical, we can expect all the algorithm to not perform as accurately as they should. Nevertheless, by comparing the results, we may come up with some interesting find which may determine a specific algorithm which is best for categorical dataset.

# The total amount of clusters can be determined by the unique values left within our reduced dataset.

# In[51]:

unique_clusters = len(crimes['Primary Type'].unique())


# ## KMean Clustering

# Since KMean is, arguably, the most used and most famous clustering algorithm, it only make sense for us to run the algorithm against our dataset. From our research, the kmean clustering should not give out a high accuracy due to the fact that the dataset is entirely made up of categorical features. Even so, we would like to see how this algorithm would perform against the other algorithm.

# In[52]:

kmeans = KMeans(n_clusters=unique_clusters)


# In[53]:

kmeans.fit(X)


# In[54]:

kmeans.cluster_centers_


# In[55]:

d = {"Data":crimes['Primary Type'].tolist(), "KMean":kmeans.labels_}

result = pd.Series(d)


# In[56]:

print("Accuracy: " + str((result['Data'] == result['KMean']).sum() / len(result['Data'])))


# In[43]:

fig , (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))

ax1.set_title('K Means')
ax1.scatter(crimes["Location Description"],crimes["Community Area"], c=kmeans.labels_, cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(crimes["Location Description"], crimes["Community Area"], c=crimes["Primary Type"], cmap='rainbow')


# As we initially believe, the accuracy for KMean, despite its effectiveness, is very low. Although we were able to apply kmean on a categorical dataset, it is probably better not to.

# ## KMode

# From our research, KMode, unlike KMean, would probably be a better, albeit not by much, clustering algorithm for a categorical dataset.

# In[57]:

k_mode = kmodes.KModes(n_clusters=unique_clusters)


# In[58]:

k_mode.fit(X)


# In[59]:

k_mode.cluster_centroids_


# In[60]:

k_modes = {"Data":crimes['Primary Type'].tolist(), "KMode":k_mode.labels_}

result = pd.Series(k_modes)


# In[61]:

print("Accuracy: " + str((result['Data'] == result['KMode']).sum() / len(result['Data'])))


# In[56]:

fig , (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))

ax1.set_title('K Modes')
ax1.scatter(crimes["Location Description"],crimes["Community Area"], c=k_mode.labels_, cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(crimes["Location Description"], crimes["Community Area"], c=crimes["Primary Type"], cmap='rainbow')


# From the result, we can see that KMode is, indeed, a better algorithm for a categorical data. However, the accuracy improvement is minimal at best, and it might be a better option to use a more established library like the KMean library, as opposed to KMode library.

# ## DBSCAN

# For the last algorithm, we will be running the DBSCAN clustering algorithm from sklearn. Like the KMean algorithm, we do not expect a high accuracy out of this algorithm, due to the fact that the distance between categorical variables do not give out any relevant information. It will interesting to see the effective of DBSCAN in comparison to the KMean method.

# In[49]:

db = DBSCAN(eps=0.2)


# In[50]:

db.fit(X)


# In[51]:

dbDict = {"Data":crimes["Primary Type"].tolist(), "DBSCAN":db.labels_}

result = pd.Series(dbDict)


# In[52]:

print("Accuracy: " + str((result['Data'] == result['DBSCAN']).sum() / len(result['Data'])))


# In[57]:

fig , (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))

ax1.set_title('DBSCAN')
ax1.scatter(crimes["Location Description"],crimes["Community Area"], c=db.labels_, cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(crimes["Location Description"], crimes["Community Area"], c=crimes["Primary Type"], cmap='rainbow')


# The accuracy for the DBSCAN algorithm was considerably low, in comparison to KMean, and KMode. So low that we can possibly conclude that, for a categorical data, it is best not to consider using DBSCAN as a clustering algorithm. From the clustering plot, we can see that there is predominantly only one cluster made from the algorithm. 

# # Conclusion
# 
# From the analysis conducted on this dataset, we can conclude that most clustering algorithms are not really effective on categorical dataset. Since categorical dataset do not really represent distance between points, most clustering algorithm cannot effectively group the observations into clusters. Despite such a result, the dataset, however, did show us interesting trends and unique relation. By putting more emphasis on the data visualization we could probably understand more about the dataset, and could possibly come up with other analysis. Although the result for the clustering analysis was interesting in its own right, we, as a team, believe that we could use this dataset for analysis in other areas: time-series, classification on whether a crime would result in an arrest or not (binary classification), and, possibly, a regression analysis on specific crime types.
