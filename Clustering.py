#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tp2_aux import *
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score


# In[2]:


df=pd.read_csv("Features/18_feat.csv")
df


# In[3]:


labels=np.loadtxt("labels.txt",delimiter=",")
df["labels"]=labels[:,1].astype(int)


# In[21]:


df


# ### Feature Selection (ANOVA F-Test)

# In[4]:



X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_new = SelectKBest(f_classif, k=6).fit_transform(X, y) #5 features, might change depending on results


# In[22]:


df_ufilter=df.iloc[:,[0,1,2,7,12,13]]
df_all=df.iloc[:,[0,1,2,7,12,13]]
df_ufilter


# In[6]:


df_ufilter["label"]=y
df_ufilter=df_ufilter[df_ufilter["label"]!=0]
df_ufilter


# In[8]:


import seaborn as sns
df=df[df["labels"]!=0]
sns_fig=sns.pairplot(df,hue="labels", diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'});
sns_fig.savefig("pairplot.png")


# In[9]:


sns.boxplot(data=df_ufilter, orient="h", palette="Set2");


# ### Recursive Feature extraction using suport vector classification
# (If they are easy to classify they are easy to cluster)

# # Clustering

# In[10]:


X=df_ufilter.iloc[:,:-1]
X


# # DBSCAN

# ### Selecting K-distance for K=5

# In[11]:


k=5
y_dummy=np.repeat(0,X.shape[0])

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X, y_dummy)

k_d=neigh.kneighbors(X)
Dist_df=pd.DataFrame(k_d[0])
dist_sorted=Dist_df.iloc[:,k-1].sort_values(ascending=False)
dist_sorted.index=np.arange(0,X.shape[0],1)
plt.figure(figsize=(12,8))
plt.grid()
#plt.axvline(45,ymin=0,ymax=1550/dist_sorted.max())
ax = sns.lineplot(data=dist_sorted, color="coral")


# In[12]:


plt.figure(figsize=(12,8))
plt.grid()
plt.scatter(2, 2.2, s=50)
ax = sns.lineplot(data=dist_sorted[0:20], color="coral")


# ## Distance found of aproximately 1807 so we are going to use 1810 to make sure no center points are rejected

# In[13]:


Distance=2.2


# In[14]:


from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=Distance, min_samples=5).fit(X)
ids=X.index
labels=clustering.labels_
report_file="ex_labels.html"
report_clusters(ids, labels, report_file)


# ## K-Means

# In[15]:


k=4
kmeans = KMeans(n_clusters=k).fit(X)
klab=kmeans.labels_
silhouette_score(X, klab, metric='euclidean')


# In[16]:


y=y[y!=0]


# In[17]:


cm=confusion_matrix(y, klab)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True);


# ## Precision and recall

# ### DBSCAN

# In[18]:


from sklearn.metrics import precision_recall_fscore_support


# In[19]:


labels.shape


# In[ ]:





# In[20]:


Precision,recall,fscore,sup=precision_recall_fscore_support(y,labels)
PR_DBSCAN=pd.DataFrame([Precision,recall],columns=np.unique(y),index=["Precision","recall"])
PR_DBSCAN


# In[ ]:



Precision,recall,fscore,sup=precision_recall_fscore_support(y, klab)
PR_Kmeans=pd.DataFrame([Precision,recall],columns=np.unique(y),index=["Precision","recall"])
PR_Kmeans


# In[ ]:


np.shape(labels)


# ### adjusted random score

# In[ ]:


adjusted_rand_score(y,klab)


# In[ ]:




