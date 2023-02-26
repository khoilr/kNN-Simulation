# %% Import libraries
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py

from IPython.display import display

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV,
                                     RepeatedStratifiedKFold,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier

# %% Generate data
X, y = make_classification(n_samples=10000,
                           n_features=3,
                           n_informative=3,
                           n_redundant=0,
                           n_classes=5,
                           n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# To DataFrame
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
df['y'] = y
display(df)

# %% Train model
# Params
n_neighbors = np.arange(1, 21)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan']
param_grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)

# Grid search model
grid = GridSearchCV(estimator=KNeighborsClassifier(),
                    param_grid=param_grid,
                    cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3),
                    scoring='accuracy',
                    n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# Summarize results
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))

# %% Predict
y_train_pred = grid.predict(X_train)
print("Train set accuracy:", accuracy_score(y_train, y_train_pred))

y_test_pred = grid.predict(X_test)
print("Test set accuracy:", accuracy_score(y_test, y_test_pred))

# %% Visualize
# color group
color_group_pastel = {
    0: 'rgb(255, 182, 193)',
    1: 'rgb(255, 217, 102)',
    2: 'rgb(190, 237, 255)',
    3: 'rgb(170, 255, 195)',
    4: 'rgb(232, 170, 255)',
}

# 3d scatter plot on train
fig = go.Figure(data=[go.Scatter3d(x=X_train[:, 0],  # type: ignore
                                   y=X_train[:, 1],  # type: ignore
                                   z=X_train[:, 2],  # type: ignore
                                   mode='markers',
                                   marker=dict(color=[color_group_pastel[i] for i in y_train],  # type: ignore
                                               size=2,
                                               opacity=0.8),
                                   name='Train')])

# plot random point in test set
random_index = np.random.randint(0, len(X_test))
x = X_test[random_index]

# get nearest neighbors
nn = grid_result.best_estimator_.kneighbors(  # type: ignore
    [x], return_distance=False
)[0]

# plot nn of this point
fig.add_trace(go.Scatter3d(x=X_train[nn, 0],  # type: ignore
                           y=X_train[nn, 1],  # type: ignore
                           z=X_train[nn, 2],  # type: ignore
                           mode='markers',
                           marker=dict(color=[color_group_pastel[i] for i in y_train[nn]],
                                       size=2,
                                       opacity=0.8),
                           name='Nearest neighbors',))

# print prediction
print("Actual class:", y_test[random_index])
print("Predicted class:", y_test_pred[random_index])

# plot this point
fig.add_trace(go.Scatter3d(x=[x[0]],  # type: ignore
                           y=[x[1]],  # type: ignore
                           z=[x[2]],  # type: ignore
                           
                           mode='markers',
                           marker=dict(color=[color_group_pastel[y_test_pred[random_index]]],
                                       size=5,
                                       opacity=1),
                           showlegend=True,
                           name='Test point'))

# Config layout
fig.update_layout(scene=dict(xaxis=dict(nticks=4, range=[-4, 4],),
                             yaxis=dict(nticks=4, range=[-4, 4],),
                             zaxis=dict(nticks=4, range=[-4, 4],),),
                  width=1000,
                  height=800,
                  margin=dict(r=0, l=0, b=0, t=0),
                  template='plotly_dark',
                  legend=dict(x=0, y=1))

# Show
fig.show()

# %%
