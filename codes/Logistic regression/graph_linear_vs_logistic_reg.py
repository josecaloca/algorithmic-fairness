import pandas as pd
import numpy as np
from random import sample
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

n = 1000

x = np.random.normal(0,1,n)

y = [1 if i > 0 else 0 for i in x ]

outliers_0 = [1 if i == 0 else 1 for i in sample(y, int(n*0.05))]

outliers_1 = [0 if i == 1 else 0 for i in sample(y, int(n*0.05))]

x = np.append(x, np.random.normal(0,1,2*len(outliers_0)))
y = y + outliers_0 + outliers_1

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(x.reshape(-1,1),y)

# predictions
preds=logreg.predict_proba(x.reshape(-1,1))

df = pd.DataFrame(dict(y = y, x = x, preds = preds[:,1]))

df.head(10)
###########################################
# linear vs logisitc regression 
###########################################

ggplot(
    data = df,
    mapping = aes(x = 'x', y = 'y')
) + geom_point(colour = "blue") + \
    stat_smooth(method="glm", se=False,colour = "red", size = 0.5) + \
    scale_y_continuous(limits=[-0.2, 1.2],
                       breaks = [0, 1], 
                       labels = ["0", "1"]) + \
    theme_classic() + \
    labs(
        x = "X",
        y = "Y"
    )

    
ggplot(
    data = df,
    mapping = aes(x = 'x', y = 'y')
) + geom_point(colour = "blue") + \
    geom_point(aes(y = 'preds'), colour = 'red', size = 0.5)  + \
    scale_y_continuous(limits=[-0.2, 1.2],
                       breaks = [0, 1], 
                       labels = ["0", "1"]) + \
    theme_classic() + \
    labs(
        x = "X",
        y = "Y"
    )
    
