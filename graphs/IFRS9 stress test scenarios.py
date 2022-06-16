import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

np.random.seed(0)
s=np.random.choice(5)  

num_rows = 15
years = list(range(2022, 2022 + num_rows))

base = np.random.random(num_rows).cumsum()
adverse = base.copy()
adverse[10:] += 1.506

good = base.copy()
good[10:] += - 2.18


data_preproc = pd.DataFrame({
    'Year': years, 
    'Base': base,
    'Adverse': adverse,
    'Good': good})


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
p = sns.lineplot(x='Year', y='value', hue='variable', 
             data=pd.melt(data_preproc, ['Year']))

p.set_xlabel("Years")
p.set_ylabel("Probability of Default")
plt.show()

