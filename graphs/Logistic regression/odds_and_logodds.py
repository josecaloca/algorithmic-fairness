import pandas as pd
import numpy as np
from plotnine import *

probability = np.arange(0, 1, 0.01)
odds = probability/(1-probability) 
log_odds = np.log(odds)

df = pd.DataFrame(dict(
    probability = probability,
    odds = odds,
    log_odds = log_odds
))

ggplot(
    data = df,
    mapping = aes(x = 'probability', y = 'odds')
) + geom_point(colour = "red", size = 1.5) + \
    scale_y_continuous(limits=[0, 10],
                       breaks = [0, 1], 
                       labels = ["0", "1"]) + \
    scale_x_continuous(breaks = [0.5]) + \
    theme_classic() 


ggplot(
    data = df,
    mapping = aes(x = 'probability', y = 'log_odds')
) + geom_point(colour = "red", size = 1.5) + \
    scale_y_continuous(breaks = [0]) + \
    scale_x_continuous(breaks = [0.5]) + \
    theme_classic() + \
    geom_segment(aes(x = 0, y = 0, xend = 0.5, yend = 0)) + \
    geom_segment(aes(x = 0.5, y = 0, xend = 0.5, yend = -5))