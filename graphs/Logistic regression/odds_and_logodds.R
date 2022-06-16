library(tidyverse)
library(gridExtra)

probability = seq(0, 1, 0.01)
odds = probability/(1-probability) 
log_odds = log(odds)
df = as.data.frame(probability)

df = df %>% mutate(odds = odds,
                   log_odds = log_odds)


odds_graph <- ggplot(df, aes(x = probability, y = odds)) +
    geom_point(colour = "red", size = 1.5) +
    scale_y_continuous(limits=c(0, 10),
                       breaks = c(0, 1), 
                       labels = c("0", "1")) +
    scale_x_continuous(breaks = 0.5) +
    theme_classic() + 
    theme(axis.text.x=element_text(size=rel(1.5)),
          axis.text.y=element_text(size=rel(1.5)))


log_odds_graph <- ggplot(df, aes(x = probability, y = log_odds)) +
    geom_point(colour = "red", size = 1.5)+
    scale_y_continuous(breaks = 0) +
    scale_x_continuous(breaks = 0.5) +
    theme_classic() + 
    theme(axis.text.x=element_text(size=rel(1.5)),
          axis.text.y=element_text(size=rel(1.5)))


sc_plots = list()

sc_plots$odds_graph = odds_graph
sc_plots$log_odds_graph = log_odds_graph

grid.arrange(sc_plots$odds_graph, sc_plots$log_odds_graph, ncol = 2)
