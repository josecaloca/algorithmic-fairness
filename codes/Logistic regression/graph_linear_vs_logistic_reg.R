library(tidyverse)

set.seed(123)

x <- rnorm(100)
y <- ifelse(x >0 , 1, 0)

outliers_0 <- ifelse(sample(y, 20) == 0, 1,1)
outliers_1 <- ifelse(sample(y, 20) == 1, 0,0)

x <- c(x,
  rnorm(length(outliers_0)),
  rnorm(length(outliers_1))
  )

y <- c(y,
       outliers_0,
       outliers_1)

plot(x,y)

df <- data.frame(x,y)

###########################################
# linear vs logisitc regression 
###########################################

linear_reg <- ggplot(df, aes(x=x, y=y)) + 
    geom_point(colour = "blue") + 
    stat_smooth(method="glm", se=FALSE,colour = "red")+
    scale_y_continuous(limits=c(-0.2, 1.2),
                       breaks = c(0, 1), 
                       labels = c("0", "1")) +
    theme_classic()

logistic_reg <- ggplot(df, aes(x=x, y=y)) + 
    geom_point(colour = "blue") + 
    stat_smooth(method="glm", method.args = list(family = "binomial"), se=FALSE,colour = "red")+
    scale_y_continuous(limits=c(-0.2, 1.2),
                       breaks = c(0, 1), 
                       labels = c("0", "1")) +
    theme_classic()


sc_plots = list()

sc_plots$linear_reg = linear_reg
sc_plots$logistic_reg = logistic_reg

grid.arrange(sc_plots$linear_reg, sc_plots$logistic_reg, ncol = 2)

###########################################
# logistic regression prediction
###########################################

ggplot(df, aes(x=x, y=y)) + 
    geom_point(colour = "blue") + 
    stat_smooth(method="glm", se=FALSE,colour = "red", size = 1.1)+
    scale_y_continuous(limits=c(-0.2, 1.2),
                       breaks = c(0, 1), 
                       labels = c("Negative Infinity", "Positive Infinity"))  + 
    labs(x = "X variable",
         y = "Log (Odds)") +
    theme_classic()
