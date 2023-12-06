library(readxl)
library(tidyverse)
library(skimr)
library(psych)
library(ggplot2)
library(caret)
library(glmnet)
library(randomForest)
library(xgboost)
library(e1071)
library(ROCR)
library(dbscan)
library(factoextra)
library(dplyr)


ccdata <- read_excel("amlcc.xls")
#Cleaning
ccdata <- data.frame(ccdata)
ccdata <- ccdata[-1,]
ccdata <- ccdata[-c(1)]

ccdata <- subset(ccdata, X3 %in% c(1, 2, 3, 4))
catlist <- c("X2","X3","X4","X6","X7","X8","X9","X10","X11","Y")
ccdata <- ccdata %>%
  mutate_at(catlist, list(~factor(.)))


numlist <- !sapply(ccdata, is.factor)
ccdata[numlist] <- lapply(ccdata[numlist], as.numeric)

skim(ccdata)
sum(is.na(ccdata))

#Summaries
#Age
summary_x5 <- ccdata %>%
  group_by(Y) %>%               
  summarize(
    mean_age = mean(X5),            
    median_age = median(X5),         
    sd_age = sd(X5)                  
  )

print(summary_x5)
#Education
summary_x3 <- ccdata %>%
  group_by(X3) %>%             
  summarize(
    default_rate = mean(Y == 1),  
    n = n()                                
  )

print(summary_x3)

#Credit Amount
summary_x1 <- ccdata %>%
  group_by(Y) %>%            
  summarize(
    avg_X1 = mean(X1)        
  )

print(summary_x1)



#EDA
#describe data
describe(ccdata)


#Number of Credit loans by age grouped by default or not
ggplot(ccdata, aes(x = X5, fill = Y)) +
  geom_histogram(binwidth = 1, position = "dodge") +
  labs(x = "Age", y = "Count of Loans") +
  scale_fill_manual(values = c("red", "blue")) +
  theme(panel.background = element_rect(fill = "black"),
        panel.border = element_rect(color = "black", fill = NA),
        plot.background = element_rect(fill = "black"),
        axis.text = element_text(color = "white"),
        axis.title = element_text(color = "white"),
        plot.title = element_text(color = "white"))

#Credit avaliability Distribution by default or not
ggplot(ccdata, aes(x = Y, y = X1)) +
  geom_boxplot(outlier.colour = "red") +
  labs(x = "Defualt", y = "Credit Amount", title = "Credit Avaliability Distribution") +
  theme(
    panel.background = element_rect(fill = "black"),
    plot.background = element_rect(fill = "black"),
    axis.text = element_text(color = "white"),
    axis.title = element_text(color = "white"),
    plot.title = element_text(color = "white"),
    panel.border = element_rect(color = "black", fill = NA)
  )
#Education vs Credit Amount
ggplot(ccdata, aes(x = X3, y= X1, fill = Y)) +
  scale_fill_manual(values=c("green", "red")) +
  geom_boxplot() +
  labs(x = "Education", y = "Credit Amount", title = "Credit Avaliability Distribution")


#Pre-Processing

#Refactor levels
ccdata$Y<-fct_recode(ccdata$Y, not_default = "0", default = "1")
ccdata$X3<-fct_recode(ccdata$X3, grad = "1", uni = "2", hs = "3", other = "4")
ccdata$X4<-fct_recode(ccdata$X4, married = "1", single = "2", other = "3")
ccdata$X2<-fct_recode(ccdata$X2, male = "1", female = "2")


#Create dummys for each model
dummies_modelx3 <- dummyVars(~ ., data = ccdata[, names(ccdata) != "X3"])


predictors_dummyx3<- data.frame(predict(dummies_modelx3, newdata = ccdata))

ccdata_x3 <- cbind(X3=ccdata$X3, predictors_dummyx3)


#create training data and testing data


index3 <- createDataPartition(ccdata_x3$X3, p = .8,list = FALSE)
x3train_data <- ccdata_x3[index3,]
x3test_data <- ccdata_x3[-index3,]


#modeling source: Data Analytics 2, Brittany Green

ctrl <- trainControl(method = "cv", number = 5)

#Creating classification models for education
classctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = defaultSummary, verboseIter = TRUE)


x3xgb <- train(
  X3 ~ .,
  data = x3train_data,
  method = "xgbTree",  
  trControl = classctrl,
  tuneLength = 5
)

x3xgbpred <- predict(x3xgb, newdata = x3test_data)
x3xgbcm<-confusionMatrix(x3xgbpred,x3test_data$X3)
print(x3xgbcm)


# Unsurpervised method
#clustering
ccdata1 <- as.data.frame(lapply(ccdata, as.numeric))




ccdatamatrix <- as.matrix(ccdata1[,c(1,2,4:11,24)])
db <- dbscan(ccdatamatrix, eps=0.5, MinPts=2)
db

df <- ccdata1[,c(1,2,4:11,24)]


fviz_nbclust(df, kmeans, method = "wss")
fviz_nbclust(df, kmeans, method = "silhouette")
final <- kmeans(df, 3, nstart=27)
fviz_cluster(final, data=df)

avg1 <- df %>%
  mutate(Cluster = final$cluster) %>%
  group_by(Cluster) %>%
  summarize_all("mean")
df$cluster <- final$cluster
df$X3 <- ccdata1$X3
print(avg1)

#modeling
df <- df %>%
  mutate_at(catlist, list(~factor(.)))

df$Y<-fct_recode(df$Y, not_default = "0", default = "1")
df$X3<-fct_recode(df$X3, grad = "1", uni = "2", hs = "3", other = "4")
df$X4<-fct_recode(df$X4, married = "1", single = "2", other = "3")
df$X2<-fct_recode(df$X2, male = "1", female = "2")


#Create dummys for each model
dfdummies_modelx3 <- dummyVars(~ ., data = df[, names(df) != "X3"])


dfpredictors_dummyx3<- data.frame(predict(dfdummies_modelx3, newdata = df))

df_x3 <- cbind(X3=df$X3, dfpredictors_dummyx3)


#create training data and testing data


dfindex3 <- createDataPartition(df_x3$X3, p = .8,list = FALSE)
dfx3train_data <- df_x3[dfindex3,]
dfx3test_data <- df_x3[-dfindex3,]

#training
x3xgbclust <- train(
  X3 ~ .,
  data = dfx3train_data,
  method = "xgbTree",  
  trControl = classctrl,
  tuneLength = 5
)

#Testing
x3xgbpredclust <- predict(x3xgbclust, newdata = dfx3test_data)
x3xgbcmclust<-confusionMatrix(x3xgbpredclust,dfx3test_data$X3)
print(x3xgbcmclust)
