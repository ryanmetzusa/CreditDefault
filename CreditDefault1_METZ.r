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
library(performanceEstimation)


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

#class imbalance
table(ccdata$Y)

#Number of Credit loans by age grouped by default or not
ggplot(ccdata, aes(x = X5, fill = Y)) +
  geom_histogram(binwidth = 1, position = "dodge") +
  labs(x = "Age", y = "Count of Loans", title = "Number of accounts by Age Distribution") +
  scale_fill_manual(values = c("red", "blue")) +
  theme(panel.background = element_rect(fill = "white"),
        panel.border = element_rect(color = "white", fill = NA),
        plot.background = element_rect(fill = "white"),
        axis.text = element_text(color = "black"),
        axis.title = element_text(color = "black"),
        plot.title = element_text(color = "black"))

#Credit avaliability Distribution by default or not
ggplot(ccdata, aes(x = Y, y = X1)) +
  geom_boxplot(outlier.colour = "red") +
  labs(x = "Defualt", y = "Credit Amount", title = "Credit Avaliability Distribution") +
  theme(
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black"),
    plot.title = element_text(color = "black"),
    panel.border = element_rect(color = "white", fill = NA)
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

ccdata_smote <- smote(Y~., ccdata, perc.over=7, perc.under = 1)
table(ccdata_smote$Y)


#pre-smote
#Create dummys for each model
dummies_model_pre <- dummyVars(~ ., data = ccdata[, names(ccdata) != "Y"])


predictors_dummy_pre<- data.frame(predict(dummies_model_pre, newdata = ccdata))

ccdata_pre <- cbind(Y=ccdata$Y, predictors_dummy_pre)


#create training data and testing data


index_pre <- createDataPartition(ccdata_pre$Y, p = .8,list = FALSE)
train_data_pre <- ccdata_pre[index_pre,]
test_data_pre <- ccdata_pre[-index_pre,]


#modeling source: Data Analytics 2, Brittany Green

ctrl <- trainControl(method = "cv", number = 5)

#Creating classification models for education
classctrl <- trainControl(method = "cv", number = 3, classProbs = TRUE, summaryFunction = defaultSummary, verboseIter = TRUE)


xgb_pre <- train(
  Y ~ .,
  data = train_data_pre,
  method = "xgbTree",  
  trControl = classctrl,
  tuneLength = 5
)

xgbpred_pre <- predict(xgb_pre, newdata = test_data_pre)
xgbcm_pre<-confusionMatrix(xgbpred_pre,test_data_pre$Y)
print(xgbcm_pre)




#Create dummys for each model
dummies_model <- dummyVars(~ ., data = ccdata_smote[, names(ccdata_smote) != "Y"])


predictors_dummy<- data.frame(predict(dummies_model, newdata = ccdata_smote))

ccdata_post <- cbind(Y=ccdata_smote$Y, predictors_dummy)


#create training data and testing data


index <- createDataPartition(ccdata_post$Y, p = .8,list = FALSE)
train_data <- ccdata_post[index,]
test_data <- ccdata_post[-index,]


#creating smote model

xgb <- train(
  Y ~ .,
  data = train_data,
  method = "xgbTree",  
  trControl = classctrl,
  tuneLength = 5
)

xgbpred <- predict(xgb, newdata = test_data)
xgbcm<-confusionMatrix(xgbpred,test_data$Y)
print(xgbcm)


# Unsurpervised method
#clustering
ccdata_clust <- as.data.frame(lapply(ccdata_smote, as.numeric))




ccdatamatrix <- as.matrix(ccdata_clust)
db <- dbscan(ccdatamatrix, eps=0.5, MinPts=2)
db

df <- ccdata_clust


#fviz_nbclust(df, kmeans, method = "wss")
#fviz_nbclust(df, kmeans, method = "silhouette")
final <- kmeans(df, 3, nstart=27)
fviz_cluster(final, data=df)

avg1 <- df %>%
  mutate(Cluster = final$cluster) %>%
  group_by(Cluster) %>%
  summarize_all("mean")
df$cluster <- final$cluster
df$Y <- ccdata_clust$Y
print(avg1)


#modeling
df <- df %>%
  mutate_at(catlist, list(~factor(.)))

df$Y<-fct_recode(df$Y, not_default = "1", default = "2")
df$X3<-fct_recode(df$X3, grad = "1", uni = "2", hs = "3", other = "4")
df$X4<-fct_recode(df$X4, married = "1", single = "2", other = "3")
df$X2<-fct_recode(df$X2, male = "1", female = "2")


#Create dummys for each model
dfdummies_model_clust <- dummyVars(~ ., data = df[, names(df) != "Y"])


dfpredictors_dummy_clust<- data.frame(predict(dfdummies_model_clust, newdata = df))

df_clust <- cbind(Y=df$Y, dfpredictors_dummy_clust)


#create training data and testing data


dfindex_clust <- createDataPartition(df_clust$Y, p = .8,list = FALSE)
dftrain_data_clust <- df_clust[dfindex_clust,]
dftest_data_clust <- df_clust[-dfindex_clust,]

#training
xgbclust <- train(
  Y ~ .,
  data = dftrain_data_clust,
  method = "xgbTree",  
  trControl = classctrl,
  tuneLength = 5
)

#Testing
xgbpredclust <- predict(xgbclust, newdata = dftest_data_clust)
xgbcmclust<-confusionMatrix(xgbpredclust,dftest_data_clust$Y)
print(xgbcmclust)

varImp(xgbclust)
