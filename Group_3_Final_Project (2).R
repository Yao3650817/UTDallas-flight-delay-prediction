library(readr)
library(caret); library(gains); library(pROC)
library(rpart); library(rpart.plot)
flightDelay.df <- read_csv("~/Desktop/Sruthi/Summer2022-Sem2/BUAN6356BAwithR/Group_Project/HistoricFlight/12-2019.csv")
#--------------------------------DATA PRE-PROCESSING--------------------------------#
# Identify the Missing Values and remove all NA rows.
colSums(is.na(flightDelay.df))
flightDelay.df <- na.omit(flightDelay.df)
colSums(is.na(flightDelay.df)) #NA now 0 for every column
# To make consistent, considering only non canceled flights and scheduled_elapsed_time b/w 90-100 minutes.
library(plyr)
library(dplyr)
flightDelay.df = flightDelay.df%>%
  filter(scheduled_elapsed_time>=90 & scheduled_elapsed_time<=100 & cancelled_code=="N") 
# 65925 observations were identified with similar travel time.
# Retain necessary columns in the data set - Flight and Carrier details, Delay and Weather parameters.
# keep these columns: 1, (3,4) 6, (8,9), 11, 18, 19, 25-29, 31-35
flightDelay.df = flightDelay.df[,c(1,3,4,6,8:14,18,19,25:29,31:35)]
# Create ID column - ID for each row which acts as unique identifier for each observation.
flightDelay.df$ID=c(1:nrow(flightDelay.df))
flightDelay.df = flightDelay.df[,c("ID",setdiff(colnames(flightDelay.df),"ID"))] # Make ID the first column
# Create real arrival_delay column to know the true delay considering departure_delay.As there is high correlation b/w these 
# variables, using difference and removing individual columns from the data set.
flightDelay.df$real_arrival_delay <- flightDelay.df$arrival_delay - flightDelay.df$departure_delay
flightDelay.df = flightDelay.df[,-c(6,7)] # then remove departure_delay and arrival_delay
# Create column for outcome variable: Delay/no delay (if real arrival delay exceeds 1minute - treat delayed).
flightDelay.df$isDelayed = ifelse(flightDelay.df$real_arrival_delay>=1, 1, 0)
table(flightDelay.df$isDelayed)
# Identified 49763 flights being not delayed and 16162 flights delayed as per the data set.
# Create weekday labels for weekday column: 0=Mon, 6=Sun and remove the weekday column from the data set.
flightDelay.df$weekdayName = 
  mapvalues(flightDelay.df$weekday, from = c(0:6), 
            to = c("Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"))
# Create column for weekend variable = 1 when Saturday or Sunday or 0 otherwise.
flightDelay.df$Weekend = ifelse(flightDelay.df$weekday==5 | flightDelay.df$weekday==6, 1, 0)
flightDelay.df <- flightDelay.df[,-12] #remove weekday column.
#
##Partition the data into Training and Validation Sets in 60:40 ratio.
set.seed(123)
train.index <- sample(c(1:dim(flightDelay.df)[1]), dim(flightDelay.df)[1]*0.6)  
train.df <- flightDelay.df[train.index, ]
valid.df <- flightDelay.df[-train.index, ]
##Understand Data
install.packages(c("GGally", "ggmap", "mosaic", "treemap"))
## Heatmap with values
library(gplots)
my_data <- flightDelay.df[,c(5:10,12:23,25)] #Consider only numeric variables
heatmap.2(cor(my_data), Rowv = FALSE, Colv = FALSE, dendrogram = "none", 
          cellnote = round(cor(my_data),2), 
          notecol = "black", key = FALSE, trace = 'none', margins = c(10,10))
#Barplot to visualize Average Flight Delays by Carrier and Weekday.
barplot(aggregate(flightDelay.df$isDelayed == 1, by = list(flightDelay.df$weekdayName),
                  mean, rm.na = T)[,2], xlab = "Day of Week", ylab = "Average Delay",
        names.arg = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))

barplot(aggregate(flightDelay.df$isDelayed == 1, by = list(flightDelay.df$carrier_code),
                  mean, rm.na = T)[,2], xlab = "Carrier_Code", ylab = "Average Delay",
        names.arg = c("AA","AS","B6","DL","F9","G4","NK","UA","WN"))
#--------------------------------DATA MODELLING--------------------------------#
#------METHOD-1 LOGISTIC REGRESSION METHOD------#
#Build the Model using statistically significant predictors.Considered predictors based on correlation and domain knowledge.
#Step:1 Obtain Estimates of the probabilities of belonging to each class.
logit.reg <- glm(isDelayed ~ delay_carrier +delay_late_aircarft_arrival +delay_national_aviation_system
                 +Weekend  +HourlyDryBulbTemperature_x + HourlyPrecipitation_x 
                 +HourlyStationPressure_x +HourlyVisibility_x
                + HourlyPrecipitation_y 
                  +HourlyVisibility_y + HourlyWindSpeed_y
                 +carrier_code , data = train.df, family = "binomial")
data.frame(summary(logit.reg)$coefficients, odds = exp(coef(logit.reg)))
#summary(logit.reg)
#
#Step:2 Determine Cut-off
pred_ <- as.factor(ifelse(predict(logit.reg, valid.df, type="response")>0.36,"1","0"))
confusionMatrix(pred_, as.factor(valid.df$isDelayed))
pred_ <- as.factor(ifelse(predict(logit.reg, train.df, type="response")>0.36,"1","0"))
confusionMatrix(pred_, as.factor(train.df$isDelayed))
#Based on trial-error, obtained cutoff at 0.36 with maximum classification accuracy and maximum sensitivity subject to 
#minimum specificity.
#Plot ROC Curve
library(pROC)
test_prob = predict(logit.reg, newdata = valid.df, type = "response")
test_roc = roc(valid.df$isDelayed ~ test_prob, plot = TRUE, print.auc = TRUE)

#
#------METHOD-2 CLASSIFICATION TREE------#
default.ct <- rpart(isDelayed ~ delay_carrier +delay_late_aircarft_arrival +delay_national_aviation_system
                    +Weekend  +HourlyDryBulbTemperature_x + HourlyPrecipitation_x 
                    +HourlyStationPressure_x +HourlyVisibility_x
                    + HourlyPrecipitation_y 
                    +HourlyVisibility_y + HourlyWindSpeed_y
                    +carrier_code, 
                    data = train.df, cp=0,xval=5,maxdepth=7, method = "class")
length(default.ct$frame$var[default.ct$frame$var == "<leaf>"])
#Terminal Nodes-25 ; Decision Nodes-24.
# Plot tree
rpart.plot(default.ct)
prp(default.ct, type = 1, extra = 1, split.font = 1, varlen = -10)  
#Determine Accuracy for the default tree
predict_train<-predict(default.ct,train.df,type='class')
confusionMatrix(predict_train, as.factor(train.df$isDelayed)) 
predict_valid<-predict(default.ct,valid.df,type='class')
confusionMatrix(predict_valid, as.factor(valid.df$isDelayed)) 
#To improve the performance and avoid any overfitting the data, pruning the tree
printcp(default.ct)
pruned.ct <- prune(default.ct, cp = 9.3400e-04 )
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])
# plot pruned tree
rpart.plot(pruned.ct)
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10)  
#Determine Accuracy for the pruned tree
predict_train<-predict(pruned.ct,train.df,type='class')
confusionMatrix(predict_train, as.factor(train.df$isDelayed)) 
predict_valid<-predict(pruned.ct,valid.df,type='class')
confusionMatrix(predict_valid, as.factor(valid.df$isDelayed)) 
#Plot roc
install.packages("ROCR")
library(ROCR)
#When increasing cutoff values, accuracy increases, sensitivity increases and specificity decreases
pred <- prediction(predict(default.ct, type = "prob")[, 2], train.df$isDelayed)
plot(performance(pred, "tpr", "fpr"))
abline(0, 1, lty = 2)
plot(performance(pred, "acc"))







