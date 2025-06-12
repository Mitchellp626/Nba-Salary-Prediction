# Capstone Project
library(dplyr)
library(ggplot2)
library(scales)
library(caTools)
library(glmnet)
library(torch)
library(luz) 
library(torchvision) 
library(zeallot)
library(randomForest)
library(e1071)
library(ISLR)
library(caret)
library(usethis)

## Load in csv files
Salaries <- read.csv('Salaries.csv')
Player_Stats <- read.csv('Player_Stats.csv')

#Set row names
rownames(Salaries) <- Salaries$Player
rownames(Player_Stats) <- Player_Stats$Player

#Confirm no duplicates
sum(duplicated(Salaries$player.id))
sum(duplicated(Player_Stats$player.id))

#Removing what I don't want
Salaries_Clean <- Salaries %>% select(2,3,10) %>% filter(X2024.25 > 1000000)
Player_Stats_Clean <- Player_Stats %>% select(-1,-6,-7,-8,-24,-31,-32)

sum(duplicated(Player_Stats_Clean))

colSums(is.na(Salaries_Clean))
colSums(is.na(Player_Stats_Clean))
summary(Salaries_Clean)

#Setting NA Values to zero
Player_Stats_Clean[is.na(Player_Stats_Clean)] = 0
colSums(is.na(Player_Stats_Clean))

Player_Stats_Clean$Pos <- as.factor(Player_Stats_Clean$Pos)

Stats_Salaries_MERGED <- merge(Salaries_Clean,Player_Stats_Clean , by ="player.id")
rownames(Stats_Salaries_MERGED) <- Stats_Salaries_MERGED$Player

Stats_Salaries_MERGED_CLEAN <- Stats_Salaries_MERGED %>% select(-6)


## Just exploring the data, was curious to see what teams pay the highest salaries.
ggplot(data = Stats_Salaries_MERGED_CLEAN, aes(x = Team.x, y = X2024.25, col = Team.x, size = X2024.25)) + 
  geom_point() +   
  scale_y_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
  ggtitle("Salaries of Players Based on Team (in millions)") +
  labs(x = "Team", y = "Player Salary")


FinalData <- Stats_Salaries_MERGED_CLEAN %>% select(-1,-2,-4)

set.seed(234)
Trainsamplesize <- floor(0.8 * nrow(FinalData))

train_ind <- sample(seq_len(nrow(FinalData)), size = Trainsamplesize)
train_2025 <- FinalData[train_ind, ]
test_2025 <- FinalData[-train_ind, ]

lm.salary <- lm(X2024.25 ~ ., data = train_2025)

lm.salary.predvactual <- data.frame(pred = round(predict(lm.salary, test_2025)), actual = test_2025$X2024.25)

lm.salary.mse <- mean((lm.salary.predvactual$actual - lm.salary.predvactual$pred)^2)
lm.salary.rmse <- sqrt(mean((lm.salary.predvactual$actual - lm.salary.predvactual$pred)^2))

coef(lm.salary)

summary(lm.salary)

ggplot(lm.salary.predvactual, aes(x = pred, y = actual)) +
  geom_point() +
  scale_y_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
  scale_x_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
  stat_smooth(method = "lm", se=FALSE, colour="blue") +
  ggtitle("Predicted vs Actual Salaries for Linear Model (in millions)") +
  xlab("Predicted Salary") +
  ylab("Actual Salary")

####################### lasso

x_train <- as.matrix(train_2025 %>% select(-X2024.25,-Pos)) 
y_train <- train_2025$X2024.25
x_test <- as.matrix(test_2025 %>% select(-X2024.25,-Pos))
y_test <- test_2025$X2024.25

lasso.mod <- cv.glmnet(x_train, y_train, alpha = 1)
cv.out.lasso = cv.glmnet(x_train, y_train, alpha = 1) 
bestlam.lasso = cv.out.lasso$lambda.min
lasso.pred = round(predict(lasso.mod, s = bestlam.lasso, newx = x_test))
MSE_bestLasso <- mean((lasso.pred - y_test)^2)
RMSE_bestLasso <- sqrt(mean((lasso.pred - y_test)^2))
lasso_r2 <- cor(test_2025$X2024.25, lasso.pred)^2

plot(lasso.mod)
coef(lasso.mod, s = lasso.mod$lambda.min)
lasso.salary.predvactual <- data.frame(pred = round(lasso.pred), actual = test_2025$X2024.25)


##################### Random Forest

set.seed(123)

forest_model <- randomForest(X2024.25 ~ ., data = train_2025, ntree = 500, mtry = 3, importance = TRUE)
print(forest_model)

forest_predictions <- predict(forest_model, newdata = test_2025)
forest_mse <- mean((test_2025$X2024.25 - forest_predictions)^2)
forest_rmse <- sqrt(mean((test_2025$X2024.25 - forest_predictions)^2))
forest_r2 <- cor(test_2025$X2024.25, forest_predictions)^2

varImpPlot(forest_model, main = "VarImpPlot for Random Forest Model")

################### SVM

Svm_model <- svm(X2024.25 ~ ., data = train_2025, 
                 kernel = "radial", 
                 cost = 1, 
                 gamma = 0.1)
svm_predictions <- predict(Svm_model, test_2025)
mse_svm <- mean((svm_predictions - test_2025$X2024.25)^2)
rmse_svm <- sqrt(mean((svm_predictions - test_2025$X2024.25)^2))
r2_svm<- cor(test_2025$X2024.25, svm_predictions)^2
svm.salary.predvactual <- data.frame(pred = svm_predictions, actual = test_2025$X2024.25)

tuned_svm <- tune(svm, X2024.25 ~ ., 
                  data = train_2025,
                  kernel = "radial",
                  ranges = list(cost = c(0.1, 1, 10, 100),
                                gamma = c(0.01, 0.1, 1, 10)))
best_tuned_model <- tuned_svm$best.model
tuned_predictions <- predict(best_tuned_model, test_2025)
tuned_rmse_svm <- sqrt(mean((tuned_predictions - test_2025$X2024.25)^2))
tuned.svm.salary.predvactual <- data.frame(pred = round(tuned_predictions), actual = test_2025$X2024.25)

ggplot(tuned.svm.salary.predvactual, aes(x = pred, y = actual)) +
  geom_point() +
  scale_y_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
  scale_x_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
  stat_smooth(method = "svm", se=FALSE, colour="blue") +
  ggtitle("Predicted vs Actual Salaries for SVM Model (in millions)") +
  xlab("Predicted Salary") +
  ylab("Actual Salary")

ggplot(svm.salary.predvactual, aes(x = pred, y = actual)) +
  geom_point() +
  scale_y_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
  scale_x_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
  stat_smooth(method = "svm", se=FALSE, colour="blue")