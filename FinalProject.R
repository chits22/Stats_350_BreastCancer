
#All the other files are attempts at feature engineering
breast_cancer = read.csv('/Users/jakeatlas/Desktop/Junior/STAT350/Final Project/breastcancer.csv',header=T)
breast_cancer2 = read.csv('/Users/jakeatlas/Desktop/Junior/STAT350/Final Project/breastcancer2.csv',header=T)
breast_cancer3 = read.csv('/Users/jakeatlas/Desktop/Junior/STAT350/Final Project/breastcancer3.csv',header=T)
breast_cancer4 = read.csv('/Users/jakeatlas/Desktop/Junior/STAT350/Final Project/breastcancer4.csv',header=T)

attach(breast_cancer)
#________________________CLASSIFICATION LOGIT MODEL______________________________________________________________________________

#Fitting the model on all of the predictors doesn't work
fit_logit = glm(diagnosis~.-id, data=breast_cancer, family=binomial,control = list(maxit = 50))
predictions = predict(fit_logit, type="response")
#exp(predict(fit_logit))/(1+exp(predict(fit_logit))) --> formula for finding probabilities from predict(fit_logit)

predictors = breast_cancer[,3:(length(colnames(breast_cancer)))]
breast_cancer_with_predictions = breast_cancer
breast_cancer_with_predictions$predictions = predict(fit_logit,type="response")
breast_cancer_with_predictions$accuracy = ifelse(((breast_cancer_with_predictions$predictions == 1.000000e+00 & diagnosis=='M') | (breast_cancer_with_predictions$predictions == 0 & diagnosis=='B')),1,0)

#Model with only radius_mean as predictor
test_fit = glm(diagnosis~radius_mean, family=binomial, data=breast_cancer)
predictions_test = predict(test_fit, type= "response")
table(diagnosis,ifelse(predictions>.5,"predict: M", "predict: B")) # --> .5 is adjustable threshold


#Correlation matrix:
cor(breast_cancer[,-c(1,2)])


#________________________DECISION TREE MODEL______________________________________________________________________________
library(caret)

#Create training and test set
set.seed(1)
intrain = createDataPartition(y = diagnosis, p=.5, list=F)
train_set = breast_cancer4[intrain,]
test_set = breast_cancer4[-intrain,]

#Training decision tree classifier using information gain criterion
trctrl = trainControl(method = "repeatedcv", number=10, repeats = 3)
dtree_fit_ig = train(diagnosis~.-id, data=train_set, method="rpart", parms=list(split="information"), trControl=trctrl, tuneLength=10)
prp(dtree_fit_ig$finalModel, box.palette = "Reds", tweak = 1.2,varlen=0)

#Applying model to test set
test_pred_ig = predict(dtree_fit_ig, newdata=test_set)
confusionMatrix(test_pred_ig,test_set$diagnosis)

#Training decision tree classifier using Gini Index criterion
dtree_fit_gini = train(diagnosis~.-id, data=train_set, method="rpart", parms=list(split="gini"), trControl=trctrl, tuneLength=10)
test_pred_gini = predict(dtree_fit_gini, newdata=test_set)
prp(dtree_fit_gini$finalModel, box.palette = "Reds", tweak = 1.2,varlen=0)

confusionMatrix(test_pred_gini, test_set$diagnosis)

#Gini is better

#Using rpart - models aren't as good as Gini above
rpart_tree_info = rpart(diagnosis~.-id, data=train_set,cp=.02, parms=list(split="information"))
rpart_tree_gini = rpart(diagnosis~.-id, data=train_set,cp=.02,parms=list(split="gini"))
test_pred_info = predict(rpart_tree_info, newdata=test_set, type='class')
conf_matrix_info = table(test_pred_info, test_set$diagnosis)
rownames(conf_matrix_info) = paste("Prediction", rownames(conf_matrix_info), sep=":")
colnames(conf_matrix_info) = paste("Actual", colnames(conf_matrix_info), sep=":")
conf_matrix_info
rpart.plot(rpart_tree_info)

test_pred_gini = predict(rpart_tree_gini, newdata=test_set, type='class')
conf_matrix_gini = table(test_pred_gini, test_set$diagnosis)
rownames(conf_matrix_gini) = paste("Prediction", rownames(conf_matrix_gini), sep=":")
colnames(conf_matrix_gini) = paste("Actual", colnames(conf_matrix_gini), sep=":")
conf_matrix_gini
rpart.plot(rpart_tree_gini)
#______________________________BOOSTED TREE?________________________________________________________________________

class(diagnosis)
boosted_tree = gbm(as.factor(diagnosis)~.-id, data=train_set)
predict(boosted_tree, newdata=test_set,n.trees=100)






#______________________________LASSO?________________________________________________________________________
#library(glmnet)
# exclude_vars = names(breast_cancer) %in% c("id","diagnosis")
# predictors = as.matrix(data.frame(breast_cancer[!exclude_vars]))
# 
# grid = 10^seq(10,-2,length=100)
# lasso.mod = glmnet(predictors,y=as.factor(diagnosis),alpha=1,lambda=grid)
# set.seed(1)
# cv.out = cv.glmnet(predictors,y=as.factor(diagnosis),alpha=1)
# bestlam = cv.out$lambda.min
# out = glmnet(predictors,y=as.factor(diagnosis),alpha=1,lambda=grid)
# lass.coef = predict(out,type="coefficients",s=bestlam)[1:20,]


