library(tree)
library(Metrics)
library (randomForest)
library(glmnet)
library(gbm)
library(yardstick)
library(readxl)

dat <- read.table("auto-mpg.data")
head(dat)

#preprocessing
dat <- dat[,-c(8,9)]
dat[,4] <- as.numeric(dat[,4])
dat <- na.omit(dat)
head(dat)

# feature selection
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train_scaled = dat[-testrows,]
test_scaled = dat[testrows,]

treemod <- tree (V1 ~ ., train_scaled)
train.pred = predict(treemod,train_scaled)
test.pred = predict(treemod,test_scaled)

frame = treemod$frame
leaves = frame$var == "<leaf>"
used = unique(frame$var[!leaves])
features.select = names(dat)[(names(dat) %in% used)]
features.select

#test-train split
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]

#new data after feature selection
train <- train[,c(1,4,3,7)]
test <- test[,c(1,4,3,7)]

# dt
tree.dat <- tree (V1 ~ ., train)
ypred <- predict (tree.dat , newdata = test)
rss <- sum((test$V1-ypred)^2)
tss <- sum((test$V1-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V1,ypred)
rmse(test$V1,ypred)
mape(test$V1,ypred)
r2
adr2

# ridge
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c(1,4,3,7)]
test <- test[,c(1,4,3,7)]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 0, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$V1-ypred)^2)
tss <- sum((test$V1-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V1,ypred)
rmse(test$V1,ypred)
mape(test$V1,ypred)
r2
adr2

# lasso
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c(1,4,3,7)]
test <- test[,c(1,4,3,7)]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 1, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 4, newx = as.matrix(test[,-1]))
rss <- sum((test$V1-ypred)^2)
tss <- sum((test$V1-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V1,ypred)
rmse(test$V1,ypred)
mape(test$V1,ypred)
r2
adr2

# rf
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c(1,4,3,7)]
test <- test[,c(1,4,3,7)]
rf.dat <- randomForest(V1 ~ .,train, mtry=40)
ypred <- predict (rf.dat, newdata = test)
rss <- sum((test$V1-ypred)^2)
tss <- sum((test$V1-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V1,ypred)
rmse(test$V1,ypred)
mape(test$V1,ypred)
r2
adr2

# boost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c(1,4,3,7)]
test <- test[,c(1,4,3,7)]
boost.dat <- gbm (V1 ~ ., data = train, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
ypred <-  predict (boost.dat , newdata = test, n.trees = 5000)
rss <- sum((test$V1-ypred)^2)
tss <- sum((test$V1-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V1,ypred)
rmse(test$V1,ypred)
mape(test$V1,ypred)
r2
adr2

# svr 
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c(1,4,3,7)]
test <- test[,c(1,4,3,7)]
dtn <- DecisionTreeNode$new(x=train[,-1],y=train[,1],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn$split_node()
ypred <- as.numeric(dtn$predict(test[,-1]))
rss <- sum((test$V1-ypred)^2)
tss <- sum((test$V1-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V1,ypred)
rmse(test$V1,ypred)
mape(test$V1,ypred)
r2
adr2

# svrf
set.seed(5)
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c(1,4,3,7)]
test <- test[,c(1,4,3,7)]
ypred <- rf(60,train[,-1],train[,1],test[,-1])
rss <- sum((test$V1-ypred)^2)
tss <- sum((test$V1-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V1,ypred)
rmse(test$V1,ypred)
mape(test$V1,ypred)
r2
adr2

# svrboost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- boost(train[,-1], train[,1], test[,-1], 40)
rss <- sum((test$V1-ypred)^2)
tss <- sum((test$V1-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V1,ypred)
rmse(test$V1,ypred)
mape(test$V1,ypred)
r2
adr2



dat <- read.table("communities.data",sep=",")
head(dat)

#preprocessing
dat <- dat[,-c(2,3,4,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,122,123,124,125,127)]
head(dat)
dim(dat)

# feature selection
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train_scaled = dat[-testrows,]
test_scaled = dat[testrows,]

treemod <- tree (V128 ~ ., train_scaled)
train.pred = predict(treemod,train_scaled)
test.pred = predict(treemod,test_scaled)

frame = treemod$frame
leaves = frame$var == "<leaf>"
used = unique(frame$var[!leaves])
features.select = names(dat)[(names(dat) %in% used)]
features.select

#test-train split
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]

#new data after feature selection
set.seed(890)
train <- train[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
test <- test[,c("V128","V9" ,"V33","V44","V50","V56","V96")]

# dt
tree.dat <- tree (V128 ~ ., train)
ypred <- predict (tree.dat , newdata = test)
rss <- sum((test$V128-ypred)^2)
tss <- sum((test$V128-mean(test$V128))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V128,ypred)
rmse(test$V128,ypred)
mape(test$V128,ypred)
r2
adr2

# ridge
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
test <- test[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 0, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$V128-ypred)^2)
tss <- sum((test$V128-mean(test$V128))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V128,ypred)
rmse(test$V128,ypred)
mape(test$V128,ypred)
r2
adr2

# lasso
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
test <- test[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 1, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$V128-ypred)^2)
tss <- sum((test$V128-mean(test$V128))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V128,ypred)
rmse(test$V128,ypred)
mape(test$V128,ypred)
r2
adr2

# rf
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
test <- test[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
rf.dat <- randomForest(V128 ~ .,train, mtry=40)
ypred <- predict (rf.dat, newdata = test)
rss <- sum((test$V128-ypred)^2)
tss <- sum((test$V128-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V128,ypred)
rmse(test$V128,ypred)
mape(test$V128,ypred)
r2
adr2

# boost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
test <- test[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
boost.dat <- gbm (V128 ~ ., data = train, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
ypred <-  predict (boost.dat , newdata = test, n.trees = 50)
rss <- sum((test$V128-ypred)^2)
tss <- sum((test$V128-mean(test$V1))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V128,ypred)
rmse(test$V128,ypred)
mape(test$V128,ypred)
r2
adr2

# svr 
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c("V128", "V9" ,"V33","V44","V50","V56","V96","V40","V46","V49","V55","V74","V98")]
test <- test[,c("V128", "V9" ,"V33","V44","V50","V56","V96","V40","V46","V49","V55","V74","V98")]
dtn <- DecisionTreeNode$new(x=train[,-1],y=train[,1],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn$split_node()
ypred <- as.numeric(dtn$predict(test[,-1]))
rss <- sum((test$V128-ypred)^2)
tss <- sum((test$V128-mean(test$V128))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V128,ypred)
rmse(test$V128,ypred)
mape(test$V128,ypred)
r2
adr2

# svrf
set.seed(5)
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
test <- test[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
ypred <- rf(40,train[,-1],train[,1],test[,-1])
rss <- sum((test$V128-ypred)^2)
tss <- sum((test$V128-mean(test$V128))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V128,ypred)
rmse(test$V128,ypred)
mape(test$V128,ypred)
r2
adr2

# svrboost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
train <- train[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
test <- test[,c("V128", "V9" ,"V33","V44","V50","V56","V96")]
ypred <- boost(train[,-1], train[,1], test[,-1], 30)
rss <- sum((test$V128-ypred)^2)
tss <- sum((test$V128-mean(test$V128))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V128,ypred)
rmse(test$V128,ypred)
mape(test$V128,ypred)
r2
adr2

dat <- as.data.frame(read_xlsx("Folds5x2_pp.xlsx"))
head(dat)
dim(dat)

testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train_scaled = dat[-testrows,]
test_scaled = dat[testrows,]

treemod <- tree (PE ~ ., train_scaled)
train.pred = predict(treemod,train_scaled)
test.pred = predict(treemod,test_scaled)

frame = treemod$frame
leaves = frame$var == "<leaf>"
used = unique(frame$var[!leaves])
features.select = names(dat)[(names(dat) %in% used)]
features.select

#test-train split
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
tree.dat <- tree (PE ~ ., train)
ypred <- predict (tree.dat , newdata = test)
rss <- sum((test$PE-ypred)^2)
tss <- sum((test$PE-mean(test$PE))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$PE,ypred)
rmse(test$PE,ypred)
mape(test$PE,ypred)
r2
adr2

# ridge
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-5], train[,5], alpha = 0, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-5]))
rss <- sum((test$PE-ypred)^2)
tss <- sum((test$PE-mean(test$PE))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$PE,ypred)
rmse(test$PE,ypred)
mape(test$PE,ypred)
r2
adr2

# lasso
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-5], train[,5], alpha = 1, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-5]))
rss <- sum((test$PE-ypred)^2)
tss <- sum((test$PE-mean(test$PE))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$PE,ypred)
rmse(test$PE,ypred)
mape(test$PE,ypred)
r2
adr2

# rf
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
rf.dat <- randomForest(PE ~ .,train, mtry=40)
ypred <- predict (rf.dat, newdata = test)
rss <- sum((test$PE-ypred)^2)
tss <- sum((test$PE-mean(test$PE))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$PE,ypred)
rmse(test$PE,ypred)
mape(test$PE,ypred)
r2
adr2

# boost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
boost.dat <- gbm (PE ~ ., data = train, distribution = "gaussian", n.trees = 50, interaction.depth = 4)
ypred <-  predict (boost.dat , newdata = test, n.trees = 20)
rss <- sum((test$PE-ypred)^2)
tss <- sum((test$PE-mean(test$PE))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$PE,ypred)
rmse(test$PE,ypred)
mape(test$PE,ypred)
r2
adr2

# svr 
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
dtn <- DecisionTreeNode$new(x=train[,-5],y=train[,5],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn$split_node()
ypred <- as.numeric(dtn$predict(test[,-5]))
rss <- sum((test$PE-ypred)^2)
tss <- sum((test$PE-mean(test$PE))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$PE,ypred)
rmse(test$PE,ypred)
mape(test$PE,ypred)
r2
adr2

# svrf
set.seed(5)
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- rf(20,train[,-5],train[,5],test[,-5])
rss <- sum((test$PE-ypred)^2)
tss <- sum((test$PE-mean(test$PE))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$PE,ypred)
rmse(test$PE,ypred)
mape(test$PE,ypred)
r2
adr2

# svrboost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- boost(train[,-5], train[,5], test[,-5], 30)
rss <- sum((test$PE-ypred)^2)
tss <- sum((test$PE-mean(test$PE))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$PE,ypred)
rmse(test$PE,ypred)
mape(test$PE,ypred)
r2
adr2































dat <- as.data.frame(read_xls("Concrete_Data.xls"))
dat <- dat[,-8]
colnames(dat) <- c("V1","V2","V3","V4","V5","V6","V7","V8")
head(dat)
dim(dat)

testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train_scaled = dat[-testrows,]
test_scaled = dat[testrows,]

treemod <- tree (V8 ~ ., train_scaled)
train.pred = predict(treemod,train_scaled)
test.pred = predict(treemod,test_scaled)

frame = treemod$frame
leaves = frame$var == "<leaf>"
used = unique(frame$var[!leaves])
features.select = names(dat)[(names(dat) %in% used)]
features.select

library(ISLR2)
dat <- Boston
head(dat)
dim(dat)



testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train_scaled = dat[-testrows,]
test_scaled = dat[testrows,]

treemod <- tree (crim ~ ., train_scaled)
train.pred = predict(treemod,train_scaled)
test.pred = predict(treemod,test_scaled)

frame = treemod$frame
leaves = frame$var == "<leaf>"
used = unique(frame$var[!leaves])
features.select = names(dat)[(names(dat) %in% used)]
features.select


#test-train split
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]

tree.dat <- tree (V8 ~ ., train)
ypred <- predict (tree.dat , newdata = test)
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2

# ridge
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-8], train[,8], alpha = 0, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-8]))
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2

# lasso
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-8], train[,8], alpha = 1, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-8]))
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2

# rf
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
rf.dat <- randomForest(V8 ~ .,train, mtry=40)
ypred <- predict (rf.dat, newdata = test)
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2

# boost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
boost.dat <- gbm (V8 ~ ., data = train, distribution = "gaussian", n.trees = 50, interaction.depth = 4)
ypred <-  predict (boost.dat , newdata = test, n.trees = 50)
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2

# svr 
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]

dtn <- DecisionTreeNode$new(x=train[,-8],y=train[,8],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn$split_node()
ypred <- as.numeric(dtn$predict(test[,-8]))
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2

# svrf
set.seed(7)
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- rf(50,train[,-8],train[,8],test[,-8])
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2

# svrboost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- boost(train[,-8], train[,8], test[,-8], 30)
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2

dat <- read.csv("Accel.csv")
head(dat)
dat <- dat[,-c(2,3,6)]
head(dat)

testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
tree.dat <- tree (acceleration ~ ., train)
ypred <- predict (tree.dat , newdata = test)
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$acceleration,ypred)
rmse(test$acceleration,ypred)
mape(test$acceleration,ypred)
r2
adr2


testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
dtn <- DecisionTreeNode$new(x=train[,-1],y=train[,1],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn$split_node()
ypred <- as.numeric(dtn$predict(test[,-1]))
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$acceleration,ypred)
r2
adr2



# ridge
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 0, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$acceleration,ypred)
r2
adr2

# lasso
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 1, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$acceleration,ypred)
r2
adr2

# rf
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
rf.dat <- randomForest(acceleration ~ .,train, mtry=1)
ypred <- predict (rf.dat, newdata = test[,-1])
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$acceleration,ypred)
r2
adr2

# boost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
boost.dat <- gbm (acceleration ~ ., data = train, distribution = "gaussian", n.trees = 50, interaction.depth = 4)
ypred <-  predict (boost.dat , newdata = test, n.trees = 40)
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$acceleration,ypred)
r2
adr2


# svrf
set.seed(7)
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- rf(1,train[,-1],train[,1],test[,-1])
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$acceleration,ypred)
r2
adr2

# svrboost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- boost(train[,-8], train[,8], test[,-8], 30)
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2


dat <- read.csv("maxTorque.csv")
head(dat)
dat <- dat[,-c(2,3,4,6,16,17,18,27,28,30,31,32,33)]
head(dat)

# ridge
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 0, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$maximal.torque-ypred)^2)
tss <- sum((test$maximal.torque-mean(test$maximal.torque))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$maximal.torque,ypred)
r2
adr2

# lasso
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 1, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$maximal.torque-ypred)^2)
tss <- sum((test$maximal.torque-mean(test$maximal.torque))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$maximal.torque,ypred)
r2
adr2

testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
tree.dat <- tree ( maximal.torque ~ ., train)
ypred <- predict (tree.dat , newdata = test)
rss <- sum((test$maximal.torque-ypred)^2)
tss <- sum((test$maximal.torque-mean(test$maximal.torque))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$maximal.torque,ypred)
r2
adr2

head(dat)

# rf
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
rf.dat <- randomForest(maximal.torque ~ .,train, mtry=20)
ypred <- predict (rf.dat, newdata = test)
rss <- sum((test$maximal.torque-ypred)^2)
tss <- sum((test$maximal.torque-mean(test$maximal.torque))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$maximal.torque,ypred)
r2
adr2

# boost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
boost.dat <- gbm (acceleration ~ ., data = train, distribution = "gaussian", n.trees = 50, interaction.depth = 4)
ypred <-  predict (boost.dat , newdata = test, n.trees = 40)
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$acceleration,ypred)
r2
adr2

# svrf
set.seed(7)
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- rf(1,train[,-1],train[,1],test[,-1])
rss <- sum((test$maximal.torque-ypred)^2)
tss <- sum((test$maximal.torque-mean(test$maximal.torque))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$maximal.torque,ypred)
r2
adr2

# svrboost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- boost(train[,-8], train[,8], test[,-8], 30)
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2
















testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
dtn <- DecisionTreeNode$new(x=train[,-1],y=train[,1],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn$split_node()
ypred <- as.numeric(dtn$predict(test[,-1]))
rss <- sum((test$maximal.torque-ypred)^2)
tss <- sum((test$maximal.torque-mean(test$maximal.torque))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$maximal.torque,ypred)
r2
adr2

# ridge
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 0, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$maximal.torque-ypred)^2)
tss <- sum((test$maximal.torque-mean(test$maximal.torque))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$maximal.torque,ypred)
r2
adr2

# lasso
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 1, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$acceleration,ypred)
r2
adr2

dat <- read.csv("cpuSm.csv")
head(dat)
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
tree.dat <- tree ( usr ~ ., train)
ypred <- predict (tree.dat , newdata = test)
rss <- sum((test$usr-ypred)^2)
tss <- sum((test$usr-mean(test$usr))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$usr,ypred)
r2
adr2

testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
dtn <- DecisionTreeNode$new(x=train[,-1],y=train[,1],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn$split_node()
ypred <- as.numeric(dtn$predict(test[,-1]))
rss <- sum((test$usr-ypred)^2)
tss <- sum((test$usr-mean(test$usr))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$usr,ypred)
r2
adr2


# ridge
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 0, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$usr-ypred)^2)
tss <- sum((test$usr-mean(test$usr))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$usr,ypred)
r2
adr2

# lasso
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 1, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$usr-ypred)^2)
tss <- sum((test$usr-mean(test$usr))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$usr,ypred)
r2
adr2

mean(9.876155,10.31628,9.711858)
mean(0.705165,0.7268186,0.7256184)
mean(0.7037173,0.7254772,0.7242712)

dat <- read.csv("fuelCons.csv")
head(dat)
dat <- dat[,-c(2,3,4,6,21,22,24,25,26,27,37,38)]
head(dat)


testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
tree.dat <- tree ( fuel.consumption.country ~ ., train)
ypred <- predict (tree.dat , newdata = test)
rss <- sum((test$fuel.consumption.country-ypred)^2)
tss <- sum((test$fuel.consumption.country-mean(test$fuel.consumption.country))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$fuel.consumption.country,ypred)
r2
adr2

testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
dtn <- DecisionTreeNode$new(x=train[,-1],y=train[,1],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn$split_node()
ypred <- as.numeric(dtn$predict(test[,-1]))
rss <- sum((test$fuel.consumption.country-ypred)^2)
tss <- sum((test$fuel.consumption.country-mean(test$fuel.consumption.country))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$fuel.consumption.country,ypred)
r2
adr2


# ridge
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 0, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$fuel.consumption.country-ypred)^2)
tss <- sum((test$fuel.consumption.country-mean(test$fuel.consumption.country))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$fuel.consumption.country,ypred)
r2
adr2

# lasso
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
grid <- 10^ seq (10, -2, length = 100)
ridge.dat <- glmnet (train[,-1], train[,1], alpha = 1, lambda = grid, thresh = 1e-12)
ypred <- predict (ridge.dat , s = 0.1, newx = as.matrix(test[,-1]))
rss <- sum((test$fuel.consumption.country-ypred)^2)
tss <- sum((test$fuel.consumption.country-mean(test$fuel.consumption.country))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$fuel.consumption.country,ypred)
r2
adr2


# rf
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
rf.dat <- randomForest(fuel.consumption.country ~ .,train, mtry=20)
ypred <- predict (rf.dat, newdata = test)
rss <- sum((test$fuel.consumption.country-ypred)^2)
tss <- sum((test$fuel.consumption.country-mean(test$fuel.consumption.country))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$fuel.consumption.country,ypred)
r2
adr2

# boost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
boost.dat <- gbm (acceleration ~ ., data = train, distribution = "gaussian", n.trees = 50, interaction.depth = 4)
ypred <-  predict (boost.dat , newdata = test, n.trees = 40)
rss <- sum((test$acceleration-ypred)^2)
tss <- sum((test$acceleration-mean(test$acceleration))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$acceleration,ypred)
r2
adr2

# svrf
set.seed(7)
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- rf(20,train[,-1],train[,1],test[,-1])
rss <- sum((test$fuel.consumption.country-ypred)^2)
tss <- sum((test$fuel.consumption.country-mean(test$fuel.consumption.country))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$fuel.consumption.country,ypred)
r2
adr2

# svrboost
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
ypred <- boost(train[,-8], train[,8], test[,-8], 30)
rss <- sum((test$V8-ypred)^2)
tss <- sum((test$V8-mean(test$V8))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
mae(test$V8,ypred)
rmse(test$V8,ypred)
mape(test$V8,ypred)
r2
adr2


mean(0.8232584,0.8192387,0.8587368)
mean(0.7706353,0.7697994,0.7891214)
mean(0.7592355,0.758358,0.7786404)




dat <- read.csv("a6.csv")
head(dat)
dat <- dat[,-c(3,4)]
dat[,2] <- as.numeric(as.factor(dat[,2]))
head(dat)

testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
tree.dat <- tree ( a6 ~ ., train)
ypred <- predict (tree.dat , newdata = test)
rss <- sum((test$a6-ypred)^2)
tss <- sum((test$a6-mean(test$a6))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$a6,ypred)
r2
adr2

testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train = dat[-testrows,]
test = dat[testrows,]
dtn <- DecisionTreeNode$new(x=train[,-1],y=train[,1],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn$split_node()
ypred <- as.numeric(dtn$predict(test[,-1]))
rss <- sum((test$a6-ypred)^2)
tss <- sum((test$a6-mean(test$a6))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(dim(test)[1]-dim(test)[2]))/(tss/(dim(test)[1]-1)))
rmse(test$a6,ypred)
r2
adr2














