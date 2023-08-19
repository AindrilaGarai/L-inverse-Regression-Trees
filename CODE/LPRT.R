library(datasets)
library(ISLR2)


data(iris)
data(ToothGrowth)


variance <- function(y){
  if(length(y) <= 1) return(0)
  var(y)
}


sv_ratio <- function(x, feature, val) # x= whole data, feature= which feature , mask= same feature region
{
  # new_data <- x[,!names(iris) %in% feature]
  new_data <- x[,-feature]
  d <- dim(new_data)[2]
  
  ran <- numeric(d+1)
  ran[d+1] <- val-min(x[,feature]) # length of feature
  
  for(i in 1:d)
  { ran[i] <- max(x[,i]) - min(x[,i]) }
  
  if(ran[d+1] == 0)
  { 
    volume <- prod(ran[-(d+1)])
    
    sur <- numeric(length=d-1)
    for(j in 1:d-1)
    {
      term <- 0
      for(k in (j+1):d) 
      {
        term <- term + ran[j]*ran[k]
        sur[j] <- term
      }
    }
    surface <- 2*sum(sur)
  }
  else { volume <- prod(ran)
  
  sur <- numeric(length=d)
  for(j in 1:d)
  {
    term <- 0
    for(k in (j+1):(d+1)) 
    {
      term <- term + ran[j]*ran[k]
      sur[j] <- term
    }
  }
  surface <- 2*sum(sur)
  
  } 
  rtn <- surface/volume
  return(rtn)
}

print.DecisionTreeNode <- function(node,level=0){
  response <- paste("|->",node$split_description)
  if(level < node$max_depth){
    if(!is.null(node$branches$left)){
      
      response <- paste0(response,"\n",paste(rep(" ",2*(level+1)),collapse=" "),print(node$branches$left,level+1))
      
    }
    if(!is.null(node$branches$right)){
      
      response <- paste0(response,"\n",paste(rep(" ",2*(level+1)),collapse=" "),print(node$branches$right,level+1))
      
    }
  }
  
  if(level==0) {
    cat(response)
  } else {
    return(response)
  }
}

DecisionTreeNode <- setRefClass("DecisionTreeNode",
                                 fields = list(x = "data.frame",
                                               y = "ANY",
                                               is_leaf="logical",
                                               split_description="character",
                                               best_split="list",
                                               branches="list",
                                               depth="numeric",
                                               minimize_func="function",
                                               min_information_gain="numeric",
                                               min_leaf_size="numeric",
                                               max_depth="numeric"),
                                 methods = list(
                                   initialize = function(...){
                                     defaults <- list(x = data.frame(),
                                                      y=c(),
                                                      depth=0,
                                                      minimize_func=variance,
                                                      min_information_gain=1e-3,
                                                      min_leaf_size=20,
                                                      max_depth=3,
                                                      is_leaf=T,
                                                      split_description="root",
                                                      best_split=NULL,
                                                      branches=list("left"=NULL,"right"=NULL))
                                     params <- list(...)
                                     fields <- names(getRefClass()$fields())
                                     for( field in fields){
                                       if (!(field %in% names(params))) {
                                         params[[field]] <- defaults[[field]]
                                       }
                                     }
                                     for( param in names(params)){
                                       do.call("<<-",list(param, params[[param]]))
                                     }
                                     
                                   },
                                   information_gain = function(mask){
                                     
                                     s1 = sum(mask)
                                     s2 = length(mask)-s1
                                     if ( s1 == 0 | s2 == 0) return(0)
                                     minimize_func(y)-s1/(s1+s2)*minimize_func(y[mask])-s2/(s1+s2)*minimize_func(y[!mask])
                                   },
                                   sv_ratio = function(feature, val) # x= whole data, feature= which feature , mask= same feature region
                                   {
                                     
                                     d <- dim(x)
                                     for (i in 1:d[2]) {
                                       take <- sum(x[,i] == feature)
                                       if(take == d[1]){
                                         f <- i
                                       }
                                     }
                                     
                                     new_data <- x[,-f] 
                                     d <- dim(new_data)[2]
                                     
                                     ran <- numeric(d+1)
                                     ran[d+1] <- val-min(feature) # length of feature
                                     
                                     for(i in 1:d)
                                     { ran[i] <- max(x[,i]) - min(x[,i]) }
                                     
                                     if(ran[d+1] == 0)
                                     { 
                                       volume <- prod(ran[-(d+1)])
                                       
                                       sur <- numeric(length=d-1)
                                       for(j in 1:d-1)
                                       {
                                         term <- 0
                                         for(k in (j+1):d) 
                                         {
                                           term <- term + ran[j]*ran[k]
                                           sur[j] <- term
                                         }
                                       }
                                       surface <- 2*sum(sur)
                                     }
                                     else { volume <- prod(ran)
                                     
                                     sur <- numeric(length=d)
                                     for(j in 1:d)
                                     {
                                       term <- 0
                                       for(k in (j+1):(d+1)) 
                                       {
                                         term <- term + ran[j]*ran[k]
                                         sur[j] <- term
                                       }
                                     }
                                     surface <- 2*sum(sur)
                                     
                                     } 
                                     rtn <- surface/volume
                                     return(rtn)
                                   },
                                   max_information_gain_split = function(feature){
                                     
                                     best_change = NA
                                     split_value = NA
                                     is_numeric = !(is.factor(feature)|is.logical(feature)|is.character(feature))
                                     
                                     previous_val <- NA
                                     for( val in sort(unique(feature))){
                                       
                                       mask <- feature == val
                                       
                                       if (is_numeric) mask <- feature < val
                                       change <- information_gain(mask) + (0.001*sv_ratio(feature,val))
                                       
                                       s1 = sum(mask)
                                       s2 = length(mask)-s1
                                       if(is.na(best_change) & s1 >= min_leaf_size & s2 >= min_leaf_size){
                                         best_change = change
                                         split_value = ifelse(is.na(previous_val),
                                                              val,
                                                              mean(c(val,previous_val)))
                                       } else if( change > best_change & s1 >= min_leaf_size & s2 >= min_leaf_size ){
                                         best_change = change
                                         split_value = ifelse(is_numeric,
                                                              mean(c(val,previous_val)),
                                                              val)
                                       }
                                       previous_val <- val
                                       
                                     }
                                     return(list("best_change"=best_change,
                                                 "split_value"=split_value,
                                                 "is_numeric"=is_numeric))
                                   },
                                   best_feature_split = function(){
                                     results <- sapply(x,function(feature) max_information_gain_split(feature))
                                     if (!all(is.na(unlist(results['best_change',])))) {
                                       best_name <- names(which.max(results['best_change',]))
                                       best_result <- results[,best_name]
                                       best_result[["name"]] <- best_name
                                       best_split <<- best_result
                                     }
                                     
                                   },
                                   best_mask = function(){
                                     best_mask <- x[,best_split$name] == best_split$split_value
                                     if(best_split$is_numeric){
                                       best_mask <- x[,best_split$name] < best_split$split_value
                                     }
                                     return(best_mask)
                                     
                                   },
                                   split_node = function() {
                                     if(depth < max_depth){ 
                                       best_feature_split() 
                                       if(!is.null(best_split) & best_split$best_change > min_information_gain ){
                                         
                                         mask = best_mask()
                                         if(sum(mask) >= min_leaf_size && length(mask)-sum(mask) >= min_leaf_size){
                                           is_leaf <<- F
                                           
                                           branches$left <<- .self$copy()
                                           branches$left$is_leaf <<- T
                                           branches$left$x <<-  branches$left$x[mask,]
                                           branches$left$y <<-  branches$left$y[mask]
                                           
                                           branches$left$split_description <<- ifelse(best_split$is_numeric,
                                                                                      paste(c(best_split$name,
                                                                                              "<",
                                                                                              as.numeric(as.character(best_split$split_value))),
                                                                                            collapse = " "),
                                                                                      paste(c(best_split$name,
                                                                                              "=",
                                                                                              best_split$split_value),
                                                                                            collapse = " "))
                                           
                                           branches$left$depth <<-  branches$left$depth+1
                                           branches$left$branches <<- list("left"=NULL,"right"=NULL)
                                           branches$left$split_node()
                                           
                                           branches$right <<- .self$copy()
                                           branches$right$is_leaf <<- T
                                           branches$right$x <<-  branches$right$x[!mask,]
                                           branches$right$y <<-  branches$right$y[!mask]
                                           
                                           branches$right$split_description <<- ifelse(best_split$is_numeric, 
                                                                                       paste(c(best_split$name, ">=",
                                                                                               best_split$split_value),
                                                                                             collapse = " "),
                                                                                       paste(c(best_split$name,
                                                                                               "!=",
                                                                                               best_split$split_value),
                                                                                             collapse = " "))
                                           
                                           branches$right$depth <<-  branches$right$depth+1
                                           branches$right$branches <<- list("left"=NULL,"right"=NULL)
                                           branches$right$split_node()
                                         }
                                       }
                                     }
                                     if(is_leaf){
                                       split_description <<- ifelse(identical(minimize_func,variance),
                                                                    paste(c(split_description,
                                                                            ":",
                                                                            "predict - ",
                                                                            mean(y)),
                                                                          collapse=" "),
                                                                    paste(c(split_description,
                                                                            ":",
                                                                            "predict - ",
                                                                            names(which.max(table(y)))),
                                                                          collapse=" "))
                                     }
                                     
                                   },
                                   predict_row = function(row){
                                     if(is_leaf){
                                       predict_value <- ifelse(identical(minimize_func,variance),
                                                               mean(y),
                                                               names(which.max(table(y))))
                                     } else {
                                       if(best_split$is_numeric){
                                         left = row[best_split$name] < best_split$split_value
                                       } else{
                                         left = row[best_split$name] == best_split$split_value
                                       }
                                       if(left){
                                         predict_value = branches$left$predict_row(row)
                                       } else {
                                         predict_value = branches$right$predict_row(row)
                                       }
                                     }
                                     return(predict_value)
                                   },
                                   
                                   predict = function(features){
                                     pred <- character(length=dim(features)[1])
                                     if(identical(minimize_func,variance)) pred <- numeric(length=dim(features)[1])
                                     for(i in 1:dim(features)[1]){
                                       pred[i] = predict_row(features[i,])
                                     }
                                     pred
                                   }
                                 ))

DecisionTree <- setRefClass("DecisionTree",
                            fields = list(minimize_func="function",
                                          min_information_gain="numeric",
                                          min_leaf_size="numeric",
                                          max_depth="numeric",
                                          root = "DecisionTreeNode"),
                            methods = list(
                              initialize = function(...){
                                defaults <- list(minimize_func=variance,
                                                 min_information_gain=1e-3,
                                                 min_leaf_size=20,
                                                 max_depth=3,
                                                 root=NULL)
                                
                                params <- list(...)
                                fields <- names(getRefClass()$fields())
                                for( field in fields){
                                  if (!(field %in% names(params))) {
                                    params[[field]] <- defaults[[field]]
                                  }
                                }
                                for( param in names(params)){
                                  do.call("<<-",list(param, params[[param]]))
                                }
                                
                              },
                              fit = function(features,target){
                                root <<- DecisionTreeNode$new(x=features,
                                                              y=target,
                                                              minimize_func=minimize_func,
                                                              min_information_gain=min_information_gain,
                                                              min_leaf_size=min_leaf_size,
                                                              max_depth=max_depth
                                )
                                root$split_node()
                                
                              },
                              predict = function(features){
                                root$predict(features)
                              }
                            ))

print.DecisionTree <- function(tree){
  print(tree$root)
}

a <- sample(150,100)
train <- iris[a,]
test <- iris[-a,]

dtn1 <- DecisionTreeNode$new(x=train[,-1],y=train[,1],max_depth=7,min_leaf_size=10,min_information_gain=1e-3)
dtn1$split_node()
print(dtn1)

pred1 <- as.numeric(dtn1$predict(test[,-1]))
rmse(test[,1],pred1)
mae(test[,1],pred1)
mape(test[,1],pred1)

rss <- sum((test[,1]-pred1)^2)
tss <- sum((test[,1]-mean(test[,1]))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(117-3))/(tss/(117-1)))
r2
adr2


dat <- read.table("auto-mpg.data")
dat <- dat[,-9]
dat[,4] <- as.numeric(dat[,4])
dat <- na.omit(dat)
a <- sample(392,300)
train <- dat[a,]
test <- dat[-a,]

dtn1 <- DecisionTreeNode1$new(x=train[,2:8],y=train[,1],max_depth=5,min_leaf_size=10,min_information_gain=1e-7)
dtn1$split_node()
print(dtn1)
library(Metrics)
pred1 <- as.numeric(dtn1$predict(test[,2:8]))
rmse(test[,1],pred1)
mae(test[,1],pred1)


dtn <- DecisionTreeNode$new(x=train[,-1],y=train[,1],max_depth=5,min_leaf_size=10,min_information_gain=1e-7)
dtn$split_node()
print(dtn)
pred <- dtn$predict(test[,-1])
rmse(test[,1],pred)
mae(test[,1],pred)

rf <- function(no_of_trees,x_train,y_train,x_test)
{
  pred <- matrix(0,dim(x_test)[1],no_of_trees)
  for(i in 1:no_of_trees)
  {
    set.seed(i)
    boot_data <- sample(1:dim(x_train)[1], dim(x_train)[1], replace = TRUE)
    #features <- sample(1:dim(x_train)[2], ceiling(sqrt(dim(x_train)[2])))
    x <- x_train[boot_data,]
    y <- y_train[boot_data]
    
    dt_bts <- DecisionTree(max_depth=5,min_leaf_size=5,minimize_func=variance)
    dt_bts$fit(x,y)
    dt_pred <- dt_bts$predict(x_test) 
    for(j in 1:dim(x_test)[1])
    {
      pred[j,i] <- (dt_pred[j]) 
    }
  }
  rvf <- NULL
  new <- numeric(no_of_trees)
  for (k in 1:dim(x_test)[1])
  {
    new <- pred[k,]
    rvf <- c(rvf,mean(new))
  }
  rtn <- rvf 
  return(rtn)
  
}
newpred <- rf(50,train[,-1],train[,1],test[,-1])
mae(test[,1],newpred)
mape(test[,1],newpred)
rmse(test[,1],newpred)
rss <- sum((test[,1]-newpred)^2)
tss <- sum((test[,1]-mean(test[,1]))^2)
r2 <- 1-(rss/tss)
adr2 <- 1-((rss/(117-3))/(tss/(117-1)))
r2
adr2


boost <- function(x_train, y_train, x_test, no_of_trees)
{
  avg <- mean(y_train)
  train_pred <- avg
  test_pred <- avg
  residual <- y_train - avg
  
  for (i in 1:no_of_trees) 
  {
  dt_bts <- DecisionTree(max_depth=7, min_leaf_size=5, minimize_func=variance)
  dt_bts$fit(x_train,residual)
  dt_pred <- dt_bts$predict(x_train)
  dt_test_pred <- dt_bts$predict(x_test)
  
  train_pred <- train_pred + (0.1*dt_pred)
  test_pred <- test_pred + (0.1*dt_test_pred)
  residual <- y_train - train_pred
  }
  return(test_pred)
}

tsam <- sample(1:150,100)
x_train <- iris[tsam,1:3]
x_test <- iris[-tsam,1:3]
y_train <- iris[tsam,4]
y_test <- iris[-tsam,4]

a <- boost(x_train,y_train,x_test,50)
























