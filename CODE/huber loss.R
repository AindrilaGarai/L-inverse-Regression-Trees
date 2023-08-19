library(yardstick)

delta =20
huber_loss <- function(y)
{
  er <- y-mean(y)
  if(mean(abs(er)) <= delta)
  {rtn <- mean((er^2)/2)}
  else
  {
    rtn <- mean(delta*(abs(er)-(delta/2)))
  }
  return(rtn)
}
huber_loss(ToothGrowth[,'len'])

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
                                                     minimize_func=huber_loss,
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
                                    # new_data <- x[,!names(iris) %in% feature]
                                    d <- dim(x)
                                    for (i in 1:d[2]) {
                                      take <- sum(x[,i] == feature)
                                      if(take == d[1]){
                                        f <- i
                                      }
                                    }
                                    
                                    new_data <- x[,-f] #78767768
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
