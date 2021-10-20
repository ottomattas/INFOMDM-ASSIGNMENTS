train.mnb <- function (dtm,labels) 
{
call <- match.call()
V <- ncol(dtm)
N <- nrow(dtm)
prior <- table(labels)/N
labelnames <- names(prior)
nclass <- length(prior)
cond.probs <- matrix(nrow=V,ncol=nclass)
dimnames(cond.probs)[[1]] <- dimnames(dtm)[[2]]
dimnames(cond.probs)[[2]] <- labelnames
index <- list(length=nclass)
for(j in 1:nclass){
 index[[j]] <- c(1:N)[labels == labelnames[j]]
}

for(i in 1:V){
  for(j in 1:nclass){
    cond.probs[i,j] <- (sum(dtm[index[[j]],i])+1)/(sum(dtm[index[[j]],])+V)
  }
}
list(call=call,prior=prior,cond.probs=cond.probs)    
}

predict.mnb <-
function (model,dtm) 
{
classlabels <- dimnames(model$cond.probs)[[2]]
logprobs <- dtm %*% log(model$cond.probs)
N <- nrow(dtm)
nclass <- ncol(model$cond.probs)
logprobs <- logprobs+matrix(nrow=N,ncol=nclass,log(model$prior),byrow=T)
classlabels[max.col(logprobs)]
}
