require(sirus)
library(Metrics)
path <- "C:/Users/loure/Desktop/Tese/sns.csv"

content <- read.csv(path)

#print(content)


path2 <- "C:/Users/loure/Desktop/Tese/sns.csv"

sns <- read.csv(path2)

sns_prep <- sns[,-1]
sns_clean <- sns_prep[,-1]
factors <- factor(sns_clean$ars)
encoded <- as.numeric(factors)
sns_clean$ars <- encoded

data2 <- sns_clean                                          # Duplicate data frame
for(i in 1:ncol(sns_clean)) {                                   # Replace NA in all columns
  data2[ , i][is.na(data2[ , i])] <- mean(data2[ , i], na.rm = TRUE)
}


data3 <- na.aggregate(sns_clean)   


split <- sample(c(rep(0, 0.8 * nrow(data2)), rep(1, 0.2 * nrow(data2))))

#table(split)

train <- data2[split == 0, ]   

test <- data2[split == 1, ] 

train_y <- train[,"urg"]

test_y <- test[,"urg"]

train_x <- train[,-7]

test_x <- test[,-7]

#a <- sirus.cv(train_x, train_y) #Nao preciso disto qd for escolher o p0
#print(a$p0.stab)             ### RECOMENDED FOR REGRESSION CHECK MANUAL OF CV

sirus_funct <- function(New_p0){
  #print(New_p0)
  # p0=a$p0.stab
  #FIT WITH THAT VALUE
  model <- sirus.fit(train_x, train_y, p0=New_p0, num.rule.max=100)
  #plot.error <- sirus.plot.cv(a)$error
  #plot(plot.error)

  predicted_y_train <- sirus.predict(model, train_x)
  error_train <- mae(actual = train_y, predicted = predicted_y_train)
  length_train <- length(model$rules)
  
  predicted_y_test <- sirus.predict(model, test_x)
  error_test <- mae(actual = test_y, predicted = predicted_y_test)
  #print(error)
  length_test <- length(model$rules)

  
  
  output <- list(error_test, length_test, error_train, length_train)
  return (output)
}

mae_values_test <- rep()
length_values_test <-rep()
mae_values_train <- rep()
length_values_train <-rep()

for (val in c(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1)) {
  
  output <- sirus_funct(val)
  un <- unlist(output)
  mae_values_test <- append(mae_values_test, un[1])
  length_values_test <- append(length_values_test, un[2])
  
  mae_values_train <- append(mae_values_train, un[3])
  length_values_train <- append(length_values_train, un[4])
}

print(mae_values_train)
print(length_values_train)
print(mae_values_test)
print(length_values_test)  
  
  
plot(mae_values_train, length_values_train,type="b",pch=22,xlim=c(min(mae_values_train),max(mae_values_test)), ylim=c(min(length_values_train),max(length_values_train)),col="red",lty=1,xlab="Mean Absolute Error",lwd=2,ylab="Rules",xaxt="n")
lines(mae_values_test,length_values_test,type="b",col="red",lty=2,lwd=2)
axis(1, at = seq(200,700, by = 50), las=2)
title(main="Accuracy by number of Rules", col.main="black", font.main=2)
grid()
legend("topright",legend = c("Train", "Validation"),lty=c(1,2,3),col=c("red","red"),bg="white",lwd=2)