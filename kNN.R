
library(FNN) #load lib

# I have not yes tested

train <- read.csv("./dataset/webkb-train-stemmed.txt-R-data.csv", header=FALSE)
test <- read.csv("./dataset/webkb-test-stemmed.txt-R-data.csv", header=FALSE)

labels <- train[,1]
train <- train[,-1]

tlabels <- test[, 1]
test <- test[, -1]

results <- knn(train, test, labels, k = 10, algorithm="cover_tree")

diff <- tlabels - results
correct <- diff[diff != 0]
accury <- correct / length(labels)

accury
