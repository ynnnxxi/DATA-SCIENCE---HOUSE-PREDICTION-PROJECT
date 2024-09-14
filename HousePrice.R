data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv")
summary(data)
fix(data)
names(data)
str(data)
# Z-점수 계산
z_scores <- (data - mean(data)) / sd(data)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv")
# Z-점수 계산
z_scores <- (data - mean(data)) / sd(data)
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3]
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv")
str(data)
# Z-점수 계산
z_scores <- (data - mean(data)) / sd(data)
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3]
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv")
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[, -1])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv")
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[, -1])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv")
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[, -12])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv")
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[, -12])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv")
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[, -12])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
print(outliers)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv")
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
print(outliers)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
print(outliers)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
print(outliers)
options(max.print = 1000)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
options(max.print = 1000)
print(outliers)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
str(data)
# 각 열에 대한 Z-점수 계산 (1번 열은 ID 또는 다른 식별자일 수 있으므로 제외)
z_scores <- scale(data[])  # 1번 열을 제외한 모든 열에 대한 Z-점수 계산
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
print(outliers)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
str(data)
# 각 열에 대한 Z-점수 계산
z_scores <- scale(data[])
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
print(outliers)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
str(data)
# 각 열에 대한 Z-점수 계산
z_scores <- scale(data[])
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- data[abs(z_scores) > 3, ]
print(outliers)
sum(is.na(data))
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- apply(z_scores, 2, function(x) abs(x) > 3)
# 실제 데이터에서 이상치가 있는 행 추출
data_outliers <- data[rowSums(outliers) > 0, ]
print(data_outliers)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- apply(z_scores, 2, function(x) abs(x) > 3)
# 실제 데이터에서 이상치가 있는 행 추출
data_outliers <- data[rowSums(outliers) > 0, ]
# 이상치가 있는 행 삭제
data_cleaned <- data[rowSums(outliers) == 0, ]
# 이상치가 제거된 데이터 확인
head(data_cleaned)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치 식별 (예: Z-점수가 3보다 큰 경우)
outliers <- apply(z_scores, 2, function(x) abs(x) > 3)
# 실제 데이터에서 이상치가 있는 행 추출
data_outliers <- data[rowSums(outliers) > 0, ]
print(data_outliers)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치가 아닌 값을 식별 (예: Z-점수의 절대값이 3 이하인 경우)
not_outliers <- apply(z_scores, 2, function(x) abs(x) <= 3)
# 모든 변수에서 이상치가 아닌 관측치만을 가지고 있는 행을 식별
not_outlier_rows <- apply(not_outliers, 1, all)
# 이상치가 없는 행만을 가진 데이터 프레임 생성
clean_data <- data[not_outlier_rows, ]
# 결과 출력
print(clean_data)
lm.fit = lm(Price~., data = clean_data)
summary(lm.fit)
### 능형 회귀 ###
library(glmnet)
### 능형 회귀 ###
library(glmnet)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치가 아닌 값을 식별 (예: Z-점수의 절대값이 3 이하인 경우)
not_outliers <- apply(z_scores, 2, function(x) abs(x) <= 3)
# 모든 변수에서 이상치가 아닌 관측치만을 가지고 있는 행을 식별
not_outlier_rows <- apply(not_outliers, 1, all)
# 이상치가 없는 행만을 가진 데이터 프레임 생성
clean_data <- data[not_outlier_rows, ]
# 결과 출력
print(clean_data)
y = clean_data$Price
grid = 10^seq(10, -2, length = 100)
ridge.mod = glmnet(x, y, alpha = 0, lambda = grid)
### 능형 회귀 ###
library(glmnet)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치가 아닌 값을 식별 (예: Z-점수의 절대값이 3 이하인 경우)
not_outliers <- apply(z_scores, 2, function(x) abs(x) <= 3)
# 모든 변수에서 이상치가 아닌 관측치만을 가지고 있는 행을 식별
not_outlier_rows <- apply(not_outliers, 1, all)
# 이상치가 없는 행만을 가진 데이터 프레임 생성
clean_data <- data[not_outlier_rows, ]
# 결과 출력
print(clean_data)
x = model.matrix(Price~., clean_data)[, -12]
y = clean_data$Price
grid = 10^seq(10, -2, length = 100)
ridge.mod = glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(ridge.mod))
ridge.mod$lambda[50]
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1, 50]^2))
ridge.mod$lambda[60]
coef(ridge.mod)[, 60]
predict(ridge.mod, s = 50, type = "coefficients")[1:12,]
set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
y.test = y[test]
ridge.mod = glmnet(x[train,], y[train], alpha = 0, lambda = grid, thresh = 1e-12)
ridge.pred = predict(ridge.mod, s=4, newx = x[test,])
mean((ridge.pred-y.test)^2)
mean((mean(y[train])-y.test)^2)
ridge.pred = predict(ridge.mod, s = 1e10, newx = x[test,])
mean((ridge.pred-y.test)^2)
ridge.pred = predict(ridge.mod, s=0, newx=x[test,])
mean((ridge.pred-y.test)^2)
lm(y~x, subset = train)
ridge.pred = predict(ridge.mod, s = 0, newx = x[test,],
                     type = "coefficients")[1:12,]
set.seed(1)
cv.out = cv.glmnet(x[train,], y[train], alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam
ridge.pred=predict(ridge.mod, s=bestlam, newx=x[test,])
mean((ridge.pred-y.test)^2)
out=glmnet(x, y, alpha=0)
predict(out, type="coefficients", s=bestlam)[1:12,]
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치가 아닌 값을 식별 (예: Z-점수의 절대값이 3 이하인 경우)
not_outliers <- apply(z_scores, 2, function(x) abs(x) <= 3)
# 모든 변수에서 이상치가 아닌 관측치만을 가지고 있는 행을 식별
not_outlier_rows <- apply(not_outliers, 1, all)
# 이상치가 없는 행만을 가진 데이터 프레임 생성
clean_data <- data[not_outlier_rows, ]
# 결과 출력
print(clean_data)
lm.fit = lm(Price~., data = clean_data)
summary(lm.fit)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치가 아닌 값을 식별 (예: Z-점수의 절대값이 3 이하인 경우)
not_outliers <- apply(z_scores, 2, function(x) abs(x) <= 3)
# 모든 변수에서 이상치가 아닌 관측치만을 가지고 있는 행을 식별
not_outlier_rows <- apply(not_outliers, 1, all)
# 이상치가 없는 행만을 가진 데이터 프레임 생성
clean_data <- data[not_outlier_rows, ]
# 결과 출력
print(clean_data)
lasso.mod = glmnet(x[train,], y[train], alpha = 1, lambda = grid)
par(mar=c(4, 4, 2, 2))
options(repr.plot.width=8, repr.plot.height=4)
plot(lasso.mod)
set.seed(1)
cv.out = cv.glmnet(x[train,], y[train], alpha = 1)
plot(cv.out)
bestlam = cv.out$lambda.min
lasso.pred = predict(lasso.mod, s = bestlam, newx = x[test,])
mean((lasso.pred-y.test)^2)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치가 아닌 값을 식별 (예: Z-점수의 절대값이 3 이하인 경우)
not_outliers <- apply(z_scores, 2, function(x) abs(x) <= 3)
# 모든 변수에서 이상치가 아닌 관측치만을 가지고 있는 행을 식별
not_outlier_rows <- apply(not_outliers, 1, all)
# 이상치가 없는 행만을 가진 데이터 프레임 생성
clean_data <- data[not_outlier_rows, ]
# 결과 출력
print(clean_data)
lm.fit = lm(Price~., data = clean_data)
summary(lm.fit)
### 능형 회귀 ###
library(glmnet)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
# 이상치가 아닌 값을 식별 (예: Z-점수의 절대값이 3 이하인 경우)
not_outliers <- apply(z_scores, 2, function(x) abs(x) <= 3)
# 모든 변수에서 이상치가 아닌 관측치만을 가지고 있는 행을 식별
not_outlier_rows <- apply(not_outliers, 1, all)
# 이상치가 없는 행만을 가진 데이터 프레임 생성
clean_data <- data[not_outlier_rows, ]
# 결과 출력
print(clean_data)
x = model.matrix(Price~., clean_data)[, -12]
y = clean_data$Price
grid = 10^seq(10, -2, length = 100)
ridge.mod = glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(ridge.mod))
ridge.mod$lambda[50]
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1, 50]^2))
ridge.mod$lambda[60]
coef(ridge.mod)[, 60]
predict(ridge.mod, s = 50, type = "coefficients")[1:12,]
set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
y.test = y[test]
ridge.mod = glmnet(x[train,], y[train], alpha = 0, lambda = grid, thresh = 1e-12)
ridge.pred = predict(ridge.mod, s=4, newx = x[test,])
mean((ridge.pred-y.test)^2)
mean((mean(y[train])-y.test)^2)
ridge.pred = predict(ridge.mod, s = 1e10, newx = x[test,])
mean((ridge.pred-y.test)^2)
ridge.pred = predict(ridge.mod, s=0, newx=x[test,])
mean((ridge.pred-y.test)^2)
lm(y~x, subset = train)
ridge.pred = predict(ridge.mod, s = 0, newx = x[test,],
                     type = "coefficients")[1:12,]
set.seed(1)
cv.out = cv.glmnet(x[train,], y[train], alpha=0)
plot(cv.out)
data = read.csv("D:\\Project\\Chapter01\\project\\again\\housedata.csv", header = TRUE)
# 각 열에 대한 Z-점수 계산
z_scores <- as.data.frame(scale(data))
