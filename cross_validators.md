1636 Competition Assignment - The Cross Validators
================
Feb 2023

### Group Members

- Talis Tebecis h12135076
- Ema Ivanova h11900690
- Karin Nagamine h12131854
- Stephan Gavric h12025079

## Setup

First we import the data and load required packages.

``` r
#load packages
pacman::p_load(here, tidyverse, ranger, caret, tuneRanger, BMS)

#load training data
training_data <- read.csv(here("input_data", "trainig_test_data.csv"))

#load hold out data
holdout_data <- read.csv(here("input_data", "holdout_data.csv"))
```

## Data cleaning and imputation

Then, we check the data for NAs and impute where necessary using median
for numerical variables and mode for categorical variables.

``` r
### Training Data
#check NAs - we find 21,668 observations with NAs, so we shouldn't just drop them
checkNAs <- training_data %>% filter_all(any_vars(is.na(.))) %>% count()
checkNA_vars <- training_data %>% summarise_all(funs(sum(is.na(.))))

#replace NAs with medians of columns for numeric values
training_data1 <- training_data %>% 
  mutate_if(is.numeric, funs(replace(., is.na(.), median(., na.rm = TRUE))))

#check NAs - reduced down to 12,728 obs with NAs
checkNAs <- training_data1 %>% filter_all(any_vars(is.na(.))) %>% count()
checkNA_vars <- training_data1 %>% summarise_all(funs(sum(is.na(.))))

#create mode function for categorical variables
getmode <- function(v) {
   uniqv <- unique(v)
   uniqv[which.max(tabulate(match(v, uniqv)))]
}

#replace NAs with modes of columns for character values
training_data2 <- training_data1 %>% 
  mutate_if(is.character, funs(replace(., is.na(.), getmode(.))))

#check NAs - reduced down to 0 obs with NAs
checkNAs <- training_data2 %>% filter_all(any_vars(is.na(.))) %>% count()

#turn all categorical variables into factors, then into numerical variables, for computational simplicity
training_data2 <- as.data.frame(unclass(training_data2),
                                stringsAsFactors = TRUE)
training_data2 <- lapply(training_data2, as.numeric) %>% 
  as.data.frame()


### Holdout Data

#replace NAs with medians of columns for numeric values
holdout_data <- holdout_data %>% 
  mutate_if(is.numeric, funs(replace(., is.na(.), median(., na.rm = TRUE))))

#replace NAs with modes of columns for character values
holdout_data <- holdout_data %>% 
  mutate_if(is.character, funs(replace(., is.na(.), getmode(.))))

#turn all categorical variables into factors, then into numerical variables, for computational simplicity
holdout_data <- as.data.frame(unclass(holdout_data),
                                stringsAsFactors = TRUE)
holdout_data <- lapply(holdout_data, as.numeric) %>% 
  as.data.frame()
```

## Random Forest

For modelling, we first try a Random Forest model using ranger, then a
tuned random forest using the caret package.

``` r
#basic random forest
rf_1 = ranger(data = training_data2, dependent.variable.name = "income", importance = "impurity")

#tuned random forest
#cross validation setup
control <- trainControl(method = "cv", number = 8)

#tuning grid
tuning_grid <- expand.grid(mtry = c(5, 10, 15, 20, 25, 28),
                           splitrule = "variance",
                           min.node.size = seq(2, 8, by = 3))

#run models and extract best model
rf_caret = caret::train(data = training_data2,
                 income ~ .,
                 method = "ranger",
                 trControl = control,
                 tuneGrid = tuning_grid,
                 importance = "impurity")
rf_2 <- rf_caret$finalModel
```

## Boosted Regression

Next, we test Boosted Regression models, also using caret.

``` r
#setup control
control <- trainControl(method = "cv", number = 5)

### Tuning attempt 1

#tuning grid
tuning_grid <- expand.grid(nrounds = c(130, 150, 170),
                          max_depth = c(5, 6, 7),
                          eta = c(0.2, 0.3, 0.4),
                          gamma = 0.02,
                          colsample_bytree = 1,
                          min_child_weight = 1,
                          subsample = 1)

#run model and extract models
gb_caret <- caret::train(data = training_data2,
                 income ~ .,
                 method = "xgbTree",
                 trControl = control,
                 tuneGrid = tuning_grid,
                 verbosity = 0)
gb_1 <- gb_caret$finalModel

### Tuning attempt 2

#tuning grid
tuning_grid <- expand.grid(nrounds = c(170, 180, 190),
                          max_depth = 5,
                          eta = c(0.1, 0.2),
                          gamma = 0.02,
                          colsample_bytree = 1,
                          min_child_weight = 1,
                          subsample = 1)

#run model and extract models
gb_caret1 <- caret::train(data = training_data2,
                 income ~ .,
                 method = "xgbTree",
                 trControl = control,
                 tuneGrid = tuning_grid,
                 verbosity = 0)
gb_2 <- gb_caret1$finalModel
```

## Bayesian Model Averaging

Then, we make predictions using Bayesian Model Averaging with the BMS
package.

``` r
#run BMA model
bma_1 <- bms(training_data2,
           mprior="uniform",
           burn=20000,
           iter=50000,
           nmodel=2000,
           g="BRIC",
           mcmc="bd")

#predictions
bma_pred <- pred.density(bma_1, training_data2)

#RMSE = 1201.7
rmse_check <- cbind(bma_pred$fit, training_data2$income) %>% 
  as.data.frame() %>% 
  mutate(sq_err = (V1-V2)^2)

RMSE_BMA <- sqrt(sum(rmse_check$sq_err)/count(rmse_check))
```

## Compare models

We extract the top model from each tuning process and compare them by
RMSE. The best model was the first tuning round of the Boosted
Regression model with an RMSE of 966.8296.

``` r
#basic random forest prediction error. RMSE = 1028.189
sqrt(rf_1$prediction.error)

#tuned random forest prediction error. RMSE = 1017.295
sqrt(rf_2$prediction.error)

#tuned boosted regression prediction error. RMSE = 966.8296
min(gb_caret$results$RMSE)

#second tuned boosted regression prediction error. RMSE = 967.0741
min(gb_caret1$results$RMSE)

#Bayesian model averaging prediction error. RMSE = 1201.7
RMSE_BMA

#gb_1 (from gb_caret) was chosen as model with lowest RMSE
```

## Final model

We then run the chosen model a final time with the parameter values as
outlined below. We then make predictions based on the holdout data and
export them to a .csv file.

``` r
### Run final model (other chunks set eval=FALSE for computational time)
#setup control
control <- trainControl(method = "cv", number = 5)

#tuning grid
tuning_grid <- expand.grid(nrounds = 170,
                          max_depth = 5,
                          eta = 0.2,
                          gamma = 0.02,
                          colsample_bytree = 1,
                          min_child_weight = 1,
                          subsample = 1)

#run model and extract models
gb_caret_final <- caret::train(data = training_data2,
                 income ~ .,
                 method = "xgbTree",
                 trControl = control,
                 tuneGrid = tuning_grid,
                 verbosity = 0)

gb_final <- gb_caret_final$finalModel

#final predictions
final_predictions <- predict(gb_final, newdata = as.matrix(holdout_data)) %>% 
  as.data.frame() %>% 
  rename(income_cross_validators = 1)

#export to csv
write.csv(final_predictions,
        "./output_data/predictions.csv",
        row.names=FALSE)
```
