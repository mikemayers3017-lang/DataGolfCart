library(tidyverse)
library(xgboost)
library(Matrix)
library(SHAPforxgboost)
library(slider)
library(lubridate)
library(caret)

source("playersDataFunctions.R")

prepare_feature_data <- function(data) {
  # Creates training data for Course Fit Models using sliding last 50 averages of predictors
  #
  # Result:
  #   [ player | dates | course | sgTot | is_top_20 | n_rounds_history | sg{stat}_l50 | drDist_l50 | drAcc_l50]
  
  # Fix dates format in data
  data_date_fixed <- data %>% 
    mutate(dates = mdy(dates)) %>% 
    filter(!is.na(dates))
  
  # Make Sliding L50 Averages for Predictors
  #   Iterates through dataset that is ordered by date and Round
  #   Takes average of each stat of last 50 rounds in prior rows for each player
  #   Keeps 1 row per date checkpoint for last 50 averages
  #   Gets rid of checkpoints where the player doesn't have 10+ rounds logged
  #   [ player | dates | n_rounds_history | sg{stat}_l50 | drDist_l50 | drAcc_l50]
  player_form <- data_date_fixed %>% 
    filter(Round != "Event") %>% 
    arrange(player, dates, Round) %>% 
    group_by(player) %>% 
    mutate(
      n_rounds_history = slide_int(rep(1, n()), sum, .before = 36, .after = -1, .complete = FALSE),
      
      across(c(sgPutt, sgArg, sgApp, drDist, drAcc),
             ~slide_dbl(.x, \(x) mean(x, na.rm = TRUE), .before = 36, .after = -1, .complete = FALSE),
             .names = "{.col}_l50")
    ) %>% 
    filter(n_rounds_history >= 10) %>% 
    select(player, dates, n_rounds_history, ends_with("_l50")) %>% 
    group_by(player, dates) %>% 
    slice(1) %>% 
    ungroup()
  
  # Grabs All Unique Player, Date, Course Combos with sgTot and sets is_top_20
  #   [ player | dates | course | sgTot | is_top_20 ]
  event_targets <- data_date_fixed %>%
    filter(Round == "Event") %>%
    mutate(is_top_20 = ifelse(finish <= 20, 1, 0)) %>%
    select(player, dates, course, sgTot, is_top_20)

  # Creates Training Data
  #   [ player | dates | course | sgTot | is_top_20 | n_rounds_history | sg{stat}_l50 | drDist_l50 | drAcc_l50]
  final_training_set <- event_targets %>%
    inner_join(player_form, by = c("player", "dates"))
  
  # Rename Final Training Set Columns
  final_training_set <- final_training_set %>% 
    rename(
      sg_putt_l50 = sgPutt_l50,
      sg_arg_l50 = sgArg_l50,
      sg_app_l50 = sgApp_l50,
      dr_dist_l50 = drDist_l50,
      dr_acc_l50 = drAcc_l50
    )

  return(final_training_set)
}

trainCourseFitModels <- function() {
  
  processed_data <- prepare_feature_data(data)
  course_models <- list()
  training_ledger <- data.frame()
  
  # Get all course names
  courses <- unique(courseStatsData$course)
  # courses <- c("Augusta National Golf Club")
  # courses <- c("Quail Hollow Club")
  # courses <- c("Albany GC") # Only top 20's ...
  
  # Training Proportion
  train_prop <- 0.65
  
  for (curr_course in courses) {
    message("Training Course Fit for: ", curr_course)
    
    # DataFrame of Rounds (and preceding data) for current course
    course_df <- processed_data %>% 
      filter(course == curr_course) %>% 
      drop_na()
    
    # Number of Total 'Player Playing Event at this Course' occurances
    n_total <- nrow(course_df)
    
    if (n_total < 20) {
      message("  - Skipping: Not enough data points (n < 40)")
      next
    }
    
    # 1. STRATIFIED RAMDON SPLIT
    set.seed(42)
    if (length(unique(course_df$is_top_20)) > 1) {
      # Use Stratified Split if possible
      train_idx <- caret::createDataPartition(course_df$is_top_20, p = train_prop, list = FALSE)
    } else {
      # Fallback to Simple Random Split if only one class exists
      message("  - Note: Only one class detected for is_top_20. Using simple random split.")
      train_idx <- sample(1:n_total, size = floor(train_prop * n_total))
    }
    
    training_df <- course_df[train_idx, ]
    test_df <- course_df[-train_idx, ]
    
    #   Define Predictors
    #X_vars <- c("n_rounds_history", "sg_putt_l50", "sg_arg_l50", "sg_app_l50", "dr_dist_l50", "dr_acc_l50")
    X_vars <- c("sg_putt_l50", "sg_arg_l50", "sg_app_l50", "dr_dist_l50", "dr_acc_l50")
    
    X_train <- as.matrix(training_df[, X_vars])
    X_test  <- as.matrix(test_df[, X_vars])
    
    #   Targets
    y_reg_train <- training_df$sgTot
    y_reg_test  <- test_df$sgTot
    y_clf_train <- training_df$is_top_20
    y_clf_test  <- test_df$is_top_20
    
    #   Create DMatrices
    dtrain_reg <- xgb.DMatrix(data = X_train, label = y_reg_train)
    dtest_reg  <- xgb.DMatrix(data = X_test, label = y_reg_test)
    dtrain_clf <- xgb.DMatrix(data = X_train, label = y_clf_train)
    dtest_clf  <- xgb.DMatrix(data = X_test, label = y_clf_test)
    
    # 3. REGULARIZED PARAMS
    params_base <- list(
      max_depth = 3,
      eta = 0.05,
      subsample = 0.8,
      colsample_bytree = 0.8,
      lambda = 2, 
      alpha = 0.5
    )
    
    # 4. TRAIN REGRESSION (SG Total)
    model_reg <- xgb.train(
      params = modifyList(params_base, list(objective = "reg:squarederror")),
      data = dtrain_reg,
      nrounds = 500,
      watchlist = list(train = dtrain_reg, eval = dtest_reg),
      early_stopping_rounds = 30,
      print_every_n = 0
    )
    
    #   Capture Regression Stats
    model_attrs <- xgb.attributes(model_reg)
    best_rmse <- as.numeric(model_attrs$best_score)
    best_iter_reg <- as.numeric(model_attrs$best_iteration)
    
    # 5. TRAIN CLASSIFIER
    best_auc <- NA
    best_iter_clf <- NA
    
    #   Ensure both Test Classes and Train Classes Exist
    if (length(unique(y_clf_test)) < 2) {
      message("  - Skipping Classifier for ", curr_course, ": One or more sets lacks class diversity (Need both Top 20 and Non-Top 20).")
      model_clf <- NA
    } else {
      model_clf <- xgb.train(
        params = modifyList(params_base, list(objective = "binary:logistic", eval_metric = "auc", base_score = 0.5)),
        data = dtrain_clf,
        nrounds = 500,
        watchlist = list(train = dtrain_clf, eval = dtest_clf),
        early_stopping_rounds = 30,
        print_every_n = 0
      )
      model_attrs <- xgb.attributes(model_clf)
      best_auc <- model_attrs$best_score
      best_iter_clf <- model_attrs$best_iteration
    }
    
    # 6. UPDATE LEDGER
    course_stats <- data.frame(
      course = as.character(curr_course),
      n_train = nrow(training_df),
      n_val = nrow(test_df),
      val_rmse = as.numeric(unlist(best_rmse)),
      reg_iter = as.numeric(unlist(best_iter_reg)),
      val_auc = if(is.null(best_auc)) NA else as.numeric(unlist(best_auc)),
      clf_iter = if(is.null(best_iter_clf)) NA else as.numeric(unlist(best_iter_clf)),
      stringsAsFactors = FALSE
    )
    training_ledger <- dplyr::bind_rows(training_ledger, course_stats)
    
    # 7. FEATURE IMPORTANCE & SHAP - REGRESSION
    # --> We use the regression model to understand "Course Fit"
    
    # ===== shap_values =====
      # The contribution of each feature to the prediction for every individual player
      # Ex: For a player, if model predicts they will gain +1.5 strokes, SHAP tells you
      #     how many of those strokes came from each predictor
      # 
      # Contains $shap_score and $mean_shap_score for all variables:
        # n_rounds_history | sg{Putt/Arg/App}_l50 | dr{Dist/Acc}_l50
    shap_values <- shap.values(xgb_model = model_reg, X_train = X_train)
    
    # ===== shap_long =====
      # Pairs each SHAP value with its original 'raw feature value' (rfvalue)
      #
      # Produces Various Values Per Variable/Player Combo
        # ID | variable | value | rfvalue | stdfvalue | mean_value
    shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = X_train)
    
    # ===== shap_agg - compute mean_abs_shap =====
      # Computes mean_abs_shap, which measures the total impact of the variable,
      # regardless of if it is positive or negative. Measurement of 'how much does
      # this stat matter at this course'.
      #
      # Computes, for each variable, the mean shap value, and mean absolute shap value
        # variable | mean_abs_shap | mean_shap
    shap_agg <- shap_long %>% 
      group_by(variable) %>% 
      summarise(
        mean_abs_shap = mean(abs(value)),
        mean_shap = mean(value)
      )
    
    # Grab Model Variables
      # n_rounds_history | sg{Putt/Arg/App}_l50 | dr{Dist/Acc}_l50
    vars <- shap_agg$variable
    
    # ===== direction =====
      # Verifies general direction of how a certain stat impacts sgTot. In general, these
      # should all be positive, higher values of each predictor typically means
      # better overall performance.
    direction <- sapply(vars, function(v) {
      col_name <- as.character(v)
      
      predictor_values <- training_df[[col_name]]
      
      cor(predictor_values, training_df$sgTot, use = "complete.obs")
    })
    
    # ===== shap_agg - directional_importance & dir_rel =====
      # Combines the magnitude (how much the stat matters) with the direction.
      #   dir_rel: Scales everything relative to the most important feature
      #   mean_abs_shap_rel: Represents % contribution of each feature to success at course
      #
      # variable | mean_abs_shap | mean_shap | direction | directional_importance | dir_rel
    shap_agg <- shap_agg %>% 
      mutate(
        direction = direction,
        directional_importance = mean_abs_shap * sign(direction),
        dir_rel = directional_importance / max(abs(directional_importance), na.rm = TRUE),
        mean_abs_shap_rel = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE)
      )
    
    # ===== Pivots DataFrame - Results in mean_abs_shap_rel per variable =====
    #   drDist_l50 | drAcc_l50 | sgArg_l50 | sgApp_l50 | sgPutt_l50 | n_rounds_history
    shap_agg_mean <- shap_agg %>% 
      select(variable, mean_abs_shap_rel) %>% 
      arrange(desc(mean_abs_shap_rel)) %>% 
      pivot_wider(
        names_from = variable,
        values_from = mean_abs_shap_rel
      )
    
    # 8. SHAP FOR CLASSIFIER (Top 20 Probability)
    shap_agg_clf_mean <- NA
    
    if(!is.na(model_clf)[1]) {
      # Get SHAP Values for classification
      shap_values_clf <- shap.values(xgb_model = model_clf, X_train = X_train)
      shap_long_clf <- shap.prep(shap_contrib = shap_values_clf$shap_score, X_train = X_train)
      
      # Aggregate and Pivot
      shap_agg_clf_mean <- shap_long_clf %>% 
        group_by(variable) %>% 
        summarise(mean_abs_shap = mean(abs(value))) %>% 
        mutate(mean_abs_shap_rel = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE)) %>% 
        select(variable, mean_abs_shap_rel) %>% 
        arrange(desc(mean_abs_shap_rel)) %>% 
        pivot_wider(
          names_from = variable,
          values_from = mean_abs_shap_rel
        )
    }
    
    # Save results
    course_models[[curr_course]] <- list(
      model_reg = model_reg, # $model in courseOverview
      model_clf = model_clf, # $model in courseOverview
      features = X_vars,
      shap_agg_reg = shap_agg_mean,
      shap_agg_clf = shap_agg_clf_mean
    )
  }
  
  # val_rmse: How Far off, on average, models predictinos are from actual SG Tot in 20% validation data
  View(training_ledger)
  
  return(course_models)
}








