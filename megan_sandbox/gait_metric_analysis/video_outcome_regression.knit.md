---
title: "Megan Dissertation - Regression Models"
author: "Megan"
date: "2025-02-21"
output: html_document
---




```r
library(readxl)
```

```
## Warning: package 'readxl' was built under R version 4.1.3
```

```r
library(dplyr)
```

```
## Warning: package 'dplyr' was built under R version 4.1.3
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(tidyverse)
```

```
## Warning: package 'tidyverse' was built under R version 4.1.3
```

```
## Warning: package 'tibble' was built under R version 4.1.3
```

```
## Warning: package 'tidyr' was built under R version 4.1.3
```

```
## Warning: package 'readr' was built under R version 4.1.3
```

```
## Warning: package 'purrr' was built under R version 4.1.3
```

```
## Warning: package 'stringr' was built under R version 4.1.3
```

```
## Warning: package 'forcats' was built under R version 4.1.3
```

```
## Warning: package 'lubridate' was built under R version 4.1.3
```

```
## -- Attaching core tidyverse packages ------------------------ tidyverse 2.0.0 --
## v forcats   1.0.0     v readr     2.1.4
## v ggplot2   3.4.4     v stringr   1.5.0
## v lubridate 1.9.2     v tibble    3.2.1
## v purrr     1.0.1     v tidyr     1.3.0
```

```
## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
## i Use the ]8;;http://conflicted.r-lib.org/conflicted package]8;; to force all conflicts to become errors
```


## Load In-Person Video Data   
All videos included in analysis. Videos were included based on if a walking segment could be identified. 
Some participants may have both a fast walk and preferred walking speed video. Some may have just fast walk or just preferred walking speed. 

Preferred walking speed 


```r
# Preferred Walking Speed 
zeno_pws_df <- read.csv('C:/Users/mmccu/Box/MM_Personal/5_Projects/BoveLab/3_Data_and_Code/gait_bw_zeno_home_analysis/004/zv_bw_merged_gait_vertical_PWS_1.csv')
table(zeno_pws_df$task_pose)
```

```
## 
## gait_vertical_PWS_1 
##                 251
```

```r
table(zeno_pws_df$demographic_diagnosis)
```

```
## 
##  HC  MS 
##  36 215
```
Drop healthy controls - no EDSS and T25FW 

```r
zeno_pws_df <- zeno_pws_df[grepl('MS', zeno_pws_df$demographic_diagnosis), ]
table(zeno_pws_df$demographic_diagnosis)
```

```
## 
##  MS 
## 215
```
Missing Data 

```r
# count number of missing variables in preferred walking speed data frame 
sum(is.na(zeno_pws_df$stride_time_mean_sec_pose_zv))
```

```
## [1] 9
```

```r
sum(is.na(zeno_pws_df$mean_cadence_step_per_min_pose_zv))
```

```
## [1] 0
```

```r
sum(is.na(zeno_pws_df$stride_width_mean_cm_pose_zv))
```

```
## [1] 0
```

```r
# stance and all double support measures 
sum(is.na(zeno_pws_df$foot1_gait_cycle_time_mean_pose_zv))
```

```
## [1] 142
```

```r
# ground truth preferred walk zeno mat data 
sum(is.na(zeno_pws_df$PWS_cadencestepsminmean))
```

```
## [1] 3
```

```r
sum(is.na(zeno_pws_df$bingoEHR_EDSS_measure_value))
```

```
## [1] 10
```

```r
sum(is.na(zeno_pws_df$msfcEHR_T25FW.SPEED.AVG))
```

```
## [1] 27
```

```r
# demographics 
sum(is.na(zeno_pws_df$demoEHR_Age))
```

```
## [1] 4
```

```r
sum(is.na(zeno_pws_df$demoEHR_DiseaseDuration)) 
```

```
## [1] 4
```

```r
sum(is.na(zeno_pws_df$clean_race))
```

```
## [1] 0
```

```r
sum(is.na(zeno_pws_df$clean_ethnicity))
```

```
## [1] 0
```

```r
sum(is.na(zeno_pws_df$clean_sex))
```

```
## [1] 0
```

```r
sum(is.na(zeno_pws_df$bingoEHR_DX_MS.DX))
```

```
## [1] 0
```
Number of participants in preferred walk data frame without any missing values in columns used in video analysis 

Question: should I use this clean df? or before each regression, remove relevant missing variables?

```r
zeno_pws_df_clean = zeno_pws_df %>% drop_na(stride_time_mean_sec_pose_zv, 
                                            mean_cadence_step_per_min_pose_zv, 
                                            stride_width_mean_cm_pose_zv,
                                            bingoEHR_EDSS_measure_value,
                                            msfcEHR_T25FW.SPEED.AVG,
                                            demoEHR_Age, 
                                            demoEHR_DiseaseDuration, 
                                            clean_race,
                                            clean_ethnicity,
                                            clean_sex,
                                            bingoEHR_DX_MS.DX)

nrow(zeno_pws_df_clean)
```

```
## [1] 177
```

```r
# with stance and double support measures 
zeno_pws_df_clean_2 <- zeno_pws_df_clean %>% drop_na(foot1_gait_cycle_time_mean_pose_zv)
nrow(zeno_pws_df_clean_2)
```

```
## [1] 62
```

Fast walking speed videos  


```r
zeno_fw_df <- read.csv('C:/Users/mmccu/Box/MM_Personal/5_Projects/BoveLab/3_Data_and_Code/gait_bw_zeno_home_analysis/004/zv_bw_merged_gait_vertical_FW_1.csv')
table(zeno_fw_df$task_pose)
```

```
## 
## gait_vertical_FW_1 
##                244
```

```r
table(zeno_fw_df$demographic_diagnosis)
```

```
## 
##  HC  MS 
##  39 205
```

```r
zeno_fw_df <- zeno_fw_df[grepl('MS', zeno_fw_df$demographic_diagnosis), ]
table(zeno_fw_df$demographic_diagnosis)
```

```
## 
##  MS 
## 205
```



```r
# count number of missing variables in fast walking speed data frame 
sum(is.na(zeno_fw_df$stride_time_mean_sec_pose_zv))
```

```
## [1] 9
```

```r
sum(is.na(zeno_fw_df$mean_cadence_step_per_min_pose_zv))
```

```
## [1] 0
```

```r
sum(is.na(zeno_fw_df$stride_width_mean_cm_pose_zv))
```

```
## [1] 1
```

```r
# stance and all double support measures 
sum(is.na(zeno_fw_df$foot1_gait_cycle_time_mean_pose_zv))
```

```
## [1] 164
```

```r
# ground truth preferred walk zeno mat data 
sum(is.na(zeno_fw_df$PWS_cadencestepsminmean))
```

```
## [1] 0
```

```r
sum(is.na(zeno_fw_df$bingoEHR_EDSS_measure_value))
```

```
## [1] 9
```

```r
sum(is.na(zeno_fw_df$msfcEHR_T25FW.SPEED.AVG))
```

```
## [1] 21
```

```r
# demographics 
sum(is.na(zeno_fw_df$demoEHR_Age))
```

```
## [1] 4
```

```r
sum(is.na(zeno_fw_df$demoEHR_DiseaseDuration)) 
```

```
## [1] 4
```

```r
sum(is.na(zeno_fw_df$clean_race))
```

```
## [1] 0
```

```r
sum(is.na(zeno_fw_df$clean_ethnicity))
```

```
## [1] 0
```

```r
sum(is.na(zeno_fw_df$clean_sex))
```

```
## [1] 0
```

```r
sum(is.na(zeno_fw_df$bingoEHR_DX_MS.DX))
```

```
## [1] 0
```
Number of participants in fast walk data frame without any missing values in columns used in video analysis 

```r
zeno_fw_df_clean = zeno_fw_df %>% drop_na(stride_time_mean_sec_pose_zv, 
                                          mean_cadence_step_per_min_pose_zv, 
                                          stride_width_mean_cm_pose_zv,
                                          bingoEHR_EDSS_measure_value,
                                          msfcEHR_T25FW.SPEED.AVG,
                                          demoEHR_Age, 
                                          demoEHR_DiseaseDuration, 
                                          clean_race,
                                          clean_ethnicity,
                                          clean_sex,
                                          bingoEHR_DX_MS.DX)

nrow(zeno_fw_df_clean)
```

```
## [1] 175
```

```r
# with stance and double support measures 
zeno_fw_df_clean_2 <- zeno_fw_df_clean %>% drop_na(foot1_gait_cycle_time_mean_pose_zv)
nrow(zeno_fw_df_clean_2)
```

```
## [1] 34
```

## T25FW - plot and log transform 


```r
# preferred walking speed videos 
zeno_pws_df_clean$t25fw_log <- log(zeno_pws_df_clean$msfcEHR_T25FW.SPEED.AVG)
hist(zeno_pws_df_clean$msfcEHR_T25FW.SPEED.AVG)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-10-1.png" width="672" />

```r
hist(zeno_pws_df_clean$t25fw_log)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-10-2.png" width="672" />

```r
plot(zeno_pws_df_clean$msfcEHR_T25FW.SPEED.AVG, zeno_pws_df_clean$stride_time_mean_sec_pose_zv)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-10-3.png" width="672" />

```r
plot(zeno_pws_df_clean$t25fw_log, zeno_pws_df_clean$stride_time_mean_sec_pose_zv)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-10-4.png" width="672" />

```r
# fast walking speed videos 
zeno_fw_df_clean$t25fw_log <- log(zeno_fw_df_clean$msfcEHR_T25FW.SPEED.AVG)
hist(zeno_fw_df_clean$msfcEHR_T25FW.SPEED.AVG)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-11-1.png" width="672" />

```r
hist(zeno_fw_df_clean$t25fw_log)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-11-2.png" width="672" />

```r
plot(zeno_fw_df_clean$msfcEHR_T25FW.SPEED.AVG, zeno_fw_df_clean$stride_time_mean_sec_pose_zv)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-11-3.png" width="672" />

```r
plot(zeno_fw_df_clean$t25fw_log, zeno_fw_df_clean$stride_time_mean_sec_pose_zv)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-11-4.png" width="672" />

## Walking speed pixel proxy 
Transform - log or square root? which is more linear 

```r
# tbd - need to incorporate into full python pipeline
```


## Linear Regression - T25FW 

### Demographics and Disease Info --> Log T25FW

```r
# Preferred walking speed participants 
pws_model_1 <- lm(t25fw_log ~ demoEHR_Age + 
                    demoEHR_DiseaseDuration + 
                    as.factor(bingoEHR_DX_MS.DX) + 
                    as.factor(clean_race) + 
                    as.factor(clean_sex),
                  data = zeno_pws_df_clean)

summary(pws_model_1)
```

```
## 
## Call:
## lm(formula = t25fw_log ~ demoEHR_Age + demoEHR_DiseaseDuration + 
##     as.factor(bingoEHR_DX_MS.DX) + as.factor(clean_race) + as.factor(clean_sex), 
##     data = zeno_pws_df_clean)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.70111 -0.19445 -0.05461  0.14377  1.04656 
## 
## Coefficients:
##                                                                              Estimate
## (Intercept)                                                                  1.914610
## demoEHR_Age                                                                  0.004956
## demoEHR_DiseaseDuration                                                      0.003306
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)    0.314046
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)   -0.003182
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)  0.233955
## as.factor(clean_race)Asian                                                  -0.473887
## as.factor(clean_race)Black Or African American                              -0.231051
## as.factor(clean_race)Declined                                               -0.663545
## as.factor(clean_race)Other                                                  -0.619734
## as.factor(clean_race)Other Pacific Islander                                 -0.618295
## as.factor(clean_race)White                                                  -0.666153
## as.factor(clean_sex)Male                                                    -0.055921
## as.factor(clean_sex)Non-Binary                                               0.030827
##                                                                             Std. Error
## (Intercept)                                                                   0.317188
## demoEHR_Age                                                                   0.002590
## demoEHR_DiseaseDuration                                                       0.003801
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)     0.207558
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)     0.190610
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)   0.208086
## as.factor(clean_race)Asian                                                    0.240804
## as.factor(clean_race)Black Or African American                                0.239347
## as.factor(clean_race)Declined                                                 0.313576
## as.factor(clean_race)Other                                                    0.240932
## as.factor(clean_race)Other Pacific Islander                                   0.384696
## as.factor(clean_race)White                                                    0.225651
## as.factor(clean_sex)Male                                                      0.056067
## as.factor(clean_sex)Non-Binary                                                0.317157
##                                                                             t value
## (Intercept)                                                                   6.036
## demoEHR_Age                                                                   1.913
## demoEHR_DiseaseDuration                                                       0.870
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)     1.513
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)    -0.017
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)   1.124
## as.factor(clean_race)Asian                                                   -1.968
## as.factor(clean_race)Black Or African American                               -0.965
## as.factor(clean_race)Declined                                                -2.116
## as.factor(clean_race)Other                                                   -2.572
## as.factor(clean_race)Other Pacific Islander                                  -1.607
## as.factor(clean_race)White                                                   -2.952
## as.factor(clean_sex)Male                                                     -0.997
## as.factor(clean_sex)Non-Binary                                                0.097
##                                                                             Pr(>|t|)
## (Intercept)                                                                 1.03e-08
## demoEHR_Age                                                                  0.05744
## demoEHR_DiseaseDuration                                                      0.38565
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)    0.13220
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)    0.98670
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)  0.26253
## as.factor(clean_race)Asian                                                   0.05077
## as.factor(clean_race)Black Or African American                               0.33580
## as.factor(clean_race)Declined                                                0.03586
## as.factor(clean_race)Other                                                   0.01100
## as.factor(clean_race)Other Pacific Islander                                  0.10994
## as.factor(clean_race)White                                                   0.00362
## as.factor(clean_sex)Male                                                     0.32005
## as.factor(clean_sex)Non-Binary                                               0.92269
##                                                                                
## (Intercept)                                                                 ***
## demoEHR_Age                                                                 .  
## demoEHR_DiseaseDuration                                                        
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)      
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)      
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)    
## as.factor(clean_race)Asian                                                  .  
## as.factor(clean_race)Black Or African American                                 
## as.factor(clean_race)Declined                                               *  
## as.factor(clean_race)Other                                                  *  
## as.factor(clean_race)Other Pacific Islander                                    
## as.factor(clean_race)White                                                  ** 
## as.factor(clean_sex)Male                                                       
## as.factor(clean_sex)Non-Binary                                                 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.3134 on 163 degrees of freedom
## Multiple R-squared:  0.2667,	Adjusted R-squared:  0.2083 
## F-statistic: 4.561 on 13 and 163 DF,  p-value: 1.233e-06
```

```r
# check residuals 
hist(resid(pws_model_1))
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-13-1.png" width="672" />

```r
# Fast Walking speed participants 
fw_model_1 <- lm(t25fw_log ~ demoEHR_Age + 
                    demoEHR_DiseaseDuration + 
                    as.factor(bingoEHR_DX_MS.DX) + 
                    as.factor(clean_race) + 
                    as.factor(clean_sex),
                  data = zeno_fw_df_clean)

summary(fw_model_1)
```

```
## 
## Call:
## lm(formula = t25fw_log ~ demoEHR_Age + demoEHR_DiseaseDuration + 
##     as.factor(bingoEHR_DX_MS.DX) + as.factor(clean_race) + as.factor(clean_sex), 
##     data = zeno_fw_df_clean)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.69996 -0.19139 -0.03302  0.15034  1.56756 
## 
## Coefficients:
##                                                                              Estimate
## (Intercept)                                                                  1.718392
## demoEHR_Age                                                                  0.005369
## demoEHR_DiseaseDuration                                                      0.001878
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)    0.570202
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)    0.018272
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)  0.338657
## as.factor(clean_race)Asian                                                  -0.265886
## as.factor(clean_race)Black Or African American                              -0.064184
## as.factor(clean_race)Declined                                               -0.511889
## as.factor(clean_race)Other                                                  -0.541290
## as.factor(clean_race)Other Pacific Islander                                 -0.660435
## as.factor(clean_race)White                                                  -0.519700
## as.factor(clean_sex)Male                                                    -0.080747
## as.factor(clean_sex)Non-Binary                                               0.072414
##                                                                             Std. Error
## (Intercept)                                                                   0.369841
## demoEHR_Age                                                                   0.002915
## demoEHR_DiseaseDuration                                                       0.004303
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)     0.277195
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)     0.263680
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)   0.281364
## as.factor(clean_race)Asian                                                    0.230257
## as.factor(clean_race)Black Or African American                                0.233662
## as.factor(clean_race)Declined                                                 0.296761
## as.factor(clean_race)Other                                                    0.236355
## as.factor(clean_race)Other Pacific Islander                                   0.420180
## as.factor(clean_race)White                                                    0.214978
## as.factor(clean_sex)Male                                                      0.065074
## as.factor(clean_sex)Non-Binary                                                0.263119
##                                                                             t value
## (Intercept)                                                                   4.646
## demoEHR_Age                                                                   1.842
## demoEHR_DiseaseDuration                                                       0.437
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)     2.057
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)     0.069
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)   1.204
## as.factor(clean_race)Asian                                                   -1.155
## as.factor(clean_race)Black Or African American                               -0.275
## as.factor(clean_race)Declined                                                -1.725
## as.factor(clean_race)Other                                                   -2.290
## as.factor(clean_race)Other Pacific Islander                                  -1.572
## as.factor(clean_race)White                                                   -2.417
## as.factor(clean_sex)Male                                                     -1.241
## as.factor(clean_sex)Non-Binary                                                0.275
##                                                                             Pr(>|t|)
## (Intercept)                                                                 6.99e-06
## demoEHR_Age                                                                   0.0673
## demoEHR_DiseaseDuration                                                       0.6630
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)     0.0413
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)     0.9448
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)   0.2305
## as.factor(clean_race)Asian                                                    0.2499
## as.factor(clean_race)Black Or African American                                0.7839
## as.factor(clean_race)Declined                                                 0.0865
## as.factor(clean_race)Other                                                    0.0233
## as.factor(clean_race)Other Pacific Islander                                   0.1180
## as.factor(clean_race)White                                                    0.0167
## as.factor(clean_sex)Male                                                      0.2165
## as.factor(clean_sex)Non-Binary                                                0.7835
##                                                                                
## (Intercept)                                                                 ***
## demoEHR_Age                                                                 .  
## demoEHR_DiseaseDuration                                                        
## as.factor(bingoEHR_DX_MS.DX)PPMS (Primary-progressive Multiple Sclerosis)   *  
## as.factor(bingoEHR_DX_MS.DX)RRMS (Relapsing-remitting Multiple Sclerosis)      
## as.factor(bingoEHR_DX_MS.DX)SPMS (Secondary-progressive Multiple Sclerosis)    
## as.factor(clean_race)Asian                                                     
## as.factor(clean_race)Black Or African American                                 
## as.factor(clean_race)Declined                                               .  
## as.factor(clean_race)Other                                                  *  
## as.factor(clean_race)Other Pacific Islander                                    
## as.factor(clean_race)White                                                  *  
## as.factor(clean_sex)Male                                                       
## as.factor(clean_sex)Non-Binary                                                 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.3629 on 161 degrees of freedom
## Multiple R-squared:  0.3125,	Adjusted R-squared:  0.257 
## F-statistic: 5.628 on 13 and 161 DF,  p-value: 2.032e-08
```

```r
# check residuals 
hist(resid(fw_model_1))
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-14-1.png" width="672" />
###  Each Video Metric  --> Log T25FW 


```r
# preferred walking speed, stride time mean 
plot(zeno_pws_df_clean$t25fw_log, zeno_pws_df_clean$stride_time_mean_sec_pose_zv)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-15-1.png" width="672" />

```r
pws_stride_time_model <- lm(t25fw_log ~ stride_time_mean_sec_pose_zv, 
                            data = zeno_pws_df_clean)

summary(pws_stride_time_model)
```

```
## 
## Call:
## lm(formula = t25fw_log ~ stride_time_mean_sec_pose_zv, data = zeno_pws_df_clean)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.80674 -0.20859 -0.06938  0.15387  1.11214 
## 
## Coefficients:
##                              Estimate Std. Error t value Pr(>|t|)    
## (Intercept)                    0.6396     0.1830   3.494 0.000603 ***
## stride_time_mean_sec_pose_zv   0.8505     0.1599   5.319 3.15e-07 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.3278 on 175 degrees of freedom
## Multiple R-squared:  0.1392,	Adjusted R-squared:  0.1343 
## F-statistic: 28.29 on 1 and 175 DF,  p-value: 3.154e-07
```

```r
hist(resid(pws_stride_time_model))
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-15-2.png" width="672" />

```r
# fast walking speed, stride time mean
plot(zeno_fw_df_clean$t25fw_log, zeno_fw_df_clean$stride_time_mean_sec_pose_zv)
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-15-3.png" width="672" />

```r
fw_stride_time_model <- lm(t25fw_log ~ stride_time_mean_sec_pose_zv, 
                            data = zeno_fw_df_clean)

summary(fw_stride_time_model)
```

```
## 
## Call:
## lm(formula = t25fw_log ~ stride_time_mean_sec_pose_zv, data = zeno_fw_df_clean)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.80202 -0.18462 -0.04216  0.12364  1.95943 
## 
## Coefficients:
##                              Estimate Std. Error t value Pr(>|t|)    
## (Intercept)                    0.2919     0.1376   2.122   0.0353 *  
## stride_time_mean_sec_pose_zv   1.3739     0.1406   9.770   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.339 on 173 degrees of freedom
## Multiple R-squared:  0.3556,	Adjusted R-squared:  0.3518 
## F-statistic: 95.45 on 1 and 173 DF,  p-value: < 2.2e-16
```

```r
hist(resid(fw_stride_time_model))
```

<img src="video_outcome_regression_files/figure-html/unnamed-chunk-15-4.png" width="672" />

```r
# preferred walking speed, stride time median 

# fast walking speed, stride time median
```

## Ordinal Regression --> EDSS

###  Each Video Metric  --> PWS or FWS velocity (check distribution)
