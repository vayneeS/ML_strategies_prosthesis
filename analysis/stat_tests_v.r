library(rcompanion)
library(FSA)
library(dplyr)
library(tidyverse)

print('** ACCURACY TEST')
data_acc_init <- read.csv("accuracies_init.csv", header=TRUE, stringsAsFactors=FALSE)
print(shapiro.test(unlist(data_acc_init[c('accuracy')])))
srh_acc_init <- kruskal.test(accuracy ~ condition, data = data_acc_init) 
print(srh_acc_init)

data_acc_final <- read.csv("accuracies_final.csv", header=TRUE, stringsAsFactors=FALSE)
print(shapiro.test(unlist(data_acc_final[c('accuracy')])))
srh_acc_final <- kruskal.test(accuracy ~ condition, data = data_acc_final) 
print(srh_acc_final)

print('**ACCURACY ALONG TRAINING')
data_acc_along_training <- read.csv("accuracy_along_training.csv", header=TRUE, stringsAsFactors=FALSE)
for (x in 1:20) {
  data_acc_along_training_ <- data_acc_along_training[data_acc_along_training$trial == x, ]
  srh_sept <- kruskal.test(accuracy ~ condition, data = data_acc_along_training_) 
  print(x)
  print(srh_sept)
  posthoc_acc <- dunnTest(accuracy ~ condition, data = data_acc_along_training_) 
  print(posthoc_acc)
}
#df <- subset(df, select = -c(pid))
#df <- df %>% unite("combined",curriculum:trial)

#print(df_grp)
#sub <- filter(df,trial == 1 | trial == 103)
#print(sub)
#phase = c("first","last","first","last","first","last")
#diff <- sub$acc[sub$trial == 103] - sub$acc[sub$trial == 1]
#sub2 <- diff
#sub2$curriculum <- c("GSC","RC","UCC")
print('** LEARNING RATE')
data_lr <- read.csv("lr.csv", header=TRUE, stringsAsFactors=FALSE)
#print(shapiro.test(unlist(data_lr[c('lr')])))
#srh_acc_lr <- kruskal.test(lr ~ condition, data = data_lr) 
#print(srh_acc_lr)
#posthoc_lr <- dunnTest(lr ~ condition, data = data_lr) 
#print(posthoc_lr)
res.aov <- aov(lr ~ condition, data = data_lr)
print(summary(res.aov))
df_nuc <- data_lr[data_lr$condition == 'NUC']
df_nucs <- data_lr[data_lr$condition == 'NUCS']
df_uc <- data_lr[data_lr$condition == 'UC']
ind <- t.test(lr[data_lr$condition == 'NUC' | data_lr$condition == 'NUCS'] ~ condition[data_lr$condition == 'NUC' | data_lr$condition == 'NUCS'], data = data_lr)
print(ind)
ind_2 <- t.test(lr[data_lr$condition == 'UC' | data_lr$condition == 'NUC'] ~ condition[data_lr$condition == 'UC' | data_lr$condition == 'NUC'], data = data_lr)
print(ind_2)
ind_3 <- t.test(lr[data_lr$condition == 'UC' | data_lr$condition == 'NUCS'] ~ condition[data_lr$condition == 'UC' | data_lr$condition == 'NUCS'], data = data_lr)
print(ind_3)
print('** SEPARABILITY')
data_acc <- read.csv("separability_posttest.csv", header=TRUE, stringsAsFactors=FALSE)
print(shapiro.test(unlist(data_acc[c('separability')])))
srh_acc <- kruskal.test(separability ~ curriculum, data = data_acc) 
print(srh_acc)
posthoc_acc <- dunnTest(separability ~ curriculum, data = data_acc) 
print(posthoc_acc)

print(' separability init')
data_acc <- read.csv("separability_init.csv", header=TRUE, stringsAsFactors=FALSE)
srh_acc <- kruskal.test(separability ~ curriculum, data = data_acc) 
print(srh_acc)
posthoc_acc <- dunnTest(separability ~ curriculum, data = data_acc) 
print(posthoc_acc)

data_sep_t <- read.csv("separability_t.csv", header=TRUE, stringsAsFactors=FALSE)
for (x in 1:103) {
    data_sep_t_ <- data_sep_t[data_sep_t$trial == x, ]
    srh_sept <- kruskal.test(separability ~ curriculum, data = data_sep_t_) 
    print(x)
    print(srh_sept)
}


print('** POS-NEG')
data_posneg <- read.csv("pos_neg.csv", header=TRUE, stringsAsFactors=FALSE)

data_pos <- data_posneg[data_posneg$type == 'positive', ]
kw_pos <- kruskal.test(accuracy ~ condition, data = data_pos) 
print(kw_pos)

posthoc_pos <- dunnTest(accuracy ~ condition, data = data_pos) 
print(posthoc_pos)

data_neg <- data_posneg[data_posneg$type == 'negative', ]
kw_neg <- kruskal.test(accuracy ~ condition, data = data_neg) 
print(kw_neg)


