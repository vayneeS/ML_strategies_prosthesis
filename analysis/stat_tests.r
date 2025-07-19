library(rcompanion)
library(FSA)

print('** ACCURACY TEST')
data_acc <- read.csv("accuracies.csv", header=TRUE, stringsAsFactors=FALSE)
print(shapiro.test(unlist(data_acc[c('accuracy')])))
srh_acc <- kruskal.test(accuracy ~ condition, data = data_acc) 
print(srh_acc)
posthoc_acc <- dunnTest(accuracy ~ condition, data = data_acc) 
print(posthoc_acc)


print('** SEPARABILITY')
data_acc <- read.csv("separability_posttest.csv", header=TRUE, stringsAsFactors=FALSE)
srh_acc <- kruskal.test(separability ~ condition, data = data_acc) 
print(srh_acc)
posthoc_acc <- dunnTest(separability ~ condition, data = data_acc) 
print(posthoc_acc)

print(' separability init')
data_acc <- read.csv("separability_init.csv", header=TRUE, stringsAsFactors=FALSE)
srh_acc <- kruskal.test(separability ~ condition, data = data_acc) 
print(srh_acc)
posthoc_acc <- dunnTest(separability ~ condition, data = data_acc) 
print(posthoc_acc)

data_sep_t <- read.csv("separability_t.csv", header=TRUE, stringsAsFactors=FALSE)
for (x in 1:103) {
    data_sep_t_ <- data_sep_t[data_sep_t$trial == x, ]
    srh_sept <- kruskal.test(separability ~ condition, data = data_sep_t_) 
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


