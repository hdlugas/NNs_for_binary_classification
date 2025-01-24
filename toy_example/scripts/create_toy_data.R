
library(caret)
set.seed(1)

df = twoClassSim(n=100, linearVars=100, noiseVars=100, corrVars=10)
df = df[,3:ncol(df)]
colnames(df)[1:(ncol(df)-1)] = paste0('METABOLITE.',1:(ncol(df)-1))
rownames(df) = paste0('SAMPLE.',1:nrow(df))
min.abundance = min(apply(df[,1:(ncol(df)-1)], 2, min))
df[,1:(ncol(df)-1)] = df[,1:(ncol(df)-1)] + abs(min.abundance)
min.abundance = min(apply(df[,1:(ncol(df)-1)], 2, min))

rows.tmp = sample(1:nrow(df), size=10, replace=F)
cols.tmp = sample(1:(ncol(df)-1), size=10, replace=F)
for (i in 1:length(rows.tmp)){
  df[rows.tmp[i],cols.tmp[i]] = NA
}
colSums(is.na(df))

df$Group = ifelse(df$Class=='Class1',0,1)
df$Group = factor(df$Group, levels=c(0,1), labels=c(0,1))
df$Class = df$Group
df = df[,which(colnames(df) != 'Group')]

write.csv(df, '/home/hunter/mass_spec/KAN/toy_example/data/raw_toy_data.csv', row.names=F)

