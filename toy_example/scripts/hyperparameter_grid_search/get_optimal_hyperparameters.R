library(stringr)
library(caret)
library(pROC)
library(mltools)

path = '/home/hunter/KAN_toy_example/data/hyperparameter_search_results/'

get_BNN_params = function(name){
  num_epochs = str_match(name, '_num_epochs_\\s*(.*?)\\s*_lr_')[,2]
  lr = str_match(name, '_lr_\\s*(.*?)\\s*_n_neurons_')[,2]
  n_neurons = str_match(name, '_n_neurons_\\s*(.*?)\\s*_act_func_')[,2]
  act_func = str_match(name, '_act_func_\\s*(.*?)\\s*_kl_weight_')[,2]
  kl_weight = str_match(name, '_kl_weight_\\s*(.*?)\\s*.csv')[,2]
  return(list(type='BNN', num_epochs=num_epochs, lr=lr, n_neurons=n_neurons, act_func=act_func, kl_weight=kl_weight))
}

get_CNN_params = function(name){
  num_epochs = str_match(name, '_num_epochs_\\s*(.*?)\\s*_act_func_')[,2]
  act_func = str_match(name, '_act_func_\\s*(.*?)\\s*_lr_')[,2]
  lr = str_match(name, '_lr_\\s*(.*?)\\s*_n_neurons1_')[,2]
  n_neurons1 = str_match(name, '_n_neurons1_\\s*(.*?)\\s*_n_neurons2_')[,2]
  n_neurons2 = str_match(name, '_n_neurons2_\\s*(.*?)\\s*_pool_size_')[,2]
  pool_size = str_match(name, '_pool_size_\\s*(.*?)\\s*_dropout_rate_')[,2]
  kernel_size = str_match(name, '_kernel_size_\\s*(.*?)\\s*.csv')[,2]
  return(list(type='CNN', num_epochs=num_epochs, act_func=act_func, lr=lr, n_neurons1=n_neurons1, n_neurons2=n_neurons2, pool_size=pool_size, kernel_size=kernel_size))
}

get_FNN_params = function(name){
  num_epochs = str_match(name, '_num_epochs_\\s*(.*?)\\s*_act_func_')[,2]
  act_func = str_match(name, '_act_func_\\s*(.*?)\\s*_lr_')[,2]
  lr = str_match(name, '_lr_\\s*(.*?)\\s*_n_neurons_')[,2]
  n_neurons = str_match(name, '_n_neurons_\\s*(.*?)\\s*.csv')[,2]
  return(list(type='FNN', num_epochs=num_epochs, act_func=act_func, lr=lr, n_neurons=n_neurons))
}

get_KAN_params = function(name){
  num_epochs = str_match(name, '_num_epochs_\\s*(.*?)\\s*_grid_')[,2]
  grid = str_match(name, '_grid_\\s*(.*?)\\s*_k_')[,2]
  k = str_match(name, '_k_\\s*(.*?)\\s*_n_neurons_')[,2]
  n_neurons = str_match(name, '_n_neurons_\\s*(.*?)\\s*.csv')[,2]
  return(list(type='KAN', num_epochs=num_epochs, grid=grid, k=k, n_neurons=n_neurons))
}

get_SNN_params = function(name){
  num_epochs = str_match(name, '_num_epochs_\\s*(.*?)\\s*_beta_')[,2]
  beta = str_match(name, '_beta_\\s*(.*?)\\s*_num_steps_')[,2]
  num_steps = str_match(name, '_num_steps_\\s*(.*?)\\s*_n_neurons_')[,2]
  n_neurons = str_match(name, '_n_neurons_\\s*(.*?)\\s*_correct_rate_')[,2]
  correct_rate = str_match(name, '_correct_rate_\\s*(.*?)\\s*_lr_')[,2]
  lr = str_match(name, '_lr_\\s*(.*?)\\s*.csv')[,2]
  return(list(type='SNN', num_epochs=num_epochs, beta=beta, num_steps=num_steps, n_neurons=n_neurons, correct_rate=correct_rate, lr=lr))
}


get.bnn.opt.param.df = function(opt.params){
  tmp = strsplit(opt.params,'_')
  num_epochs = c()
  lrs = c()
  n_neurons = c()
  act_funcs = c()
  kl_weights = c()
  for (k in 1:length(tmp)){
	params.tmp = tmp[[k]]
	num_epochs = c(num_epochs,params.tmp[2])
	lrs = c(lrs,params.tmp[3])
	n_neurons = c(n_neurons,params.tmp[4])
	act_funcs = c(act_funcs,params.tmp[5])
	kl_weights = c(kl_weights,params.tmp[6])
  }
  df.out = data.frame(NUM.EPOCHS=num_epochs, LR=lrs, N.NEURONS=n_neurons, ACT.FUNC=act_funcs, KL.WEIGHT=kl_weights)
  return(df.out)
}

get.cnn.opt.param.df = function(opt.params){
  tmp = strsplit(opt.params,'_')
  num_epochs = c()
  act_funcs = c()
  lrs = c()
  n_neurons1 = c()
  n_neurons2 = c()
  pool_sizes = c()
  kernel_sizes = c()
  for (k in 1:length(tmp)){
	params.tmp = tmp[[k]]
	num_epochs = c(num_epochs,params.tmp[2])
	act_funcs = c(act_funcs,params.tmp[3])
	lrs = c(lrs,params.tmp[4])
	n_neurons1 = c(n_neurons1,params.tmp[5])
	n_neurons2 = c(n_neurons2,params.tmp[6])
	pool_sizes = c(pool_sizes,params.tmp[7])
	kernel_sizes = c(kernel_sizes,params.tmp[8])
  }
  df.out = data.frame(NUM.EPOCHS=num_epochs, ACT.FUNC=act_funcs, LR=lrs, N.NEURONS1=n_neurons1, N.NEURONS2=n_neurons2, POOL.SIZE=pool_sizes, KERNEL.SIZE=kernel_sizes)
  return(df.out)
}

get.fnn.opt.param.df = function(opt.params){
  tmp = strsplit(opt.params,'_')
  num_epochs = c()
  act_funcs = c()
  lrs = c()
  n_neurons = c()
  for (k in 1:length(tmp)){
	params.tmp = tmp[[k]]
	num_epochs = c(num_epochs,params.tmp[2])
	act_funcs = c(lrs,params.tmp[3])
	lrs = c(lrs,params.tmp[4])
	n_neurons = c(n_neurons,params.tmp[5])
  }
  df.out = data.frame(NUM.EPOCHS=num_epochs, ACT.FUNC=act_funcs, LR=lrs, N.NEURONS=n_neurons)
  return(df.out)
}

get.kan.opt.param.df = function(opt.params){
  tmp = strsplit(opt.params,'_')
  num_epochs = c()
  grids = c()
  Ks = c()
  n_neurons = c()
  for (k in 1:length(tmp)){
    params.tmp = tmp[[k]]
	num_epochs = c(num_epochs,params.tmp[2])
    grids = c(grids,params.tmp[3])
    Ks = c(Ks,params.tmp[4])
    n_neurons = c(n_neurons,params.tmp[5])
  }
  df.out = data.frame(NUM.EPOCHS=num_epochs, GRID=grids, K=Ks, N.NEURONS=n_neurons)
  return(df.out)
}

get.snn.opt.param.df = function(opt.params){
  tmp = strsplit(opt.params,'_')
  num_epochs = c()
  betas = c()
  num_steps = c()
  n_neurons = c()
  correct_rates = c()
  lrs = c()
  for (k in 1:length(tmp)){
	params.tmp = tmp[[k]]
	num_epochs = c(num_epochs,params.tmp[2])
	betas = c(betas,params.tmp[3])
	num_steps = c(num_steps,params.tmp[4])
	n_neurons = c(n_neurons,params.tmp[5])
	correct_rates = c(correct_rates,params.tmp[6])
	lrs = c(lrs,params.tmp[7])
  }
  df.out = data.frame(NUM.EPOCHS=num_epochs, BETA=betas, NUM.STEPS=num_steps, N.NEURONS=n_neurons, CORRECT.RATE=correct_rates, LR=lrs)
  return(df.out)
}


fs = list.files(path, full.names=T)
fs = fs[which(grepl('opt_params',fs)==F)]

params = c()
eval_stats = c()
for (i in 1:length(fs)){
  tmp = read.csv(fs[i])
  tmp$PRED = factor(tmp$PRED, levels=c(0,1))
  tmp$TRUE. = factor(tmp$TRUE., levels=c(0,1))
  
  name.tmp = gsub(path,'',fs[i])
  name.tmp = gsub('/','',name.tmp)
  if (substr(name.tmp,1,3) == 'BNN'){params.tmp = get_BNN_params(name.tmp)}
  if (substr(name.tmp,1,3) == 'CNN'){params.tmp = get_CNN_params(name.tmp)}
  if (substr(name.tmp,1,3) == 'FNN'){params.tmp = get_FNN_params(name.tmp)}
  if (substr(name.tmp,1,3) == 'KAN'){params.tmp = get_KAN_params(name.tmp)}
  if (substr(name.tmp,1,3) == 'SNN'){params.tmp = get_SNN_params(name.tmp)}
  params = c(params, paste(params.tmp,collapse='_'))

  eval.stats.tmp = c()
  for (k in 1:length(unique(tmp$ITERATION))){
    idxs.tmp = which(tmp$ITERATION == unique(tmp$ITERATION)[k])
    cm = confusionMatrix(data=tmp$PRED[idxs.tmp], reference=tmp$TRUE.[idxs.tmp])

    if (sum(is.na(tmp$PROB.CASE[idxs.tmp]))==0){
      roc_obj = roc(response=tmp$TRUE.[idxs.tmp], predictor=tmp$PROB.CASE[idxs.tmp], levels=c(0,1), direction='auto', quiet=T)
      auc.ci.tmp = ci.auc(roc_obj)
      eval.stats.tmp = rbind(eval.stats.tmp, c(cm$byClass,AUC=auc(roc_obj),AUC.CI.SIZE=auc.ci.tmp[3]-auc.ci.tmp[1]))
    }
  }
  eval_stats = rbind(eval_stats, apply(eval.stats.tmp,2,mean))
}

df = data.frame(eval_stats)
df = cbind(data.frame(PARAMS=params),df)

df.bnn = df[which(grepl('BNN',df$PARAMS)==T),]
df.cnn = df[which(grepl('CNN',df$PARAMS)==T),]
df.fnn = df[which(grepl('FNN',df$PARAMS)==T),]
df.kan = df[which(grepl('KAN',df$PARAMS)==T),]
df.snn = df[which(grepl('SNN',df$PARAMS)==T),]

df.bnn = df.bnn[which(df.bnn$AUC == max(df.bnn$AUC)),]
df.cnn = df.cnn[which(df.cnn$AUC == max(df.cnn$AUC)),]
df.fnn = df.fnn[which(df.fnn$AUC == max(df.fnn$AUC)),]
df.kan = df.kan[which(df.kan$AUC == max(df.kan$AUC)),]
df.snn = df.snn[which(df.snn$AUC == max(df.snn$AUC)),]

df.bnn = df.bnn[which(df.bnn$AUC.CI.SIZE == min(df.bnn$AUC.CI.SIZE)),]
df.cnn = df.cnn[which(df.cnn$AUC.CI.SIZE == min(df.cnn$AUC.CI.SIZE)),]
df.fnn = df.fnn[which(df.fnn$AUC.CI.SIZE == min(df.fnn$AUC.CI.SIZE)),]
df.kan = df.kan[which(df.kan$AUC.CI.SIZE == min(df.kan$AUC.CI.SIZE)),]
df.snn = df.snn[which(df.snn$AUC.CI.SIZE == min(df.snn$AUC.CI.SIZE)),]

bnn.opt.params = df.bnn$PARAMS[which(df.bnn$AUC == max(df.bnn$AUC))]
cnn.opt.params = df.cnn$PARAMS[which(df.cnn$AUC == max(df.cnn$AUC))]
fnn.opt.params = df.fnn$PARAMS[which(df.fnn$AUC == max(df.fnn$AUC))]
kan.opt.params = df.kan$PARAMS[which(df.kan$AUC == max(df.kan$AUC))]
snn.opt.params = df.snn$PARAMS[which(df.snn$AUC == max(df.snn$AUC))]

df.bnn.opt.params = get.bnn.opt.param.df(bnn.opt.params)
df.cnn.opt.params = get.cnn.opt.param.df(cnn.opt.params)
df.fnn.opt.params = get.fnn.opt.param.df(fnn.opt.params)
df.kan.opt.params = get.kan.opt.param.df(kan.opt.params)
df.snn.opt.params = get.snn.opt.param.df(snn.opt.params)

print(df.bnn.opt.params)
print(df.cnn.opt.params)
print(df.fnn.opt.params)
print(df.kan.opt.params)
print(df.snn.opt.params)

write.csv(df.bnn.opt.params, paste0('/home/hunter/KAN_toy_example/data/hyperparameter_search_results/BNN_opt_params.csv'), row.names=F)
write.csv(df.cnn.opt.params, paste0('/home/hunter/KAN_toy_example/data/hyperparameter_search_results/CNN_opt_params.csv'), row.names=F)
write.csv(df.fnn.opt.params, paste0('/home/hunter/KAN_toy_example/data/hyperparameter_search_results/FNN_opt_params.csv'), row.names=F)
write.csv(df.kan.opt.params, paste0('/home/hunter/KAN_toy_example/data/hyperparameter_search_results/KAN_opt_params.csv'), row.names=F)
write.csv(df.snn.opt.params, paste0('/home/hunter/KAN_toy_example/data/hyperparameter_search_results/SNN_opt_params.csv'), row.names=F)

