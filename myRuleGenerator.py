# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from functools import reduce
import sklearn.metrics as sm
import matplotlib.pyplot as plt    
import itertools  
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning




    
class RuleCondition():
    def __init__(self, feature_index,threshold, operator,support, feature_name = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.feature_name = feature_name
        self.support = support
        
        
    def conditionString(self):
        return ("{} {} {}".format(self.feature_name, self.operator, self.threshold))
    
    #verifica se a condiçao é satisfeita nas regras 1 = sim, 0 = nao
    def transform(self, X):
        if self.operator == "<=":
            res =  1 * (X[:,self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:,self.feature_index] > self.threshold)
        return res
    
    def get_label(self, df):
        
        if self.operator == "<=":
            res =  1 * (df[self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (df[self.feature_index] > self.threshold)
        return res
    
class Rule():
    def __init__(self,rule_conditions,prediction_value):
        self.conditions = set(rule_conditions)
        self.prediction = prediction_value
        self.support = min([x.support for x in rule_conditions])
        
    def ruleString(self):
        return  " & ".join([x.conditionString() for x in self.conditions])
    
    def predictionValue(self):
        return self.prediction
    
    #verifica todas as condicoes da regra
    def transform(self,X):
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x,y: x * y, rule_applies) # multiplica o valor de tds as condicoes, se 1 for 0, esta regra vai ser 0
   
    #Clean redundant conditions in rules
    def cleanRule(self):
        conditions_list = list(self.conditions)
        for c, d in itertools.combinations(conditions_list, 2):
            if(c.feature_name == d.feature_name and c.operator == d.operator):
                if(c.operator == "<=" and c.threshold >= d.threshold):
                    if(c in conditions_list):
                        conditions_list.remove(c)
                elif(c.operator == ">" and c.threshold >= d.threshold):
                    if(d in conditions_list):
                        conditions_list.remove(d)

        self.conditions = set(conditions_list)                 
        return self


def extractRules(tree, feature_names):
    rules = set()
    
    def traverse_Nodes(node_id=0,operator=None, threshold=None, feature=None, conditions=[]):
        
        if node_id != 0:
            #is not root
            feature_name_ = feature_names[feature]
            ruleCondition = RuleCondition(feature, threshold, operator, 
                                          support = tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                                          feature_name=feature_name_)
            new_conditions = conditions + [ruleCondition]
        else:
            #is root
            new_conditions = []
    
        #if not terminal node
        if tree.children_left[node_id]!= tree.children_right[node_id]:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            left_node_id = tree.children_left[node_id]
            traverse_Nodes(left_node_id, "<=", threshold, feature, new_conditions)

            right_node_id = tree.children_right[node_id]
            traverse_Nodes(right_node_id, ">", threshold, feature, new_conditions)
            
        else:
            if len(new_conditions)>0:
                new_Rule = Rule(new_conditions,tree.value[node_id][0][0])
                rules.add(new_Rule)
            return None
        
    traverse_Nodes()
    
    
    return rules



def winsorization(X, trim_quantile=0.025):
    winsor_lims=np.ones([2,X.shape[1]])*np.inf
    winsor_lims[0,:]=-np.inf
    if trim_quantile>0:
        for i_col in np.arange(X.shape[1]):
            lower=np.percentile(X[:,i_col],trim_quantile*100)
            upper=np.percentile(X[:,i_col],100-trim_quantile*100)
            winsor_lims[:,i_col]=[lower,upper]
    winsorized_X =X.copy()
    winsorized_X =np.where(X>winsor_lims[1,:],np.tile(winsor_lims[1,:],[X.shape[0],1]),np.where(X<winsor_lims[0,:],np.tile(winsor_lims[0,:],[X.shape[0],1]),X))
    return winsorized_X


def check_trees(model):
    print('s ', model.estimators_.shape)
    
    #check trees
    n_classes, n_estimators = model.estimators_.shape
    for c in range(n_classes):
        for t in range(n_estimators):
            dtree = model.estimators_[c, t]
            print("class={}, tree={}: {}".format(c, t, dtree.tree_))

            rules = pd.DataFrame({
                'child_left': dtree.tree_.children_left,
                'child_right': dtree.tree_.children_right,
                'feature': dtree.tree_.feature,
                'threshold': dtree.tree_.threshold,
                })
            print(rules)
            
#%% Generate Rules
def genRules(X,Y, feature_names=None, random_state=None, max_rules=2000, tree_size=4,trim_quantile=0.025):
    
    #set of rules
    rules = set()
    
    #initialise
    number_trees = int(np.ceil(max_rules/tree_size))
    N = X.shape[0]
    sub = min(0.5,(100+6*np.sqrt(N))/N)
    model = GradientBoostingRegressor(learning_rate=0.01, n_estimators=number_trees, subsample=sub, max_depth=100,
                                              random_state=1, max_features=None, max_leaf_nodes= tree_size)
    
    
    # simple fit with constant tree size 
    #model.fit(X,Y)
    #fit with Random tree size like RuleFit
    np.random.seed(random_state)
    tree_sizes=np.random.exponential(scale=tree_size-2,size=int(np.ceil(max_rules*2/tree_size)))
    tree_sizes=np.asarray([2+np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))],dtype=int)
    i=int(len(tree_sizes)/4)
    while np.sum(tree_sizes[0:i])<max_rules:
        i=i+1
    tree_sizes=tree_sizes[0:i] # array with the size of each tree
    model.set_params(warm_start=True)
    curr_est_=0
    for i_size in np.arange(len(tree_sizes)):
        size=tree_sizes[i_size] 
        model.set_params(n_estimators=curr_est_+1)
        model.set_params(max_leaf_nodes=size) #set number of rules of that tree
        random_state_add = random_state if random_state else 0
        model.set_params(random_state=i_size+random_state_add) # warm_state=True seems to reset random_state, such that the trees are highly correlated, unless we manually change the random_sate here.
        model.get_params()['n_estimators']  # number of the tree 
        model.fit(np.copy(X, order='C'), np.copy(Y, order='C'))  #np.copy to guarantee data isn't changed
        curr_est_=curr_est_+1 #next tree
    model.set_params(warm_start=False)
    
       
    #all trees (estimators)
    trees_list = model.estimators_
    
    if feature_names is None:
        feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
    else:
        feature_names=feature_names   
            

    #extract rules from tree        
    for tree in trees_list:
      new_rule = extractRules(tree[0].tree_,feature_names=feature_names)
      rules.update(new_rule)

    
    return rules


#%% Prepare Data
def prepareData(rules, X, trim_quantile=0.025, coefs=None, rulefit=False):
        
    #concatenate features and rules
    rule_list = list(rules)
    i=0
    #começa por verificar se os dados satisfazem as regras, 0 se nao, 1 se sim
    #Se tiver coefs, verifica apenas as regras que interessam, aka as que têm coef!=0
    #check rules conditions on the original features
    if(coefs is None):
        a = [rule.transform(X) for rule in rule_list]
        b = np.array(a)
        c = b.T
        X_rules = c
       # X_rules = np.array([rule.transform(X) for rule in rule_list]).T
    else: #Rules with coef 0 are set to 0 here
        for h in np.arange(len(coefs)):
            if coefs[h]!=0:
                i+=1
        res = np.array([rule_list[i_rule].transform(X) for i_rule in np.arange(len(rule_list)) if coefs[i_rule]!=0]).T #only important rules
        res_=np.zeros([X.shape[0],len(rule_list)]) 
        if i>0:
            res_[:,coefs!=0]=res
        X_rules = res_
    
    winsorized_X = []
    ## WINSORIZATION  #filtrar outliers como o rulefit
    if rulefit:
        winsorized_X = winsorization(X,trim_quantile)
    else:
        winsorized_X = X
            
    ## LINEAR STANDARDISATION
    scale_multipliers=np.ones(X.shape[1])
    stddev = np.std(winsorized_X, axis = 0)
    mean = np.mean(winsorized_X, axis = 0)
    for i_col in np.arange(X.shape[1]):
       num_uniq_vals=len(np.unique(X[:,i_col]))
       if num_uniq_vals>2: # don't scale binary variables which are effectively already rules
           scale_multipliers[i_col]=0.4/(1.0e-12 + np.std(winsorized_X[:,i_col]))
    
    X_stand = X*scale_multipliers
    
    
    ## Compile data
    
    X_concat=np.zeros([X.shape[0],0])
    X_concat = np.concatenate((X_concat,X_stand), axis=1)  # valor de cada feature standardizado em cada X, axis=1 para juntar á frente de cada X(dado)
    if X_rules.shape[0] >0:
        X_concat = np.concatenate((X_concat, X_rules), axis=1)     #juntar com o valor de tds as regras em cada X

    return X_concat, i, stddev,mean, scale_multipliers



#%%Regressors
## LASSO 
@ignore_warnings(category=ConvergenceWarning)
def LassoCrV(rules, X, trim_quantile, y, rf=False):
    
    #prepare data to train
    X_concat,_,_,_,_ = prepareData(rules, X, trim_quantile = trim_quantile,rulefit=rf)
    
    #Start Lasso
    lscv = LassoCV(n_alphas=100, alphas=None, cv=3,
                max_iter= 1000, tol=0.0001, n_jobs=None, random_state=1)
    
    lscv.fit(X_concat, y)
        
    return lscv

def LassoN(rules, X, trim_quantile, y, alphaL,rf=False):
    
    
    #prepare data to train
    X_concat,length,stddev,mean,scale_multipliers = prepareData(rules, X, trim_quantile = trim_quantile,rulefit=rf)
    
    #Start Lasso
    ls = Lasso(alpha=alphaL, tol=0.0005, max_iter=5000)
    
    ls.fit(X_concat, y)
    
        
    
    return ls,stddev,mean, scale_multipliers


#%% Get Results

def GetPredictions(regr, X, rules, trim_quantile=0.025, rf=False):
    coefs = regr.coef_
    rule_coefs = coefs[-len(rules):] #nao quero os primeiros que sao as features e nao regras, apenas quero os coefs das regras
    #use the coefs to filter the rules with 0
    X_real, length,_,_,_ = prepareData(rules, X, trim_quantile=trim_quantile, coefs=rule_coefs, rulefit=rf)
    predicted_y = regr.predict(X_real)
    
    return predicted_y, length
    

#Clean redundant conditions in rules
def cleanRules(rules):
    clean_Rules = set()
    for rule in rules:
        clean_Rules.add(rule.cleanRule())
        
    return clean_Rules

#obter regras
def get_rules(coefs, rules, feature_names, stddev, mean, scale_multipliers, exclude_zero_coef=False, subregion=None):
        """
        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """

        n_features= len(coefs) - len(rules)
        rule_ensemble = list(rules)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if True:
                coef=coefs[i]*scale_multipliers[i]
            else:
                coef=coefs[i]
            if subregion is None:
                importance = abs(coef)*stddev[i]
            else:
                subregion = np.array(subregion)
                #importance = sum(abs(coef)* abs([ x[i] for x in self.winsorizer.trim(subregion) ] - mean[i]))/len(subregion)
            output_rules += [(feature_names[i], 'linear',coef, 1, importance)]

        ## Add rules
        for i in range(0, len(rules)):
            rule = rule_ensemble[i]
            coef=coefs[i + n_features]

            if subregion is None:
                importance = abs(coef)*(rule.support * (1-rule.support))**(1/2)
            else:
                rkx = rule.transform(subregion)
                importance = sum(abs(coef) * abs(rkx - rule.support))/len(subregion)

            output_rules += [(rule.ruleString(), 'rule', coef,  rule.support, importance)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support", "importance"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules   
    




def Rules_Accuracy_Rulefit_Lasso(train_X = None, train_y= None, test_X = None, test_y = None, features_names=None):
    train_MAE = []
    val_MAE = []
    rules_length_all = []
    old_rules = genRules(train_X, train_y, features_names)
    rules = cleanRules(old_rules)
    
    def NumberOfRules_Accuracy_graph():
        #for i in (0,0.3,0.5,0.7,0.9,1.1): - sns with service column
        # for i in (0,0.3,0.5,0.6,0.7) - sns1 without service column
        #for i in (-0.1,0,0.1,0.2,0.3,0.5): - sns2 without service column
        #for i in (-0.2,0.1,0.3,0.5,0.6,0.68): - sns2 with service column
        #for i in (-0.1,0,0.1,0.2,0.3,0.4): - sns3 without service column
        #for i in (-0.2,0,0.2,0.4,0.8,1): #SNS3 with service column
        for i in (-0.8,-0.6,-0.4,-0.2,-0.1,0): #SNS4 without service column
        #for i in (-0.4,-0.2,0,0.2,0.4,0.6): #SNS4 with service column
            ls, stddev,mean, scale_multipliers = LassoN(rules, train_X, trim_quantile=0.025, y=train_y, alphaL = 10**i,rf=True)
            train_predictions, rules_length = GetPredictions(ls, train_X, rules,rf=True)
            test_predictions, rules_length = GetPredictions(ls, test_X, rules,rf=True)
            
            train_MAE.append(round(sm.mean_absolute_error(train_y, train_predictions), 2))
            val_MAE.append(round(sm.mean_absolute_error(test_y, test_predictions), 2))
            rules_length_all.append(rules_length)
            
            rules_ = get_rules(ls.coef_, rules, features_names, stddev, mean, scale_multipliers)
            rules_ = rules_[-rules_length:]
            rules_ = rules_[rules_.coef != 0].sort_values(by="support",ascending=False)
            
            f = open("rules/rulefit_sns/rulefit_rules_{}.txt".format(i), "a")
            f.write(rules_.to_string())
            f.close()
            #print(rules_)
            
            #i>0 => alpha>1 vai sempre prever o mesmo valor
            #print("Mean absolute error =", round(sm.mean_absolute_error(train_y, train_predictions), 2)) 
            #print("Mean absolute error =", round(sm.mean_absolute_error(test_y, test_predictions), 2)) 
        
    NumberOfRules_Accuracy_graph()  
    print(train_MAE)
    print(val_MAE)
    print(rules_length_all)
       
    
    plt.title("Accuracy by number of rules")
    plt.ylabel("Number of relevant rules")
    plt.xlabel("Mean Absolute Error")
    plt.plot(val_MAE,rules_length_all, 'r')
    plt.plot(train_MAE,rules_length_all, 'r--')
    #plt.xscale('symlog')
    plt.legend(["Validation", "Train"], loc="upper right")
    #plt.savefig("plots_results_sns1/Accuracy_by_Number_Rules_RuleFit.pdf", format="pdf",bbox_inches="tight")
    #plt.savefig("plots_results_sns2/Accuracy_by_Number_Rules_RuleFit.pdf", format="pdf",bbox_inches="tight")
    #plt.savefig("plots_results_sns3/Accuracy_by_Number_Rules_RuleFit.pdf", format="pdf",bbox_inches="tight")
    plt.savefig("plots_results_sns4/Accuracy_by_Number_Rules_RuleFit.pdf", format="pdf",bbox_inches="tight")
    plt.show()

#%%SNS Dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#%%WITH SERVICE COLUMN
# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns1.csv", index_col=0)
# df= df.drop(columns=['Urgency_Type'])
# df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values))
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)

# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns

#Rules_Accuracy_Rulefit_Lasso(train_X, train_y, test_X, test_y , features_names)

#%%Real RuleFit

def myrulefit(train_X, train_y, features_names, test_X, test_y):
    rules = genRules(train_X, train_y, features_names)
    rules = cleanRules(rules)
    ls = LassoCrV(rules,train_X, trim_quantile=0.025, y=train_y, rf=True)
        
    train_predictions, rules_length = GetPredictions(ls, train_X, rules, rf=True)
    test_predictions, rules_length = GetPredictions(ls, test_X, rules, rf=True)
        
    mae_val=round(sm.mean_absolute_error(test_y, test_predictions), 2)
    print("Train Error: " + str(round(sm.mean_absolute_error(train_y, train_predictions), 2)))
    print("Validation error: "+ str(mae_val))
    print(rules_length)
    
#%%Dataset without Service column
# le = preprocessing.LabelEncoder()
# # df = pd.read_csv("sns dataset\sns1_simple.csv", index_col=0)
# # df = pd.read_csv("sns dataset\sns2_simple.csv", index_col=0)
# #df = pd.read_csv("sns dataset\sns2.csv", index_col=0)
# #df = pd.read_csv("sns dataset\sns3_simple.csv", index_col=0)
# #df = pd.read_csv("sns dataset\sns3.csv", index_col=0)
# #df = pd.read_csv("sns dataset\sns4_simple.csv", index_col=0)
# df = pd.read_csv("sns dataset\sns4.csv", index_col=0)
# df= df.drop(columns=['Urgency_Type']) #for SNS with service column
# #df=df.set_index('Emergency_Stage')  # for SNS without service column
# df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values)) #for SNS with service column
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X=train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns

##### Dataset without Service Column and with Yesterday Column

le = preprocessing.LabelEncoder()
#df = pd.read_csv("sns dataset\sns1_simple2.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns2_simple2.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns3_simple2.csv", index_col=0)
df = pd.read_csv("sns dataset\sns4_simple2.csv", index_col=0)
df=df.set_index('Emergency_Stage')  # for SNS without service column
df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
train, test = train_test_split(df, test_size=0.3)
train_y = train.Waiting_Time.values
train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
train_X = train_X.values
test_y = test.Waiting_Time.values
test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
test_X = test_X.values
train = train.drop(columns=["Waiting_Time","People_Waiting"])
features_names = train.columns


#myrulefit(train_X, train_y, features_names, test_X, test_y)
#Rules_Accuracy_Rulefit_Lasso(train_X, train_y, test_X, test_y, features_names)



#%% Boston dataset


# =============================================================================
# reader = pd.read_csv("sns dataset\sns1.csv", index_col=0)
# train, test = train_test_split(reader, test_size=0.2)
# 
# 
# 
# features_names = train.columns
# train_X = train.values
# 
# test_X = test.values
# =============================================================================

#print(y)
#rules = genRules(train_X, train_y, features_names)
#b, length = LassoN(a,X, trim_quantile=0.025, y=y, alph=1)
#ls = LassoCrV(rules,train_X, trim_quantile=0.025, y=train_y)

#train_predictions, rules_length = GetPredictions(ls, train_X, rules)
#test_predictions, rules_length = GetPredictions(ls, test_X, rules)


#print Rules
#for i in a:
#    print(i.ruleString())



