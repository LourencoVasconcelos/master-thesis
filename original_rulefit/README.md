

# RuleFit
Implementation of a rule based prediction algorithm based on [the rulefit algorithm from Friedman and Popescu (PDF)](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf)

The algorithm can be used for predicting an output vector y given an input matrix X. In the first
step a tree ensemble is generated with gradient boosting. The trees are then used to form
rules, where the paths to each node in each tree form one rule. A rule is a binary decision if
an observation is in a given node, which is dependent on the input features that were used
in the splits. The ensemble of rules together with the original input features are then being input
 in a L1-regularized linear model, also called Lasso, which estimates the effects of each rule on
the output target but at the same time estimating many of those effects to zero.

You can use rulefit for predicting a numeric response (categorial not yet implemented).
The input has to be a numpy matrix with only numeric values.
