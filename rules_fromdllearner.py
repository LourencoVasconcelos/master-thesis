# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""

import string
import myRuleGenerator as rf
from pyeda.inter import *



def create_expression(conjunctions, aux):
    new_rule = ''  #string to make an expression with same logic as the rule
    i = 0
    for idx,a in enumerate(string.ascii_lowercase):
        if(idx<len(conjunctions)):
            if("(" in aux[idx]):
                new_rule += "("
            new_rule = new_rule+a
            if(")" in aux[idx]):
                new_rule +=")"
            new_rule = new_rule+conjunctions[idx]
            i = idx
        
    if(len(conjunctions)==0):
            new_rule += string.ascii_lowercase[i]
    else:
        new_rule += string.ascii_lowercase[i+1]
        if(")" in aux[i+1]):
            times = aux[i+1].count(")")
            for k in range(times):
                new_rule +=")"
                
    return new_rule

def rule_from_expression(r, aux):
    aux3=[]
    chars = list(r)
    lis = list(string.ascii_lowercase)
    for idx,a in enumerate(chars):
        if a in string.ascii_lowercase:
            aux3.append(aux[lis.index(a)])
        else:
            aux3.append(a)
    newString =''
    for c in aux3:
        newString +=c
    return newString

def extractRules_DLLearner(rules_str):
    
    ex = []
    new_rules = set()
    for rule in rules_str:
        if "not" in rule:
            # Extremely rare and we do not want negative rules, happened once and could not reproduce 
            continue
        rule = rule.replace("|", "")  # Remove as barras
        rule = rule.replace("and", "&")  # Remove as barras
        rule = rule.replace("or", "|")
        expressions = []
        conjunctions = []
        r_split = rule.split(" ")
        if len(r_split)==1:
            continue
        for idx,x in enumerate(r_split):
            if idx % 4 !=3:
                expressions.append(x)
            else:
                expressions.append(" ")
                conjunctions.append(x)
        ex.append(expressions)     #different expressions separated (a,b,c)
        mystring = ''
        for x in expressions:
            mystring += x
        #print(mystring)     
        aux = mystring.split(" ") #get the expressions like day<7 etc... saved separated from eachother
        #print(aux)
        new_rule = create_expression(conjunctions, aux)            
        py_exp = expr(new_rule) #get the expression in pyeda
        py_exp = py_exp.to_dnf() #make it in disjunctive normal form.
        stri=str(py_exp)  #transform expression into a string
        stri = stri.split("And")  #split by ands to get each expression alone
        
        rules_from_this_rule = []
        for idx,s in enumerate(stri):
            if idx !=0:
                rules_from_this_rule.append(s)  #there may be more than 1 rule if it has an or
            elif idx==0 and 'a' in s and len(stri)>1:
                rules_from_this_rule.append('a')
        if(len(stri)<2):
            rules_from_this_rule.append(stri[0])
            
        for r in rules_from_this_rule:
            if 'Or' in r and len(r)<9:
                res = r.split(",")
                for a in res:
                    rules_from_this_rule.append(a)
                rules_from_this_rule.remove(r)
            break
        
        for r in rules_from_this_rule:
            r = r.replace("(","")
            r = r.replace(")","")
            r = r.replace(",","")
            r = r.replace(" ","&")
            r = r.replace("Or","")
            if r[-1] == "&":
                r = r[:-1]
            if r[0] == "&":
                r = r[1:]
                
            rule_string = rule_from_expression(r,aux)
            rule_string = rule_string.replace("(","")
            rule_string = rule_string.replace(")","")
            new_rules.add(rule_string) #add only new elements
    return new_rules

def create_Rules(rules_s,features_names):
    rules = set()
    features = list(features_names)
    for r in rules_s:
        r = r.split("&")
        
        conditions = []
        
        for c in r:
            if ">=" in c:
                aux = c.split(">=")
                ruleCondition = rf.RuleCondition(feature_index=features.index(aux[0]), threshold=float(aux[1]),
                                                 operator=">=",
                                                 support=0, feature_name=aux[0])
            elif "<=" in c:
                aux = c.split("<=")
                ruleCondition = rf.RuleCondition(feature_index=features.index(aux[0]), threshold=float(aux[1]),
                                                 operator="<=",
                                                 support=0, feature_name=aux[0])
            elif "<" in c:
                aux = c.split("<")
                ruleCondition = rf.RuleCondition(feature_index=features.index(aux[0]), threshold=float(aux[1]),
                                                     operator="<",
                                                     support=0, feature_name=aux[0])
            elif ">" in c:
                aux = c.split(">")
                ruleCondition = rf.RuleCondition(feature_index=features.index(aux[0]), threshold=float(aux[1]),
                                                     operator=">",
                                                     support=0, feature_name=aux[0])
            else: #== does this happen? , protect anyways
                aux = c.split("==")
                ruleCondition = rf.RuleCondition(feature_index=features.index(aux[0]), threshold=float(aux[1]),
                                                 operator="==",
                                                 support=0, feature_name=aux[0])
            conditions = conditions + [ruleCondition]
        new_Rule = rf.Rule(conditions, 0)
        rules.add(new_Rule)
    return rules