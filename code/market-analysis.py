####################
#
# author: Daniel Laden
# @: dthomasladen@gmail.com
#
####################

import itertools

import numpy as np 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


####################
# Functions

### Part 1 Apriori Algorithm
##time on my own 6 hrs
def TESTApriori(dataset_transaction, k, min_sup):
	Item_set = []
	Freq_set = []
	n = 1

	#count transaction occurances
	for transaction in dataset_transaction:
		for purchase in transaction[1]:
			if purchase[1] in [e[0] for e in Item_set]:
				Item_set
			else:
				Item_set[purchase[1]] = 1


	#remove ones that are bellow min_sup
	for item, count in Item_set.items():
		if count > min_sup:
			Freq_set[item] = count

	while n < k:
		n +=1
		# print("FREQ_SET")
		# print(Freq_set)
		Item_set_n = {}# keep it as a list no tuples
		for transaction in dataset_transaction:
			transaction_groups = list(itertools.combinations(transaction[1], n))
			transaction_groups = [list(e) for e in transaction_groups]

			# print("TRANSACTION GROUP")
			# print(transaction_groups)

			#check if new Item_set is in Freqn-1
			for group in transaction_groups:
				# print("GROUP")
				# print(group)
				in_transaction = True
				group_ID = []
				for item in group:
					# print("ITEM")
					# print(item)
					group_ID.append(item[1])
					
					if not item[1] in Freq_set.keys():
						# print("FAILED")
						# print(Freq_set.keys())
						# print(item)
						in_transaction = False
				if in_transaction:
					if group_ID in Item_set_n.keys():
						Item_set_n[group_ID] += 1
					else:
						Item_set_n[group_ID] = 1


		print(len(Item_set_n))

		Freq_set = {} #clear F
		for item, count in Item_set_n.items():
			if count > min_sup:
				Freq_set[item] = count

		print(len(Freq_set))
		print(Freq_set)

def calculate_confidence(df, a, b):
	confidence = 0



# End of Functions
####################

f = open("transactions_by_dept.csv", 'r')

#['POS Txn', 'Dept', 'ID', 'Sales U']
transactions = []

skip_first = True

for line in f:
	if skip_first:
		skip_first = False
	else:
		line = line.replace("\n", "")
		line = line.split(",")
		transactions.append(line)

##print(transactions[0])

transaction_ID = None
ID_Index = 0
item_group = []
transaction_group = []

for trans in transactions:
	if int(trans[3]) > 0 :
		if not transaction_ID: #start of the transaction log
			transaction_ID = trans[0]
			item_group.append(trans[1]) #Setting them in as tulpa so the data is immutable
		elif transaction_ID == trans[0]: ## Same transaction, another item
			item_group.append(trans[1]) #format ('Dept', 'ID', 'Sales U')
		else: #different transaction, put the item_group in the transaction id and change transaction_ID
			transaction_group.append(item_group)
			ID_Index += 1
			item_group = []

			transaction_ID = trans[0]
			item_group.append(trans[1])
	## not a sale but a return??

transaction_group.append(item_group) #for the last item


#print(transaction_group)

te = TransactionEncoder()
te_ary = te.fit(transaction_group).transform(transaction_group)
df = pd.DataFrame(te_ary, columns=te.columns_)
#print(df)

tenpercent = apriori(df, min_support=0.10, use_colnames=True)

fivepercent =apriori(df, min_support=0.05, use_colnames=True)

onepercent = apriori(df, min_support=0.05, use_colnames=True)

frequent_itemsets = apriori(df, min_support=5/len(transaction_group), use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print(frequent_itemsets[frequent_itemsets['length'] >=3])

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.50)
rules['antecedents_len'] = rules["antecedents"].apply(lambda x: len(x))
rules['consequents_len'] = rules["consequents"].apply(lambda x: len(x))
max_rule = rules[ (rules['antecedents_len'] >= 2) & (rules['consequents_len'] <= 1) & (rules['lift'] > 2.0)]
max_rule = max_rule[['antecedents','consequents', 'confidence', 'lift', 'support']]
print(max_rule)
quit()



quit()


##################################
# Coding Resources
#
# https://www.geeksforgeeks.org/implementing-apriori-algorithm-in-python/
# https://realpython.com/iterate-through-dictionary-python/
# https://www.thecrazyprogrammer.com/2019/09/apriori-algorithm.html
# http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
# http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
#
#
