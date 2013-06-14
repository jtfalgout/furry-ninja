#from sklearn.ensemble import GradientBoostingClassifier
import sklearn.ensemble
from sklearn.metrics import metrics
from numpy import *
import os, sys, time, getopt, shelve, datetime

##########
# this function should be in sklearn.metrics, but isn't for the enthought distribution, WTF?
##########
def auc_score(y_true, y_score):
    """Compute Area Under the Curve (AUC) from prediction scores.

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        true binary labels

    y_score : array, shape = [n_samples]
        target scores, can either be probability estimates of
        the positive class, confidence values, or binary decisions.

    Returns
    -------
    auc : float

    References
    ----------
    http://en.wikipedia.org/wiki/Receiver_operating_characteristic

    See also
    --------
    average_precision_score: Area under the precision-recall curve
    """

    fpr, tpr, tresholds = roc_curve(y_true, y_score)
    return auc(fpr, tpr)



##########
# This python script reads in a CSV file of data, fits a boosted regression tree to it using parameters specified
# on the command line, and then saves the regression tree objects for later use
#
# 1/26/2013 T.J.H.
#
##########
def main(argv):
	##########
	# Command-line arguments
	##########
	# default values for inputs
	region_name = 'East'
	input_file = '/data/landsat/FireECV/furry-ninja/python_mpi/' + region_name + '_small_sample.csv'
	shelf_file = '/data/landsat/FireECV/furry-ninja/python_mpi/' + region_name + '.shelf'
	write_shelf = True
	n_trees = 1000
	tree_depth = 3
	learning_rate = 0.05
	out_file = '/d/workspace/FireECV/RegionalModels/' + region_name + '_diagnostics.txt'

	opts, args = getopt.getopt(argv, "hi:o:",["input=","shelf=","n_trees=","tree_depth=","learning_rate=","output=","help"])

	for opt, arg in opts:
		if opt in ( '--input', '-i' ):
			input_file = arg
			if not os.path.exists(input_file):
				print input_file + ' does not exist!'
				sys.exit()
		if opt in ( '--shelf', '-s' ):
			shelf_file = arg
			write_shelf = True
		if opt in ( '--n_trees', '-n' ):
			n_trees = int(arg)
		if opt in ( '--tree_depth', '-d' ):
			tree_depth = float(arg)
		if opt in ( '--learning_rate', '-l' ):
			learning_rate = float(arg)
		if opt in ( '--help', '-h' ):
			print 'Usage:'
			print '--input=<input_data_file.csv> | -i <input_data_file.csv>'
			print '--shelf=<output_data_file.Shelf> | -s <output_data_file.Shelf> (Output file to hold Shelf objects for import into later python sessions)'
			print '--n_trees=X | -n=X (Number of trees to use in the boosted regression tree)'
			print '--tree_depth=X | -d=X (Depth of trees in boosted regression trees)'
			print '--learning_rate=X | -l=X (Learning rate used to construct boosted regression trees)'

	##########
	# print some feedback before we get to work
	##########
	print '################################################################################'
	print 'Training Boosted Regression Tree for region ' + region_name
	print 'Start time = ' + str(datetime.datetime.now())
	startTime0 = time.time()
	print 'Input data = ' + input_file
	print 'Number of trees = ' + str(n_trees)
	print 'Tree depth = ' + str(tree_depth)
	print 'Learning rate = ' + str(learning_rate)
	print 'Shelf file = ' + shelf_file
	print 'Model diagnostics file = ' + out_file
	print '################################################################################'

	##########
	# Read a .csv file of training data
	##########
	startTime = time.time()
	#my_data = recfromcsv(input_file, delimiter=',', names=True) #, dtype="string")
	my_data = genfromtxt(fname=input_file, delimiter=",", names=True)
	print 'Data read = ' + str(time.time() - startTime) + ' seconds'

	# set ecoregion-specific parameters
	if region_name=='East':
		nDays = 16*5

	# set fire/nofire variable
	#fire = my_data['days_since_fire'] < nDays


	##########
	# split the data into training and validation data
	##########
	train_array = my_data['train'] == True
	test_array = my_data['train'] == False

	# all the data
	predictors = ["sensor","month","band1","band2","band3","band4","band5","band7","ndvi","ndmi","nbr","nbr2","wi_b3","wi_b4","wi_b5","wi_b7","wi_ndvi","wi_ndmi","wi_nbr","wi_nbr2","sp_b3","sp_b4","sp_b5","sp_b7","sp_ndvi","sp_ndmi","sp_nbr","sp_nbr2","su_b3","su_b4","su_b5","su_b7","su_ndvi","su_ndmi","su_nbr","su_nbr2","fa_b3","fa_b4","fa_b5","fa_b7","fa_ndvi","fa_ndmi","fa_nbr","fa_nbr2","ly_wi_b3","ly_wi_b4","ly_wi_b5","ly_wi_b7","ly_wi_ndvi","ly_wi_ndmi","ly_wi_nbr","ly_wi_nbr2","ly_sp_b3","ly_sp_b4","ly_sp_b5","ly_sp_b7","ly_sp_ndvi","ly_sp_ndmi","ly_sp_nbr","ly_sp_nbr2","ly_su_b3","ly_su_b4","ly_su_b5","ly_su_b7","ly_su_ndvi","ly_su_ndmi","ly_su_nbr","ly_su_nbr2","ly_fa_b3","ly_fa_b4","ly_fa_b5","ly_fa_b7","ly_fa_ndvi","ly_fa_ndmi","ly_fa_nbr","ly_fa_nbr2"]

	y_all = my_data['fire']
	X_all = my_data[predictors]

	X_all2 = array(X_all.tolist())

	# Training data
	y_train = my_data['fire'][train_array]
	X_train = X_all2[train_array,:]

	# Validation data
	y_test = my_data['fire'][test_array]
	X_test = X_all2[test_array,:]


	##########	
	# fit the model
	##########
	start_time = time.time()
	clf = sklearn.ensemble.GradientBoostingClassifier(loss='deviance', learn_rate=learning_rate, n_estimators=n_trees, max_depth=tree_depth, subsample=0.5, min_samples_leaf=10, random_state=0)
	clf.fit(X_train, y_train)
	end_time = time.time()
	print "Processing time = " + str(end_time - start_time) + " seconds"

	if False:
		##########
		# assess model fit using independent data
		##########
		print "Score (training data) = " + str(clf.score(X_train, y_train))
		#all_prob = clf.predict_proba(X_train)	# first column is unburned probability, 2nd column is burned probability
		#print "AUC (training data) = " + str(metrics.auc_score(y_train, all_prob[:,1]))
		#print "AUC (training data) = " + str(auc_score(y_train, all_prob[:,1]))
	
		print "Score (validation data) = " + str(clf.score(X_test, y_test))
		#all_prob = clf.predict_proba(X_test)
		#print "AUC (validation data) = " + str(metrics.auc_score(y_test, all_prob[:,1]))
		#print "AUC (validation data) = " + str(auc_score(y_test, all_prob[:,1]))
	
		#y_pred_train = clf.predict(X_train)
		print '################################################################################'
		print 'Classification report for training data:'
		#print(metrics.classification_report(y_train, y_pred_train))

		#y_pred_test = clf.predict(X_test)
		print '################################################################################'
		print 'Classification report for validation data:'
		#print(metrics.classification_report(y_test, y_pred_test))
	
		print '################################################################################'
		var_importance = transpose(array( [ predictors, clf.feature_importances_]))
		for i in range(0, var_importance.shape[0]):
			if i==0:
				print 'Variable, Importance'
			#print var_importance[i][0] + ", \t\t" + str(float(var_importance[i][1]))
	
		##########
		# training and validation fit statistics for each tree
		# staged_decision_function works, but staged_predict and staged_predict_proba don't for some reason!!!
		##########
		#y_pred_train_staged = clf.staged_decision_function(X_train)
		#y_pred_test_staged = clf.staged_decision_function(X_test)
		i = 0
		for yy_test in y_pred_test_staged:
			#pred_proba_test = clf._score_to_proba(yy_test)
			#pred_class_test = clf.classes_.take( argmax(pred_proba_test, axis=1), axis=0)
	
			#yy_train = y_pred_train_staged.next()
			#pred_proba_train = clf._score_to_proba(yy_train)
			#pred_class_train = clf.classes_.take( argmax(pred_proba_train, axis=1), axis=0)
	
			#pred_proba_test = clf._score_to_proba(y_test)
			#pred_class_test = clf.classes_.take( argmax(pred_proba_test, axis=1), axis=0)
	
			if i==0:
				print '################################################################################'
				print "iteration, auc_training, F1_training, loss_training, auc_validation, F1_validation, loss_validation"
	
			#print str(i+1) + ", " + \
				#str(metrics.auc_score(y_train, pred_proba_train[:,1])) + ", " + str(metrics.f1_score(y_train, pred_class_train)) + ", " + str(clf.loss_(y_train, pred_class_train)) + ", " + \
				#str(metrics.auc_score(y_test, pred_proba_test[:,1])) + ", " + str(metrics.f1_score(y_test, pred_class_test)) + ", " + str(clf.loss_(y_test, pred_class_test))
	
			i += 1

		print '################################################################################'

	##########
	# Save python session for later use
	##########
	if write_shelf:
		my_shelf = shelve.open(filename=shelf_file, flag='n')
		my_shelf['input_file'] = input_file
		my_shelf['n_trees'] = n_trees
		my_shelf['learning_rate'] = learning_rate
		my_shelf['tree_depth'] = tree_depth
		my_shelf['clf'] = clf
		my_shelf.close()

	print 'End time = ' + str(datetime.datetime.now())
	endTime0 = time.time()
	print 'Processing time = ' + str(endTime0 - startTime0) + ' seconds'
	print '################################################################################'

##########
# code to recreate saved python session
##########
# filename = "/d/workspace/FireECV/python/GBM_Model.session"
# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()

if __name__ == '__main__':
	main(sys.argv[1:])
