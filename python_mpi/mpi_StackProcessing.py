import sys
import os
import time
import shelve

import numpy as np
np.seterr(divide='ignore')

from sklearn.ensemble import GradientBoostingClassifier

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

debug_mode = True

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class Stack:
	##########
	# INPUTS
	##########
	input_dir = None			# input directory to search for input files
	stack_file = None			# filename of the original stack file
	stack = None				# array of information about images in the stack
	
	n_input_datasets = 0			# number of input images
	input_datasets = None			# array of GDAL dataset connections to input files 
	input_band_names = ['band1','band2','band3','band4','band5','band6','band7','QA']	# order/naming of bands in the input image files and the input_bands array
	input_bands = None			# array of GDAL band connections to input files

	##########
	# outputs
	##########
	output_dir = None			# where to save output files
	output_datasets = None			# array of GDAL dataset connections to output files
	output_band_names = ['cburn']		# includes QA info
	output_bands = None			# array of GDAL band connections to output files

	##########
	# MISC IMAGE INFO - for both inputs and outputs
	##########
	nCol = 0
	nRow = 0
	dx = 0
	dy = 0
	west = None
	east = None
	north = None
	south = None
	geotrans = None
	prj = None
	nodata = -9999

	##########
	# these variables control the size of data chunks used when processing a stack
	##########
	#ncol_block = -1		# how many columns of data to read at a time, -1 = all
	#nrow_block = 16		# how many rows to read at a time, -1 = all

	##########
	# classifier object and info
	##########
	clf_shelf_file = None	# shelf file containing an object named 'clf' to use as a classifier
	clf = None		# classifier object / boosted regression tree
	pBurnThreshold = 0.5	# probability threshold to use when converting burned probabilities to binary values


	##########
	# initialize a stack object - load stack from file given as an argument
	##########
	def __init__(self, stack_filename):
		start_time = time.time()
		total_time = -1

		if not os.path.exists(stack_filename):
			print stack_filename + " does not exist!"

		#print "Loading stack from " + stack_filename + "..."
		self.stack_file = stack_filename
		self.stack = np.recfromcsv(stack_filename, delimiter=',', names=True, dtype="string")

		total_time = time.time() - start_time
		#print "Loaded stack from " + stack_filename


	##########
	# processing functions
	##########
	def openClassifier(self, shelf_file):
		start_time = time.time()
		total_time = -1

		if not os.path.exists(shelf_file):
			return( [False, total_time, shelf_file + " does not exist"] )

		#print "Loading shelf objects from " + shelf_file + "..."
		self.clf_shelf_file = shelf_file
		my_shelf = shelve.open(shelf_file)
		self.clf = my_shelf['clf']
		my_shelf.close()

		total_time = time.time() - start_time
		return( [True, total_time, "Loaded classifier from " + shelf_file] )


	##########
	# create connections to input images and bands
	##########
	def openInputDatasets(self):
		start_time = time.time()
		total_time = -1

		self.n_input_datasets = self.stack.shape[0]

		# arrays to hold dataset and band objects
		#print 'Input file count is', self.n_input_datasets

		#print 'Creating arrays for input datasets and bands...'
		self.input_datasets = np.empty((self.n_input_datasets), dtype=object)
		self.input_bands = np.empty((self.n_input_datasets, len(self.input_band_names)), dtype=object)

		for i in range(0, self.n_input_datasets):
			input_image_name = self.stack['file_'][i]

			if not os.path.exists(input_image_name):
				return( [False, total_time, input_image_name + " does not exist"] )

			self.input_datasets[i] = dataset=gdal.Open(input_image_name)

			if self.input_datasets[i] is None:
				return( [False, total_time, "Failed to open dataset " + input_image_name] )

			for j in range(0,len(self.input_band_names)):
				self.input_bands[i,j] = self.input_datasets[i].GetRasterBand(j+1)

				if self.input_bands[i,j] is None:
					return( [False, total_time, "Failed to open band " + str(j) + " of dataset " + input_image_name] )


		# grab image size and projection info
		self.nCol = self.input_datasets[0].RasterXSize
		self.nRow = self.input_datasets[0].RasterYSize

		self.geotrans = self.input_datasets[0].GetGeoTransform()
		self.dx = self.geotrans[1]
		self.dy = self.geotrans[5]
		self.west = self.geotrans[0]
		self.east = self.west + (self.nCol * self.dx)
		self.north = self.geotrans[3]
		self.south = self.north + (self.nRow * self.dy)
		self.prj = self.input_datasets[0].GetProjectionRef()

		total_time = time.time() - start_time
		return( [True, total_time, "All bands opened for " + str(self.n_input_datasets) + " datasets"] )


	##########
	# create connections to output images and bands
	##########

	#def openOutputDatasets(self, create=True):
	def openOutputDatasets(self, create=False):
		start_time = time.time()
		total_time = -1

		# test to make sure the output director exists
		if not os.path.exists(self.output_dir):
			print 'Creating output directory ' + self.output_dir
			os.makedirs(self.output_dir)

		# create arrays to hold connections to output datasets and bands
		self.output_datasets = np.empty((self.n_input_datasets), dtype=object)
		self.output_bands = np.empty((self.n_input_datasets, len(self.output_band_names)), dtype=object)

		if create:
			# create the TIF driver for output data
			driver = gdal.GetDriverByName("GTiff")

			# loop through the files and create output
			for i in range(0, self.n_input_datasets):
				# exclude first year of the stack
				#if self.stack['year'][i] <> np.min(self.stack['year']):
				if True:
					output_image_name = self.output_dir + os.path.basename(self.stack['file_'][i])
					
					# create the files
					self.output_datasets[i] = driver.Create( output_image_name, self.nCol, self.nRow, len(self.output_band_names), gdal.GDT_Int16)

					if self.output_datasets[i] is None:
						return( [False, total_time, "Failed to create dataset " + output_image_name] )
					else:
						print 'Created ', self.output_datasets[i]

					# add the projection and georeferenceing info
					self.output_datasets[i].SetGeoTransform( self.geotrans )
					self.output_datasets[i].SetProjection( self.prj )

					# add the bands and set no data values
					for j in range(0, len(self.output_band_names)):
						self.output_bands[i,j] = self.output_datasets[i].GetRasterBand(j+1)

						if self.output_bands[i,j] is None:
							return( [False, total_time, "Failed to create band " + str(j) + " of dataset " + output_image_name] )

						self.output_bands[i,j].SetNoDataValue(self.nodata)
						
		else:	# open existing
			for i in range(0, self.n_input_datasets):
				# exclude first year of the stack
				# if self.stack['year'][i] <> np.min(self.stack['year']):
				if True:
					output_image_name = self.output_dir + os.path.basename(self.stack['file_'][i])

					# create the files
					self.output_datasets[i] = gdal.Open( output_image_name, gdalconst.GA_Update )

					if self.output_datasets[i] is None:
						return( [False, total_time, "Failed to create dataset " + output_image_name] )
					else:
						print 'Opened ', self.output_datasets[i]			

					for j in range(0, len(self.output_band_names)):
						self.output_bands[i,j] = self.output_datasets[i].GetRasterBand(j+1)

						if self.output_bands[i,j] is None:
							return( [False, total_time, "Failed to create band " + str(j) + " of dataset " + output_image_name] )

		total_time = time.time() - start_time
		return( [True, total_time, "All bands created for " + str(self.n_input_datasets) + " output datasets"] )

	##########
	# close output datasets and generate histograms and pyramids
	##########
	def closeOutputData(self):
		start_time = time.time()
		total_time = -1

		# generate histograms and pyramids for output data layers
		gdal.SetConfigOption('HFA_USE_RRD', 'YES')

		for i in range(0, self.n_input_datasets):
			for j in range(0, len(self.output_band_names)):
				#histogram = self.output_bands[i,j].GetDefaultHistogram()
				#if not histogram == None:
				#	self.output_bands[i,j].SetDefaultHistogram(histogram[0], histogram[1], histogram[3])
				self.output_bands[i,j] = None	# close the band

			#self.output_datasets[i].BuildOverviews(overviewlist=[3,9,27,81,243,729])
			self.output_datasets[i] = None	# close the dataset

		total_time = time.time() - start_time
		return( [True, total_time, 'Closed output files'] )


	##########
	# read a block of data from the stack input images
	##########
	def readBlock(self, startCol, endCol, startRow, endRow):
		start_time = time.time()
		total_time = -1

		# create the StackBlock that will hold input and output data layers
		this_block = StackBlock(startCol, endCol, startRow, endRow, self.stack['year'], self.stack['season'], self.stack['month'], self.clf, self.pBurnThreshold )

		# loop through images and bands, load data from file into the array
		for i in range(0, self.n_input_datasets):
			for j in range(0,len(self.input_band_names)):
				band = self.input_band_names[j]

				this_block.image_data[i, this_block.image_band_names.index(band),:,:] = \
					self.input_bands[i, self.input_band_names.index(band)].ReadAsArray(startCol, startRow, (endCol-startCol), (endRow-startRow))

		total_time = time.time() - start_time	
		return( [this_block, total_time, "Read " + str(endRow-startRow) + " rows of data, and " + str(endCol-startCol) + " columns of data" ] )			
	
	
	##########
	# write a block of output data to the stack output images
	##########
	def writeBlock(self, block):
		start_time = time.time()	
		total_time = -1

		# loop through images and bands, save data to files from array
		for i in range(0, self.n_input_datasets):
			self.output_bands[i,0].WriteArray( block.bc_data[i,0,:,:], xoff=block.startCol, yoff=block.startRow)

		total_time = time.time() - start_time
		return( [True, total_time, "Wrote " + str(block.endRow-block.startRow) + " rows of data, and " + str(block.endCol-block.startCol) + " columns of data" ] )

	def checkOutputDatasets(self):
		for i in range(0, self.n_input_datasets):
			print i, self.output_bands[i,0]

##########
# Class to hold a chunk of image data and process it
# processing includes calculating spectral indices, creating seasonal summaries, and burned probabilities, and binary images of burned|unburned
##########
class StackBlock:
	startCol = -1
	endCol = -1
	nCol = -1
	startRow = -1
	endRow = -1
	nRow = -1

	nodata = -9999

	# arrays of pixels from original images
	image_band_names = ['band1','band2','band3','band4','band5','band6','band7','QA']
	image_data = None
	image_years = None # array of 'years' for each image, duplicate values allowed and assumed
	image_seasons = None # array of 'seasons' for each image
	image_months = None # array of 'months' for each image
	
	# array of spectral indices calculated from original images
	si_band_names = ['ndvi','ndmi','nbr','nbr2']
	si_data = None	# spectral indices

	# array of seasonal summaries calculated from original images and spectral indices
	season_names = ['winter','spring','summer','fall']
	season_years = None		# array of 'years' for each season, no duplicates
	season_image_data = None	# seasonal summaries of image bands
	season_si_data = None		# seasonal summaries of spectral indices

	# array of burn probabilities
	bp_bands = ['probability','QA']
	bp_data = None

	# array of burn classification results - single band image with QA mask included
	bc_data = None

	clf = None		# classifier object / boosted regression tree
	pBurnThreshold = 0.5	# probability threshold to use when converting burned probabilities to binary values


	def __init__(self, startCol, endCol, startRow, endRow, image_years, image_seasons, image_months, clf, clf_prob):
		#print 'Initilizing StackBlock...'
		self.startCol = startCol
		self.endCol = endCol
		self.nCol = endCol - startCol

		self.startRow = startRow
		self.endRow = endRow
		self.nRow = endRow - startRow

		self.image_years = image_years
		self.image_seasons = image_seasons
		self.image_months = image_months
		self.season_years = np.sort(np.unique(self.image_years))

		# create the data arrays
		self.image_data = np.empty(( len(self.image_years), len(self.image_band_names), self.nRow, self.nCol ), dtype='int16')
		
		# copy over the classifier
		self.clf = clf
		self.pBurnThreshold = clf_prob


	def calculateSpectralIndices(self):
		start_time = time.time()
		total_time = -1

		self.si_data = np.empty(( len(self.image_years), len(self.si_band_names), self.nRow, self.nCol ), dtype='int16')

		for i in range(0, self.si_data.shape[0]):
			for j in range(0, len(self.si_band_names)):
				band = self.si_band_names[j]

				if band == 'ndmi':
					b1 = 'band4'
					b2 = 'band5'
				elif band == 'nbr':
					b1 = 'band4'
					b2 = 'band7'
				elif band == 'nbr2':
					b1 = 'band5'
					b2 = 'band7'
				else:	#band == 'ndvi':
					b1 = 'band4'
					b2 = 'band3'

				# calculate the index
				temp = \
					1000.0 * (self.image_data[i, self.image_band_names.index(b1),:,:] - self.image_data[i, self.image_band_names.index(b2),:,:]) / \
					(self.image_data[i, self.image_band_names.index(b1),:,:] + self.image_data[i, self.image_band_names.index(b2),:,:])

				# get the mask
				temp_mask = (self.image_data[i, self.image_band_names.index('QA'),:,:] < 0) | (self.image_data[i, self.image_band_names.index('band1'),:,:] == self.nodata)

				# apply the mask
				temp[temp_mask] = self.nodata

				# transfer the results
				self.si_data[i, self.si_band_names.index(band),:,:] = temp

		total_time = time.time() - start_time
		return( [True, total_time, 'Calculated spectral indices for ' + str(self.si_data.shape[0]) + ' files, and ' + str(self.nCol) + ' columns x ' + str(self.nRow) + ' rows of data'] )


	# generate seasonal summaries of the input image data and spectral indices
	def calculateSeasonalSummaries(self):
		start_time = time.time()
		total_time = -1

		# allocate space to hold the results
		self.season_image_data = np.empty((len(self.season_years), len(self.season_names), len(self.image_band_names), self.nRow, self.nCol), dtype='int16')
		self.season_si_data = np.empty((len(self.season_years), len(self.season_names), len(self.si_band_names), self.nRow, self.nCol), dtype='int16')

		for i in range(0, len(self.season_years)):
			year = self.season_years[i]

			for j in range(0, len(self.season_names)):
				season = self.season_names[j]

				# which input images correspond to the current year & season?
				season_mask = (self.image_seasons == season) & (self.image_years == year)

				# how many good pixels do we have for the current year & season?
				mask_data_good = self.image_data[ season_mask, self.image_band_names.index('QA'),:,:] >= 0
				mask_data_good_count = np.apply_over_axes(np.sum, mask_data_good, axes=[0])[0,:,:]
				self.season_image_data[i,j,self.image_band_names.index('QA'),:,:] = mask_data_good_count

				# generate average values for each band for the current year & season for the image data
				for k in range(0, len(self.image_band_names)):
					band = self.image_band_names[k]

					if band != 'QA':
						temp_data = self.image_data[ season_mask, self.image_band_names.index(band), :, :]
						sum_data = np.apply_over_axes(np.sum, temp_data, axes=[0])[0,:,:]
						mean_data = sum_data / mask_data_good_count
						self.season_image_data[i, j, k,:,:] = mean_data			
						self.season_image_data[i, j, k, mask_data_good_count==0] = self.nodata

				# generate average values for each band for the current year & season for the spectral indices data
				for k in range(0, len(self.si_band_names)):
					band = self.si_band_names[k]

					if band != 'QA':
						temp_data = self.si_data[ season_mask, self.si_band_names.index(band), :, :]
						sum_data = np.apply_over_axes(np.sum, temp_data, axes=[0])[0,:,:]
						mean_data = sum_data / mask_data_good_count
						self.season_si_data[i, j, k,:,:] = mean_data
						self.season_si_data[i, j, k, mask_data_good_count==0] = self.nodata

		total_time = time.time() - start_time
		return( [True, total_time, 'Calculated seasonal summaries for ' + str(self.si_data.shape[0]) + ' files, and ' + str(self.nCol) + ' columns x ' + str(self.nRow) + ' rows of data'] )


	# apply the boosted regression tree to the input data layers, to generate burned probabilities
	def calculateBurnProbabilities(self):
		start_time = time.time()
		total_time = -1

		# predictor names - predictors must be in the same order as they were when the boosted regression tree model was trained
		predictor_band_names = ["month","band1","band2","band3","band4","band5","band7","ndvi","ndmi","nbr","nbr2","wi_b3","wi_b4","wi_b5","wi_b7","wi_ndvi","wi_ndmi","wi_nbr","wi_nbr2","sp_b3","sp_b4","sp_b5","sp_b7","sp_ndvi","sp_ndmi","sp_nbr","sp_nbr2","su_b3","su_b4","su_b5","su_b7","su_ndvi","su_ndmi","su_nbr","su_nbr2","fa_b3","fa_b4","fa_b5","fa_b7","fa_ndvi","fa_ndmi","fa_nbr","fa_nbr2","ly_wi_b3","ly_wi_b4","ly_wi_b5","ly_wi_b7","ly_wi_ndvi","ly_wi_ndmi","ly_wi_nbr","ly_wi_nbr2","ly_sp_b3","ly_sp_b4","ly_sp_b5","ly_sp_b7","ly_sp_ndvi","ly_sp_ndmi","ly_sp_nbr","ly_sp_nbr2","ly_su_b3","ly_su_b4","ly_su_b5","ly_su_b7","ly_su_ndvi","ly_su_ndmi","ly_su_nbr","ly_su_nbr2","ly_fa_b3","ly_fa_b4","ly_fa_b5","ly_fa_b7","ly_fa_ndvi","ly_fa_ndmi","ly_fa_nbr","ly_fa_nbr2"]
		#predictor_band_names = ["month","band1","band2","band3","band4","band5","band7","ndvi","ndmi","nbr","nbr2"]		
		
		# allocate space for a row of predictors
		predictors = np.zeros((self.nCol, len(predictor_band_names)), dtype='int16')

		# allocate space for the results
		self.bp_data = np.empty((len(self.image_years), len(self.bp_bands), self.nRow, self.nCol), dtype='int16')

		# predictions depend on bands, spectral indices, and seasonal indices from:
		#	A. the current image
		#	B. the current year
		#	C. the previous year
		#
		#	Figure out what the smallest year in the stack is, and then 
		#	loop through min_years+1 to max_years
		#
		#	Fix code below to reflect this and to process multiple years
		startYear = np.min(self.image_years)
		endYear = np.max(self.image_years)

		for thisYear in range(startYear+1, endYear+1):
			lastYear = thisYear - 1
			#print 'Generating predictions for ', thisYear
			
			thisYearMask = self.season_years == thisYear
			lastYearMask = self.season_years == lastYear
			
			# make predictions 1 row at a time
			for i in range(0, self.nRow):
				# build predictors that only vary by year
				if True:
					predictors[:,predictor_band_names.index('ly_wi_b3')] =	self.season_image_data[thisYearMask, self.season_names.index('winter'),self.image_band_names.index('band3'),i,:]
					predictors[:,predictor_band_names.index('ly_wi_b4')] =	self.season_image_data[thisYearMask, self.season_names.index('winter'),self.image_band_names.index('band4'),i,:]
					predictors[:,predictor_band_names.index('ly_wi_b5')] =	self.season_image_data[thisYearMask, self.season_names.index('winter'),self.image_band_names.index('band5'),i,:]
					predictors[:,predictor_band_names.index('ly_wi_b7')] =	self.season_image_data[thisYearMask, self.season_names.index('winter'),self.image_band_names.index('band7'),i,:]
					predictors[:,predictor_band_names.index('ly_wi_ndvi')] =	self.season_si_data[thisYearMask, self.season_names.index('winter'),self.si_band_names.index('ndvi'),i,:]
					predictors[:,predictor_band_names.index('ly_wi_ndmi')] =	self.season_si_data[thisYearMask, self.season_names.index('winter'),self.si_band_names.index('ndmi'),i,:]
					predictors[:,predictor_band_names.index('ly_wi_nbr')] =		self.season_si_data[thisYearMask, self.season_names.index('winter'),self.si_band_names.index('nbr'),i,:]
					predictors[:,predictor_band_names.index('ly_wi_nbr2')] =	self.season_si_data[thisYearMask, self.season_names.index('winter'),self.si_band_names.index('nbr2'),i,:]
					predictors[:,predictor_band_names.index('ly_sp_b3')] =	self.season_image_data[thisYearMask, self.season_names.index('spring'),self.image_band_names.index('band3'),i,:]
					predictors[:,predictor_band_names.index('ly_sp_b4')] =	self.season_image_data[thisYearMask, self.season_names.index('spring'),self.image_band_names.index('band4'),i,:]
					predictors[:,predictor_band_names.index('ly_sp_b5')] =	self.season_image_data[thisYearMask, self.season_names.index('spring'),self.image_band_names.index('band5'),i,:]
					predictors[:,predictor_band_names.index('ly_sp_b7')] =	self.season_image_data[thisYearMask, self.season_names.index('spring'),self.image_band_names.index('band7'),i,:]
					predictors[:,predictor_band_names.index('ly_sp_ndvi')] =	self.season_si_data[thisYearMask, self.season_names.index('spring'),self.si_band_names.index('ndvi'),i,:]
					predictors[:,predictor_band_names.index('ly_sp_ndmi')] =	self.season_si_data[thisYearMask, self.season_names.index('spring'),self.si_band_names.index('ndmi'),i,:]
					predictors[:,predictor_band_names.index('ly_sp_nbr')] =		self.season_si_data[thisYearMask, self.season_names.index('spring'),self.si_band_names.index('nbr'),i,:]
					predictors[:,predictor_band_names.index('ly_sp_nbr2')] =	self.season_si_data[thisYearMask, self.season_names.index('spring'),self.si_band_names.index('nbr2'),i,:]
					predictors[:,predictor_band_names.index('ly_su_b3')] =	self.season_image_data[thisYearMask, self.season_names.index('summer'),self.image_band_names.index('band3'),i,:]
					predictors[:,predictor_band_names.index('ly_su_b4')] =	self.season_image_data[thisYearMask, self.season_names.index('summer'),self.image_band_names.index('band4'),i,:]
					predictors[:,predictor_band_names.index('ly_su_b5')] =	self.season_image_data[thisYearMask, self.season_names.index('summer'),self.image_band_names.index('band5'),i,:]
					predictors[:,predictor_band_names.index('ly_su_b7')] =	self.season_image_data[thisYearMask, self.season_names.index('summer'),self.image_band_names.index('band7'),i,:]
					predictors[:,predictor_band_names.index('ly_su_ndvi')] =	self.season_si_data[thisYearMask, self.season_names.index('summer'),self.si_band_names.index('ndvi'),i,:]
					predictors[:,predictor_band_names.index('ly_su_ndmi')] =	self.season_si_data[thisYearMask, self.season_names.index('summer'),self.si_band_names.index('ndmi'),i,:]
					predictors[:,predictor_band_names.index('ly_su_nbr')] =		self.season_si_data[thisYearMask, self.season_names.index('summer'),self.si_band_names.index('nbr'),i,:]
					predictors[:,predictor_band_names.index('ly_su_nbr2')] =	self.season_si_data[thisYearMask, self.season_names.index('summer'),self.si_band_names.index('nbr2'),i,:]
					predictors[:,predictor_band_names.index('ly_fa_b3')] =	self.season_image_data[thisYearMask, self.season_names.index('fall'),self.image_band_names.index('band3'),i,:]
					predictors[:,predictor_band_names.index('ly_fa_b4')] =	self.season_image_data[thisYearMask, self.season_names.index('fall'),self.image_band_names.index('band4'),i,:]
					predictors[:,predictor_band_names.index('ly_fa_b5')] =	self.season_image_data[thisYearMask, self.season_names.index('fall'),self.image_band_names.index('band5'),i,:]
					predictors[:,predictor_band_names.index('ly_fa_b7')] =	self.season_image_data[thisYearMask, self.season_names.index('fall'),self.image_band_names.index('band7'),i,:]
					predictors[:,predictor_band_names.index('ly_fa_ndvi')] =	self.season_si_data[thisYearMask, self.season_names.index('fall'),self.si_band_names.index('ndvi'),i,:]
					predictors[:,predictor_band_names.index('ly_fa_ndmi')] =	self.season_si_data[thisYearMask, self.season_names.index('fall'),self.si_band_names.index('ndmi'),i,:]
					predictors[:,predictor_band_names.index('ly_fa_nbr')] =		self.season_si_data[thisYearMask, self.season_names.index('fall'),self.si_band_names.index('nbr'),i,:]
					predictors[:,predictor_band_names.index('ly_fa_nbr2')] =	self.season_si_data[thisYearMask, self.season_names.index('fall'),self.si_band_names.index('nbr2'),i,:]

					# this year seasonal summaries
					predictors[:,predictor_band_names.index('wi_b3')] =	self.season_image_data[lastYearMask, self.season_names.index('winter'),self.image_band_names.index('band3'),i,:]
					predictors[:,predictor_band_names.index('wi_b4')] =	self.season_image_data[lastYearMask, self.season_names.index('winter'),self.image_band_names.index('band4'),i,:]
					predictors[:,predictor_band_names.index('wi_b5')] =	self.season_image_data[lastYearMask, self.season_names.index('winter'),self.image_band_names.index('band5'),i,:]
					predictors[:,predictor_band_names.index('wi_b7')] =	self.season_image_data[lastYearMask, self.season_names.index('winter'),self.image_band_names.index('band7'),i,:]
					predictors[:,predictor_band_names.index('wi_ndvi')] =	self.season_si_data[lastYearMask, self.season_names.index('winter'),self.si_band_names.index('ndvi'),i,:]
					predictors[:,predictor_band_names.index('wi_ndmi')] =	self.season_si_data[lastYearMask, self.season_names.index('winter'),self.si_band_names.index('ndmi'),i,:]
					predictors[:,predictor_band_names.index('wi_nbr')] =	self.season_si_data[lastYearMask, self.season_names.index('winter'),self.si_band_names.index('nbr'),i,:]
					predictors[:,predictor_band_names.index('wi_nbr2')] =	self.season_si_data[lastYearMask, self.season_names.index('winter'),self.si_band_names.index('nbr2'),i,:]
					predictors[:,predictor_band_names.index('sp_b3')] =	self.season_image_data[lastYearMask, self.season_names.index('spring'),self.image_band_names.index('band3'),i,:]
					predictors[:,predictor_band_names.index('sp_b4')] =	self.season_image_data[lastYearMask, self.season_names.index('spring'),self.image_band_names.index('band4'),i,:]
					predictors[:,predictor_band_names.index('sp_b5')] =	self.season_image_data[lastYearMask, self.season_names.index('spring'),self.image_band_names.index('band5'),i,:]
					predictors[:,predictor_band_names.index('sp_b7')] =	self.season_image_data[lastYearMask, self.season_names.index('spring'),self.image_band_names.index('band7'),i,:]
					predictors[:,predictor_band_names.index('sp_ndvi')] =	self.season_si_data[lastYearMask, self.season_names.index('spring'),self.si_band_names.index('ndvi'),i,:]
					predictors[:,predictor_band_names.index('sp_ndmi')] =	self.season_si_data[lastYearMask, self.season_names.index('spring'),self.si_band_names.index('ndmi'),i,:]
					predictors[:,predictor_band_names.index('sp_nbr')] =	self.season_si_data[lastYearMask, self.season_names.index('spring'),self.si_band_names.index('nbr'),i,:]
					predictors[:,predictor_band_names.index('sp_nbr2')] =	self.season_si_data[lastYearMask, self.season_names.index('spring'),self.si_band_names.index('nbr2'),i,:]
					predictors[:,predictor_band_names.index('su_b3')] =	self.season_image_data[lastYearMask, self.season_names.index('summer'),self.image_band_names.index('band3'),i,:]
					predictors[:,predictor_band_names.index('su_b4')] =	self.season_image_data[lastYearMask, self.season_names.index('summer'),self.image_band_names.index('band4'),i,:]
					predictors[:,predictor_band_names.index('su_b5')] =	self.season_image_data[lastYearMask, self.season_names.index('summer'),self.image_band_names.index('band5'),i,:]
					predictors[:,predictor_band_names.index('su_b7')] =	self.season_image_data[lastYearMask, self.season_names.index('summer'),self.image_band_names.index('band7'),i,:]
					predictors[:,predictor_band_names.index('su_ndvi')] =	self.season_si_data[lastYearMask, self.season_names.index('summer'),self.si_band_names.index('ndvi'),i,:]
					predictors[:,predictor_band_names.index('su_ndmi')] =	self.season_si_data[lastYearMask, self.season_names.index('summer'),self.si_band_names.index('ndmi'),i,:]
					predictors[:,predictor_band_names.index('su_nbr')] =	self.season_si_data[lastYearMask, self.season_names.index('summer'),self.si_band_names.index('nbr'),i,:]
					predictors[:,predictor_band_names.index('su_nbr2')] =	self.season_si_data[lastYearMask, self.season_names.index('summer'),self.si_band_names.index('nbr2'),i,:]
					predictors[:,predictor_band_names.index('fa_b3')] =	self.season_image_data[lastYearMask, self.season_names.index('fall'),self.image_band_names.index('band3'),i,:]
					predictors[:,predictor_band_names.index('fa_b4')] =	self.season_image_data[lastYearMask, self.season_names.index('fall'),self.image_band_names.index('band4'),i,:]
					predictors[:,predictor_band_names.index('fa_b5')] =	self.season_image_data[lastYearMask, self.season_names.index('fall'),self.image_band_names.index('band5'),i,:]
					predictors[:,predictor_band_names.index('fa_b7')] =	self.season_image_data[lastYearMask, self.season_names.index('fall'),self.image_band_names.index('band7'),i,:]
					predictors[:,predictor_band_names.index('fa_ndvi')] =	self.season_si_data[lastYearMask, self.season_names.index('fall'),self.si_band_names.index('ndvi'),i,:]
					predictors[:,predictor_band_names.index('fa_ndmi')] =	self.season_si_data[lastYearMask, self.season_names.index('fall'),self.si_band_names.index('ndmi'),i,:]
					predictors[:,predictor_band_names.index('fa_nbr')] =	self.season_si_data[lastYearMask, self.season_names.index('fall'),self.si_band_names.index('nbr'),i,:]
					predictors[:,predictor_band_names.index('fa_nbr2')] =	self.season_si_data[lastYearMask, self.season_names.index('fall'),self.si_band_names.index('nbr2'),i,:]

				# build predictors that vary by image
				thisYearImageMask = np.where(self.image_years == thisYear)[0]
				
				for j in thisYearImageMask:
					predictors[:,predictor_band_names.index('month')] =	self.image_months[j]
					predictors[:,predictor_band_names.index('band1')] =	self.image_data[j, self.image_band_names.index('band1'),i,:]
					predictors[:,predictor_band_names.index('band2')] =	self.image_data[j, self.image_band_names.index('band2'),i,:]
					predictors[:,predictor_band_names.index('band3')] =	self.image_data[j, self.image_band_names.index('band3'),i,:]
					predictors[:,predictor_band_names.index('band4')] =	self.image_data[j, self.image_band_names.index('band4'),i,:]
					predictors[:,predictor_band_names.index('band5')] =	self.image_data[j, self.image_band_names.index('band5'),i,:]
					predictors[:,predictor_band_names.index('band7')] =	self.image_data[j, self.image_band_names.index('band7'),i,:]
					predictors[:,predictor_band_names.index('ndvi')] =	self.si_data[j, self.si_band_names.index('ndvi'),i,:]
					predictors[:,predictor_band_names.index('ndmi')] =	self.si_data[j, self.si_band_names.index('ndmi'),i,:]
					predictors[:,predictor_band_names.index('nbr')] =	self.si_data[j, self.si_band_names.index('nbr'),i,:]
					predictors[:,predictor_band_names.index('nbr2')] =	self.si_data[j, self.si_band_names.index('nbr2'),i,:]

					# make predictions
					# COMMENTED OUT FOR CLUSTER MPI TESTSING
					# pfire = (1000.0 * self.clf.predict_proba(predictors)[:,1]).astype(int)

					# update output array
					self.bp_data[j,0,i,:] = self.image_data[j, self.image_band_names.index('QA'),i,:]
					self.bp_data[j,1,i,:] = self.image_data[j, self.image_band_names.index('band3'),i,:]	# UNCOMMENTED FOR CLUSTER/MPI TESTING

					# COOMMENTED OUT FOR CLUSTER MPI TESTING
					# self.bp_data[j,1,i,:] = pfire

		total_time = time.time() - start_time
		return( [True, total_time, 'Calculated burn probabilities summaries for ' + str(self.si_data.shape[0]) + ' files, and ' + str(self.nCol) + ' columns x ' + str(self.nRow) + ' rows of data'] )


	# apply threshold to burn probabilities to generate binary burned|unburned results
	def classifyBurnProbabilities(self):
		start_time = time.time()
		total_time = -1

		# allocate space for the results
		self.bc_data = np.empty(( len(self.image_years), 1, self.nRow, self.nCol ), dtype='int16')

		# loop through the images
		for i in range(0, self.si_data.shape[0]):
			#self.bc_data[i,0,:,:] = self.image_data[i, self.image_band_names.index('QA'),:,:]		# test to make sure QA bands are doing what we think - looks good
			self.bc_data[i,0,:,:] = self.si_data[i, self.si_band_names.index('nbr'),:,:]			# test to make sure SI bands are doing what we think - looks good
			
			# test to make sure seasonal summaries are doing what we think - no good!!!!
			#year = self.image_years[i]
			#self.bc_data[i,0,:,:] = self.season_si_data[np.where(self.season_years == year), self.season_names.index('spring'), self.si_band_names.index('nbr'),:, :]

		total_time = time.time() - start_time
		return( [True, total_time, 'Classified burn probabilities for ' + str(self.si_data.shape[0]) + ' files, and ' + str(self.nCol) + ' columns x ' + str(self.nRow) + ' rows of data'] )


##########
# function used to encapsulate block processing for multi-threading
##########
def processBlock(in_block):
	start_time = time.time()
	total_time = -1
		
	status = in_block.calculateSpectralIndices()
	#print status

	status = in_block.calculateSeasonalSummaries()
	#print status

	status = in_block.calculateBurnProbabilities()
	#print status

	status = in_block.classifyBurnProbabilities()
	#print status

	# empty all the input and intermediate data arrays in my_block before returing to main loop
	total_time = time.time() - start_time
	return( [in_block, total_time, 'Completed block processing for ' + str(in_block.si_data.shape[0]) + ' files, and rows:' + str(in_block.startRow) + '-' + str(in_block.endRow) + ' and cols:' + str(in_block.startCol) + '-' + str(in_block.endCol)] )
	

##########
# test
##########
if __name__ == "__main__":
	start_time0 = time.time()
	
	# some initial settings
	my_path = "025"
	my_row = "034"
	my_root_dir = "/data/landsat/FireECV/p" + my_path + "r" + my_row + "/"
	my_stack_file = my_root_dir + "test_tif_stack.csv"	# small stack to test with
	my_input_dir = my_root_dir + "tif/"
	#my_output_dir = my_root_dir + "test/"
	my_output_dir =  "/work/675/test/"

	# create the stack object and set values
	my_stack = Stack(my_stack_file)
	my_stack.input_dir = my_input_dir
	my_stack.output_dir = my_output_dir

	my_shelf_file = "/data/landsat/FireECV/RegionalModels/EasternTemperateForests.shelf"
	my_stack.openClassifier(my_shelf_file)

	status = my_stack.openInputDatasets()
	print status

	if rank==0:
		status = my_stack.openOutputDatasets(create=True)
		print status
	else:
		#time.sleep(10)
		status = my_stack.openOutputDatasets(create=False)

	# Let everyone catch up
	comm.barrier()
	if rank == 0: print "Off We Go"
	#print status

	##########
	# loop through 'blocks' in the images
	##########
	max_row = my_stack.nRow
	max_col = my_stack.nCol
	block_rows = 256
	block_cols = 256
	num_blocks =  (max_row/block_rows)*(max_col/block_cols)
	block_coords = [ ] 


	#Generate a list of lists that contains stack block coordinates to send to all workers
	if rank == 0:
		for _ in range(num_blocks):
			total_blocks = 0

			for startRow in range(0, max_row, block_rows):
	
				for startCol in range(0, max_col, block_cols ):
	
					endRow = startRow + block_rows 
					if endRow > max_row:
						endRow = max_row
					endCol = startCol + block_cols
					if endCol > max_col:
						endCol = max_col
	
					if ( startRow < max_row ) & ( endRow <= max_row ) & ( startCol < max_col ) & ( endCol <= max_col ):
						block_coords.append([startRow, endRow, startCol, endCol])
						total_blocks =  total_blocks + 1
					else:
						print "Coords dont make sense"
						


		print "Generated block_coords array with %d total blocks" %total_blocks	

		rank_blocks = total_blocks/( size - 1 )
		assigned_blocks = []
		for x in range(0, total_blocks, rank_blocks):
			max_x = x + rank_blocks
			ylist = []
			if max_x > total_blocks:
				max_x = total_blocks	
			for y in range(x, max_x):
				ylist.append(y)
			assigned_blocks.append(ylist)
				
	else:
		assigned_blocks = None
		block_coords = None
		total_blocks = None

	# Broadcast to all ranks
	assigned_blocks = comm.bcast(assigned_blocks, root=0)
	block_coords = comm.bcast(block_coords, root=0)
	total_blocks = comm.bcast(total_blocks, root=0)


	if rank != 0:
		for block_index in range(len(assigned_blocks[rank - 1])): 
			block_id = assigned_blocks[rank-1][block_index]
			#print "block id", block_id, "has coords", block_coords[block_id][0], block_coords[block_id][1], block_coords[block_id][2], block_coords[block_id][3]
			print "########################################"
			print "Rank ", rank, " is Processing block id", block_id, "with rows:", block_coords[block_id][0], "-", block_coords[block_id][1], " and columns:", block_coords[block_id][2], "-", block_coords[block_id][3], "..."
			result = my_stack.readBlock(block_coords[block_id][2], block_coords[block_id][3], block_coords[block_id][0], block_coords[block_id][1])
			#print result[1:3]
			in_block = result[0]
			result = processBlock(in_block)
			rout_block = result[0]
			comm.send(rout_block, tag=1, dest=0)
		# Tell the root we're done with our work
		comm.send(rank, tag=9, dest=0)

	else:
		# Rank 0 will receive all of the processed blocks and write them to disk
		fin_rank = [ ]
		out_blocks_recv = 1
		workerRank = None
		while True:
			#Test for the tag on the incoming messages.
			# Tag=1 is a processed block, tag=9 is an "all done" message
			if comm.Iprobe(source=MPI.ANY_SOURCE, tag=1):
				out_block=comm.recv(source=MPI.ANY_SOURCE, tag=1)
				#print "Got rout_block and writing out_block"
				status = my_stack.writeBlock(out_block)
				#print "Rank 0:", status
				print total_blocks - out_blocks_recv, "blocks left to process out of", total_blocks
				out_blocks_recv = out_blocks_recv + 1
			elif comm.Iprobe(source=MPI.ANY_SOURCE, tag=9):
				fin_rank.append(comm.recv(workerRank, source=MPI.ANY_SOURCE, tag=9))
				print "Received a finished message from rank", fin_rank[-1]
				if len(fin_rank) == (size - 1):
					print "All ranks completed processing"
					break

	

# Wait for everyone
comm.barrier()

# Celebrate - we're done. Close the output data
if rank==0:
	status = my_stack.closeOutputData()
	print status

	end_time0 = time.time()
	print 'Done! Total processing time = ' + str((end_time0 - start_time0)/60) + ' minutes.'

sys.exit()

