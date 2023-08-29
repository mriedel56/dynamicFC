import os
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib
import pywt
import numpy as np
import bct
import pandas as pd


def download_abide(output_dir, override_options=None):
	
	os.makedirs(output_dir, exist_ok=True)

	options = {}
	options['data_dir'] = output_dir
	options['n_subjects'] = 1
	options['pipeline'] = 'cpac'
	options['band_pass_filtering'] = True
	options['global_signal_regression'] = False
	options['derivatives'] = ['func_preproc']
	options['quality_checked'] = True
	options['url'] = None
	options['verbose'] = 1
	options['legacy_format'] = True

	if override_options:
		options.update(override_options)

	datasets.fetch_abide_pcp(data_dir=options['data_dir'], 
							 n_subjects=options['n_subjects'], 
							 pipeline=options['pipeline'], 
							 band_pass_filtering=options['band_pass_filtering'], 
							 global_signal_regression=options['global_signal_regression'],
							 derivatives=options['derivatives'],
							 quality_checked=options['quality_checked'] ,
							 url=options['url'],
							 verbose=options['verbose'],
							 legacy_format=options['legacy_format'])

## assume fmriprep has already been run

# run denoiosing using nilearn

# segment brain
def extract_roi_timeseries(fmri_filename, atlas):

	dataset = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
	atlas_filename = dataset.maps
	labels = dataset.labels

	print(f"Atlas ROIs are located in nifti image (4D) at: {atlas_filename}")

	masker = NiftiLabelsMasker(
	    labels_img=atlas_filename,
	    standardize="zscore_sample")

	time_series = masker.fit_transform(fmri_filename)

	return time_series


# perform wavelet decomposition
def calculate_wavelet_transform(time_series, levels):

	wavelet_data = [np.zeros((np.shape(time_series)[0],np.shape(time_series)[1])) for i in range(levels)]
	for i in range(np.shape(time_series)[1]):

		wavelets = pywt.swt(time_series[:,i], 'db8', level=levels, norm=True, trim_approx=True)[1:]
		for j in range(levels):

			wavelet_data[j][:,i] = wavelets[j]

	return wavelet_data


def sliding_window(time_series, width, overlap):

	ts_matrix = []
	start_ind = 0
	end_ind = width

	while end_ind <= np.shape(time_series)[0]:
		ts_matrix.append(time_series[start_ind:end_ind,:])
		start_ind = start_ind + int(width*overlap)
		end_ind = start_ind + width

	return ts_matrix


# calculate correlation matrices
def calculate_correlation_matrices(ts_matrix):

	correlation_measure = ConnectivityMeasure(kind="correlation")
	correlation_matrix = correlation_measure.fit_transform([ts_matrix])[0]

	return correlation_matrix


# threshold correlation matrices
def threshold_matrix(matrix, densities, positive_only=True, binarize=True):

	if positive_only:
		matrix[matrix<0] = 0

	density_matrices = []
	for density in densities:

		matrix_thresholded = bct.utils.other.threshold_proportional(matrix, density, copy=True)
		if binarize:
			matrix_thresholded[matrix_thresholded>0] = 1

		density_matrices.append(matrix_thresholded)

	return density_matrices


# calculate graph metrics
def calculate_graph_metrics(matrix):

	metrics_dict = {}
	#metrics_dict['betweenness'] = bct.algorithms.centrality.betweenness_bin(matrix)
	metrics_dict['global_efficiency'] = [bct.algorithms.efficiency.efficiency_bin(matrix)]
	#metrics_dict['local_efficiency'] = bct.algorithms.efficiency.efficiency_bin(matrix, local=True)

	return metrics_dict


def workflow(experiment_params):

	ts = extract_roi_timeseries(experiment_params['fmri_filename'], None)

	print('Number of Time Series Matrices: 1')


	if experiment_params['calculate_wavelets'] == True:
		ts_matrix = calculate_wavelet_transform(ts, experiment_params['wavelet_levels'])

	else:
		ts_matrix = [ts]

	print('Number of Time Series Matrices: ', len(ts_matrix))


	ts_matrices = []
	for ts in ts_matrix:
		if experiment_params['sliding_window'] == True:
			ts_sliding_window = sliding_window(ts, experiment_params['sliding_window_width'], experiment_params['sliding_window_overlap'])
			ts_matrices.append(ts_sliding_window)
		else:
			ts_matrices.append([ts])


	correlation_matrices = []
	for ts_matrix in ts_matrices:
		print('Number of Time Series Matrices: ', len(ts_matrix))
		corrmats = []
		for ts in ts_matrix:
			corrmats.append(calculate_correlation_matrices(ts))
		print('Number of Correlation Matrices: ', len(corrmats))
		correlation_matrices.append(corrmats)

	print('Number of Correlation Matrices: ', len(correlation_matrices))


	thresholded_matrices = []
	for corrmats in correlation_matrices:
		thresh_mat = []
		for corrmat in corrmats:
			thresh_mat.append(threshold_matrix(corrmat, experiment_params['threshold_densities'], positive_only=experiment_params['positive'], binarize=experiment_params['binarize']))
		thresholded_matrices.append(thresh_mat)


	df = pd.DataFrame()
	for i, thresholded_matrix in enumerate(thresholded_matrices):
		for j, matrix in enumerate(thresholded_matrix):
			for k, dense_mat in enumerate(matrix):
				tmp_dict = calculate_graph_metrics(dense_mat)

				tmp_dict['density'] = experiment_params['threshold_densities'][k]
				if experiment_params['sliding_window'] == True:
					tmp_dict['sliding_window_index'] = j
				if experiment_params['calculate_wavelets'] == True:
					tmp_dict['wavelet_level'] = i
				
				tmp_df = pd.DataFrame.from_dict(tmp_dict)
				df = df._append(tmp_df, ignore_index=True)

	print(df)


def generate_experiment_params(init_params={}):
	
	experiment_params = {}
	experiment_params['fmri_filename'] = os.path.join(os.getenv('project_path'), 'data', 'abide', 'ABIDE_pcp', 'cpac', 'filt_noglobal', 'Pitt_0050003_func_preproc.nii.gz')
	
	img = nib.load(experiment_params['fmri_filename'])
	experiment_params['tr'] = img.header['pixdim'][4]
	experiment_params['threshold_densities'] = [0.2,0.3,0.4,0.5]
	experiment_params['positive'] = True
	experiment_params['binarize'] = True
	experiment_params['sliding_window_width'] = int(44/experiment_params['tr'])
	experiment_params['sliding_window_overlap'] = 0.5
	experiment_params['sliding_window'] = True
	experiment_params['calculate_wavelets'] = False
	experiment_params['wavelet_levels'] = 2

	

	#override some keys
	for param in init_params.keys():
		experiment_params[param] = init_params[param]

	return experiment_params


experiment_params = generate_experiment_params()
print(experiment_params)
workflow(experiment_params)

