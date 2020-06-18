import numpy as np
from matplotlib import pyplot as plt

import logging
import os

n_interleaves = 8 # Number of interleaves
n_delays = 2 # Number of delays
n_scans = 101 # Number of scans

# Change directory to where files are located
path = "/Users/arthun/Documents/Uni/Masterarbeit/Messsoftware/raw_data_20200527/probe_small"
os.chdir(path)

# Preallocate arrays
m1_mean = np.load("./combined_data_sets/m1_mean.npy")
m1_weight = np.load("./combined_data_sets/m1_weight.npy")

m2_mean = np.load("./combined_data_sets/m2_mean.npy")
m2_weight = np.load("./combined_data_sets/m2_weight.npy")
m2_diff_spec = np.load("./combined_data_sets/m2_diff_spec.npy")

m3_mean = np.load("./combined_data_sets/m3_mean.npy")
m3_weight = np.load("./combined_data_sets/m3_weight.npy")
m3_diff_spec = np.load("./combined_data_sets/m3_diff_spec.npy")

# For the time being we are going to ignore the interleaves averaging and just
# process each interleave as is.


# m1 ------------- Start with taking the scan averaged data from m1 (difference spectra)

# Lets start with the simplest case.
# Just averaging the data - no weights no nothing
m1_no_weights = np.average(m1_mean, axis=1)

# Weight m1 with it's own weights (difference spectra weights)
m1_m1_weights = np.average(m1_mean, axis=1, weights=m1_weight)

# Weight m1 with the inverse variance of m2 (transmission variance)
# the weights array of m2 has one additional dimension when compared to m1.
# This is because the chopper states (high/ low) have not yet been consolidated.
# We will consolidate them by: 

# 1. Den größten wert der 2 inversen varianzen nehmen
weights = m2_weight.min(axis=-1) 
m1_m2_v1_weights = np.average(m1_mean, axis=1, weights=weights)

# 2. Die Varianzen mit der "Julians Vorlesung Formel" berechnen
# Dies entspricht in einer krassen, aber der einzig möglichen Nähereung: 
# Die Varianz zweier Zufallsvariablen Formel unter
# Vernachlässigung der Covarianz verwenden.
weights = m2_weight.sum(axis=-1)
m1_m2_v2_weights = np.average(m1_mean, axis=1, weights=weights)

# Weight m1 with the inverse variance of m3 (intensity variance)
# the weights array of m3 has one additional dimension and double the 
# rows when compared to m1.
# This is because the chopper states (high/ low) 
# and probe and reference array of the MCT have not yet been consolidated.
# We will consolidate them by: 

# 1. Den kleinsten wert der 4 inversen varianzen nehmen
weights = m3_weight.min(axis=-1)
weights = weights.reshape((2, 101, 8, 64,2)) 
weights = weights.min(axis=-1)
m1_m3_v1_weights = np.average(m1_mean, axis=1, weights=weights)

# 2. Die Varianzen mit der "Julians Vorlesung Formel" berechnen
# Dies entspricht in einer krassen, aber der einzig möglichen Nähereung: 
# Die Varianz zweier Zufallsvariablen Formel unter
# Vernachlässigung der Covarianz verwenden.
weights = m3_weight.sum(axis=-1)
weights = weights[:,:,:,::2] + weights[:,:,:,1::2]
m1_m3_v2_weights = np.average(m1_mean, axis=1, weights=weights)


#! ----------m1: The summing the variance for all pixels method is missing for m1, m2, m3 (and all versions) weights

# m2 ------------- Start with taking the scan averaged data from m2 (transmission)
def calculate_spectra_m2(data):
    spectrum = - np.log10(data.take(0, axis=-1)) + np.log10(data.take(1, axis=-1))
    return spectrum

# Lets start with the simplest case.
# Just averaging the data - no weights no nothing
m2_no_weights = np.average(m2_mean, axis=1)
m2_no_weights_ds = calculate_spectra_m2(m2_no_weights)

# Weight m2 with it's own weights (transmission weights)
m2_m2_weights = np.average(m2_mean, axis=1, weights=m2_weight)
m2_m2_weights_ds = calculate_spectra_m2(m2_m2_weights)

# Weight m2 with it's own weights (transmission weights)
weight = m2_weight.copy()
weight[:,:,:,:,1] = weight[:,:,:,:,0]
m2_m2_weights_test = np.average(m2_mean, axis=1, weights=weight)
m2_m2_weights_test_ds = calculate_spectra_m2(m2_m2_weights_test)

# Weight m2 with its' own weights (but take the same weight for every pixel by summing weights)
to_tile = m2_weight.sum(axis=-2)
# weight = np.tile(to_tile, 64).reshape((2, 101, 8, 2, 64))
# weight = np.swapaxes(weight, 3, 4)
weight = np.zeros(m2_weight.shape)
for i in range(m2_weight.shape[-2]):
    weight[:,:,:,i,:] = to_tile
m2_m2_summed_weights = np.average(m2_mean, axis=1, weights=weight)
m2_m2_summed_weights_ds = calculate_spectra_m2(m2_m2_summed_weights)

# Averaging weights is exactly identical to summing weights
# # Weight m2 with its' own weights (but take the same weight for every pixel by averaging weights)
# weight = np.average(m2_weight, axis=-2)
# weight = np.tile(weight, 64).reshape((2, 101, 8, 2, 64))
# weight = np.swapaxes(weight, 3, 4)
# m2_m2_avg_weights = np.average(m2_mean, axis=1, weights=weight)
# m2_m2_avg_weights_ds = calculate_spectra_m2(m2_m2_avg_weights)

# Weight m2 with m1 weights (shot to shot difference spectra inverse variance)
weight = np.zeros((2, 101, 8, 64, 2))
for i in range(m2_weight.shape[-1]):
    weight[:,:,:,:,i] = m1_weight
m2_m1_weights = np.average(m2_mean, axis=1, weights=weight)
m2_m1_weights_ds = calculate_spectra_m2(m2_m1_weights)

#! m1 one variance for all pixel still missing

# Weight m2 with m3 weights (intensity inverse variance)
# Using maximum inverse variance as weight
# This way toooo disgusting to program

# Weight m2 with m3 weights (intensity inverse variance)
# Summing the inverse variances
weight = m3_weight[:,:,:,::2,:] + m3_weight[:,:,:,1::2,:]
m2_m2_weights_v2 = np.average(m2_mean, axis=1, weights=weight)
m2_m3_weights_ds = calculate_spectra_m2(m2_m2_weights_v2)

#! m3 one variance for all pixel still missing

# m2 ------------- Start with taking the scan averaged data from m2 (transmission)
def calculate_spectra_m3(data):
    # transmission
    data = data[:,:,:64,:] / data[:,:,64:,:]
    spectrum = - np.log10(data.take(0, axis=-1)) + np.log10(data.take(1, axis=-1))
    return spectrum

# Lets start with the simplest case.
# Just averaging the data - no weights no nothing
m3_no_weights = np.average(m3_mean, axis=1)
m3_no_weights_ds = calculate_spectra_m3(m3_no_weights)

# Weight m3 with its' own weights
m3_m3_weights = np.average(m3_mean, axis=1, weights=m3_weight)
m3_m3_weights_ds = calculate_spectra_m3(m3_m3_weights)

#! This method is missing for m1
# Weight m3 with its' own weights (but take the same weight for every pixel by summing weights)
to_tile = m3_weight.sum(axis=-2)
# weight = np.tile(to_tile, 64).reshape((2, 101, 8, 2, 64))
# weight = np.swapaxes(weight, 3, 4)
weight = np.zeros(m3_weight.shape)
for i in range(m3_weight.shape[-2]):
    weight[:,:,:,i,:] = to_tile
m3_m3_summed_weights = np.average(m3_mean, axis=1, weights=weight)
m3_m3_summed_weights_ds = calculate_spectra_m3(m3_m3_summed_weights)

# Weight m3 with m1 weights (shot to shot difference spectra inverse variance)
weight = np.zeros(m3_mean.shape)
for i in range(m3_weight.shape[-1]):
    weight[:,:,:,::2,i] = m1_weight
    weight[:,:,:,1::2,i] = m1_weight

m3_m1_weights = np.average(m3_mean, axis=1, weights=weight)
m3_m1_weights_ds = calculate_spectra_m3(m3_m1_weights)

# Weight m3 with m2 weights (transmission inverse variance)
weight = np.zeros(m3_mean.shape)

weight[:,:,:,::2,:] = m2_weight
weight[:,:,:,1::2,:] = m2_weight

m3_m2_weights = np.average(m3_mean, axis=1, weights=weight)
m3_m2_weights_ds = calculate_spectra_m3(m3_m2_weights)

# Weight m3 with m2 weights (transmission inverse variance)  (but take the same weight for every pixel by summing weights)
to_tile = m2_weight.sum(axis=-2)
# weight = np.tile(to_tile, 64).reshape((2, 101, 8, 2, 64))
# weight = np.swapaxes(weight, 3, 4)
x = np.zeros(m2_weight.shape)
for i in range(m2_weight.shape[-2]):
    x[:,:,:,i,:] = to_tile

weight = np.zeros(m3_mean.shape)

weight[:,:,:,::2,:] = x
weight[:,:,:,1::2,:] = x

m3_m2_summed_weights = np.average(m3_mean, axis=1, weights=weight)
m3_m2_summed_weights_ds = calculate_spectra_m3(m3_m2_summed_weights)

# In between plotting
# plt.plot(m1_no_weights[1,4,:], label="m1: no weights")
# plt.plot(m1_m1_weights[1,4,:], label="m1: m1 weights")
# plt.plot(m1_m2_v1_weights[1,4,:], label="m1: m2 v1 weights")
# plt.plot(m1_m2_v2_weights[1,4,:], label="m1: m2 v2 weights")
# plt.plot(m1_m3_v1_weights[1,4,:], label="m1: m3 v1 weights")
# plt.plot(m1_m3_v2_weights[1,4,:], label="m1: m3 v2 weights")

# plt.plot(m2_no_weights_ds[1,4,:], label="m2: no weights")
# plt.plot(m2_no_weights_ds[1,:,:].sum(axis=0), label="m2: no weights")
# plt.plot(m2_m2_weights_ds[1,4,:], label="m2: m2 weights")
# plt.plot(m2_m2_weights_ds[1,:,:].sum(axis=0), label="m2: m2 weights")
# plt.plot(m2_m2_weights_test_ds[1,4,:], label="m2: m2 weights test")
# plt.plot(m2_m2_summed_weights_ds[1,4,:], label="m2: m2 weights same weight for every pixel by summing weights")
# plt.plot(m2_m1_weights_ds[1,4,:], label="m2: m1 weights")
# plt.plot(m2_m3_weights_ds[1,4,:], label="m2: m3 v2 weights")

# plt.plot(m3_no_weights_ds[1,4,:], label="m3: no weights")
# plt.plot(m3_m3_weights_ds[1,4,:], label="m3: m3 weights")
# plt.plot(m3_m3_summed_weights_ds[1,4,:], label="m3: m3 weights same weight for every pixel by summing weights")
# plt.plot(m3_m1_weights_ds[1,4,:], label="m3: m1 weights")
# plt.plot(m3_m2_weights_ds[1,4,:], label="m3: m2 weights")
# plt.plot(m3_m2_summed_weights_ds[1,4,:], label="m3: m2 weights same weight for every pixel by summing weights")

# for i in range(8):
#     plt.plot(m2_m2_weights_test_ds[1,i,:], label="m2: m2 weights test interleave {}".format(i))



# plt.xlabel("pixel")
# plt.ylabel("difference spectrum [OD]")

# plt.legend(fontsize=6)
# plt.grid()
# plt.show()



w = m2_weight.sum(axis=2)
d = np.average(m2_mean, axis=2)
t = np.average(d, axis=1, weights=d)
a = calculate_spectra_m2(t)
print(a.shape)
# plt.plot(np.average(m2_no_weights_ds[1,:,:], axis=0)-a[1], label="m2: no weights")
plt.plot(np.average(m2_no_weights_ds[1,:,:], axis=0), label="m2: no weights")
plt.plot(np.average(m2_m2_weights_ds[1,:,:], axis=0), label="m2: m2 weights")
plt.plot(a[1], label="interleaves averaged first")
# plt.plot(m2_weight[1,50,:,:,1].T)
plt.legend()
plt.grid()
plt.show()
