# The result of standardization is that the features will be rescaled so that
# they'll have the properties of a standard normal distribution.
#
# Most machine learning algoritms will only take standardized features as input,
# which makes it crucial to standardize our values. Another use-case is when
# we have features with different units. One feature of an instance may
# differ alot, and without normalization/standardization this value will cause
# inaccurate results since this feature may 'take over' another feature.

# Make sure that we use python3-like float-division
from __future__ import division

# Standard deviation defines the average distance to the mean in a sequence
# of numbers.
# We are using pow shorthand (**) to square and get the square root.
def std_dev(x):
    mean = sum(x)/len(x)
    return (1/len(x) * sum([(x_i - mean)**2 for x_i in x]))**0.5

def z_scores(x):
    mean = sum(x)/len(x)
    return [(x_i - mean)/std_dev(x) for x_i in x]

def min_max_scaling(x):
    return [(x_i - min(x)) / (max(x) - min(x)) for x_i in x]

# Test our implementation of standard deviation
seq = [3,4,6,7]
print('Stddev for: ' + repr(seq) + ' -> ' + repr(std_dev(seq)))

# Test z_score implementation
print('Z-scores for: ' + repr(seq) + ' -> ' + repr(z_scores(seq)))

# Test min max scaling
print('Min-max scaling for: ' + repr(seq) + ' -> ' + repr(min_max_scaling(seq)))
