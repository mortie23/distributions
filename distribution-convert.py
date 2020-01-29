# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Fitting statistical distributions to single variable data
# %% [markdown]
# ## Fit an exponential distribution

# %%
## Import pacakages
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


# %%
## generate random exponential data with location 0.5 and scale 1.2
exp_dist = ss.expon.rvs(loc=1, scale=1.2, size=1000)

## maximum likelihood estimation (MLE)
mle = ss.expon.fit(exp_dist)
loc, scale = mle
print(loc, scale)
## not exactly 0.5 and 1.2, due to being a finite sample


# %%
## Plotting
## generate a linespace with 100 points for the X axis
x_axis = np.linspace(0,10, 1000)
y_axis_pdf = ss.expon.pdf(x_axis, *mle)
## just unpack P with *P, instead of scale=XX and shape=XX, etc.

## need to plot the normalized histogram with `density=True`
counts, bins, bars = plt.hist(exp_dist, bins=10, density=True, alpha=0.6, color='blue')
print(counts, bins)
plt.plot(x_axis, y_axis_pdf, 'k', linewidth=2)

## Add a title
title = "Fit results:"
plt.title(title)

plt.show()

# %% [markdown]
# ## Now for a normal distribution

# %%
## Generate some data for this demonstration.
norm_dist = ss.norm.rvs(10.0, 1.5, size=1000)

## Fit a normal distribution to the data:
mu, std = ss.norm.fit(norm_dist)
print(mu, std)


# %%
## Plot the histogram.
counts, bins, bars = plt.hist(norm_dist, bins=20, density=True, alpha=0.6, color='blue')
print(counts, bins)

## Plot the probability density function (PDF).
## get range from the already plotted histogram
xmin, xmax = plt.xlim()
## generate a linespace with 100 points for the x axis
x = np.linspace(xmin, xmax, 100)
## Calculate the PDF
p = ss.norm.pdf(x, mu, std)
## Add PDF line to histogram plot
plt.plot(x, p, 'k', linewidth=5)

## Add a title
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()


# %%
# This running in VSCode on WSL

