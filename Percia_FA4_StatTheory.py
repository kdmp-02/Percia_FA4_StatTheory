#!/usr/bin/env python
# coding: utf-8

# #### Formative Assessment 4
# 
# The data in Table 5.1 samples of size 50 from a normal distribution, a skewed-right distribution, a skewed-left distribution, and a uniform distribution.
# 
# The normal data are female height measurements, the skewed-right data are age at marriage for females, the skewed-left data are obituary data that give the age at death for females, and the uniform data are the amount of cola put into a 12 ounce container by a soft drinks machine.

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

data = {
    "Normal": [67, 70, 63, 65, 68, 60, 70, 64, 69, 61, 66, 65, 71, 62, 66, 68, 64, 67, 62, 66, 65, 63, 66, 65, 63, 69, 
               62, 67, 59, 66, 65, 63, 65, 60, 67, 64, 68, 61, 69, 65, 62, 67, 70, 64, 63, 68, 64, 65, 61, 66],
    "Skewed-right": [31, 43, 30, 30, 38, 26, 29, 55, 46, 26, 29, 57, 34, 34, 36, 40, 28, 26, 66, 63, 30, 33, 24, 35, 34, 
                     40, 24, 29, 24, 27, 35, 33, 75, 38, 34, 85, 29, 40, 41, 35, 26, 34, 19, 23, 28, 26, 31, 25, 22, 28],
    "Skewed-left": [102, 55, 70, 95, 73, 79, 60, 73, 89, 85, 72, 92, 76, 93, 76, 97, 10, 70, 85, 25, 83, 58, 10, 92, 82, 
                    87, 104, 75, 80, 66, 93, 90, 84, 73, 98, 79, 35, 71, 90, 71, 63, 58, 82, 72, 93, 44, 65, 77, 81, 77],
    "Uniform": [12.1, 12.1, 12.4, 12.1, 12.1, 12.2, 12.2, 12.2, 11.9, 12.2, 12.3, 12.3, 11.7, 12.3, 12.3, 12.4, 12.4, 12.1, 
                12.4, 12.4, 12.5, 11.8, 12.5, 12.5, 12.5, 11.6, 11.6, 12, 11.6, 11.6, 11.7, 12.3, 11.7, 11.7, 11.7, 11.8, 
                12.5, 11.8, 11.8, 11.8, 11.9, 11.9, 11.9, 12.2, 11.9, 11.9, 12, 11.9, 12, 12]
}

df = pd.DataFrame(data)

plt.figure(figsize=(14, 8))

for i, column in enumerate(df.columns, 1):
    plt.subplot(2, 2, i)
    sns.distplot(df[column], kde=True)
    plt.title(f'Distribution of {column} Data')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# The graph was made above to show its skewness and visualize the data distribution.
# 
# Find the (a) first, (b) second, (c) third, and (d ) fourth moments for each of the sets of data (normal, skewed-right, skewed-left, uniform).

# In[3]:


import numpy as np
from scipy.stats import moment, skew, kurtosis

def compute_moments(data):
    mean = np.mean(data)
    variance = np.var(data)
    skewness = skew(data)
    kurt = kurtosis(data)
    return mean, variance, skewness, kurt

for key, values in data.items():
    mean, variance, skewness, kurt = compute_moments(values)
    print(f"{key} data:")
    print(f"  1st moment (Mean): {mean}")
    print(f"  2nd moment (Variance): {variance}")
    print(f"  3rd moment (Skewness): {skewness}")
    print(f"  4th moment (Kurtosis): {kurt}\n")


# Find the (a) first, (b) second, (c) third, and (d ) fourth moments about the mean for each of the sets of data (normal, skewed-right, skewed-left, uniform).

# In[4]:


def compute_moments_about_mean(data):
    mean = np.mean(data)  
    second_moment = moment(data, moment=2)
    third_moment = moment(data, moment=3)
    fourth_moment = moment(data, moment=4)
    return mean, second_moment, third_moment, fourth_moment

for key, values in data.items():
    mean, second_moment, third_moment, fourth_moment = compute_moments_about_mean(values)
    print(f"{key} data:")
    print(f"  1st moment (Mean): {mean}")
    print(f"  2nd moment (Variance): {second_moment}")
    print(f"  3rd moment: {third_moment}")
    print(f"  4th moment: {fourth_moment}\n")


# Find the (a) first, (b) second, (c) third, and (d ) fourth moments about the number 75 for the set of female height measurements.

# In[5]:


female_heights = [67, 70, 63, 65, 68, 60, 70, 64, 69, 61, 66, 65, 71, 62, 66, 68, 64, 67, 62, 66, 65, 63, 66, 65, 63, 69, 
                  62, 67, 59, 66, 65, 63, 65, 60, 67, 64, 68, 61, 69, 65, 62, 67, 70, 64, 63, 68, 64, 65, 61, 66]

adjusted_heights = np.array(female_heights) - 75

# Compute moments about 75
first_moment = np.mean(adjusted_heights)
second_moment = moment(adjusted_heights, moment=2)
third_moment = moment(adjusted_heights, moment=3) 
fourth_moment = moment(adjusted_heights, moment=4)

print(f"Moments about 75 for female height measurements:")
print(f"  1st moment: {first_moment}")
print(f"  2nd moment: {second_moment}")
print(f"  3rd moment: {third_moment}")
print(f"  4th moment: {fourth_moment}")


# Using the results of items 2 and 3 for the set of female height measurements, verify the relations between the moments

# In[6]:


female_heights = [67, 70, 63, 65, 68, 60, 70, 64, 69, 61, 66, 65, 71, 62, 66, 68, 64, 67, 62, 66, 65, 63, 66, 65, 63, 69, 
                  62, 67, 59, 66, 65, 63, 65, 60, 67, 64, 68, 61, 69, 65, 62, 67, 70, 64, 63, 68, 64, 65, 61, 66]

mean_female_heights = np.mean(female_heights)
m2 = moment(female_heights, moment=2)
m3 = moment(female_heights, moment=3)
m4 = moment(female_heights, moment=4)

adjusted_heights = np.array(female_heights) - 75

m_prime_1 = np.mean(adjusted_heights)
m_prime_2 = np.mean((adjusted_heights) ** 2)
m_prime_3 = np.mean((adjusted_heights) ** 3)
m_prime_4 = np.mean((adjusted_heights) ** 4)

# Verification of relations between moments
# a) m_2 = m'_2 - (m'_1)^2
verify_m2 = m_prime_2 - (m_prime_1)**2

# b) m_3 = m'_3 - 3m'_1m'_2 + 2(m'_1)^3
verify_m3 = m_prime_3 - 3 * m_prime_1 * m_prime_2 + 2 * (m_prime_1)**3

# c) m_4 = m'_4 - 4m'_1m'_3 + 6(m'_1)^2 m'_2 - 3(m'_1)^4
verify_m4 = m_prime_4 - 4 * m_prime_1 * m_prime_3 + 6 * (m_prime_1)**2 * m_prime_2 - 3 * (m_prime_1)**4

print("Moments about the mean:")
print(f"  m2 (variance): {m2}")
print(f"  m3: {m3}")
print(f"  m4: {m4}\n")

print("Moments about the number 75:")
print(f"  m'_1: {m_prime_1}")
print(f"  m'_2: {m_prime_2}")
print(f"  m'_3: {m_prime_3}")
print(f"  m'_4: {m_prime_4}\n")

print("Verification of relationships:")
print(f"  a) m_2 = m'_2 - (m'_1)^2: {verify_m2} (should equal {m2})")
print(f"  b) m_3 = m'_3 - 3m'_1 m'_2 + 2(m'_1)^3: {verify_m3} (should equal {m3})")
print(f"  c) m_4 = m'_4 - 4m'_1 m'_3 + 6(m'_1)^2 m'_2 - 3(m'_1)^4: {verify_m4} (should equal {m4})")

