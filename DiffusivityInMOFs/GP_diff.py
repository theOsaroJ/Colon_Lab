aimport pandas as pd
import numpy as np
import gpflow
from gpflow.utilities import print_summary
import warnings
import os
import json

kernel =  gpflow.kernels.RationalQuadratic() + gpflow.kernels.White() + gpflow.kernels.White() + gpflow.kernels.RationalQuadratic()

warnings.filterwarnings(action='ignore')

# Define custom kernels
class CustomCH4Kernel(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(active_dims=[0, 1, 2])
        self.k1 = gpflow.kernels.RationalQuadratic()
        self.k2 = gpflow.kernels.Matern12()

    def K(self, X, X2=None):
        return self.k1.K(X, X2) * self.k2.K(X, X2) + self.k1.K(X, X2) * self.k2.K(X, X2) + self.k1.K(X, X2) + self.k2.K(X, X2)

    def K_diag(self, X):
        return self.k1.K_diag(X) * self.k2.K_diag(X)

class CustomCO2Kernel(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(active_dims=[0, 1, 2])
        self.k1 = gpflow.kernels.RationalQuadratic()
        self.k2 = gpflow.kernels.Matern12()

    def K(self, X, X2=None):
        return self.k1.K(X, X2) * self.k2.K(X, X2)

    def K_diag(self, X):
        return self.k1.K_diag(X) * self.k2.K_diag(X)
# File to store the first sigma values
sigma_file = 'first_sigma_values_diff.json'

# Function to save the first sigma means to a file
def save_first_sigma_means(sigma_ch4, sigma_co2):
    if not os.path.exists(sigma_file):
        # Only save if the file doesn't exist
        first_sigma_values = {
            'first_sigma_ch4': sigma_ch4.mean().item(),
            'first_sigma_co2': sigma_co2.mean().item()
        }
        with open(sigma_file, 'w') as f:
            json.dump(first_sigma_values, f)
        #print(f"Stored first_sigma_ch4: {first_sigma_values['first_sigma_ch4']}")
        #print(f"Stored first_sigma_co2: {first_sigma_values['first_sigma_co2']}")

# Function to load the first sigma means from a file
def load_first_sigma_means():
    if os.path.exists(sigma_file):
        with open(sigma_file, 'r') as f:
            first_sigma_values = json.load(f)
        return first_sigma_values['first_sigma_ch4'], first_sigma_values['first_sigma_co2']
    else:
        return None, None

# Load data
df = pd.read_csv('Prior.csv', delimiter=',')
df2 = pd.read_csv('NewTest.csv')

# Extract training data
pressure = df.iloc[:, 0].values.reshape(-1, 1)
temp = df.iloc[:, 1].values.reshape(-1, 1)
ch4_x = df.iloc[:, 2].values.reshape(-1, 1)
co2_x = df.iloc[:, 3].values.reshape(-1, 1)
ch4_u = df.iloc[:, 4].values.reshape(-1, 1)
co2_u = df.iloc[:, 5].values.reshape(-1, 1)
ch4_d = df.iloc[:, 6].values.reshape(-1, 1)
co2_d = df.iloc[:, 7].values.reshape(-1, 1)
ch4_as = df.iloc[:, 8].values.reshape(-1, 1)
co2_as = df.iloc[:, 9].values.reshape(-1, 1)
ch4_ds = df.iloc[:, 10].values.reshape(-1, 1)
co2_ds = df.iloc[:, 11].values.reshape(-1, 1)

# Log-transform all variables
pressure = np.log10(pressure)
temp = np.log10(temp)
ch4_x = np.log10(ch4_x)
co2_x = np.log10(co2_x)
ch4_u = np.log10(ch4_u)
co2_u = np.log10(co2_u)
ch4_d = np.log10(ch4_d)
co2_d = np.log10(co2_d)
ch4_as = np.log10(ch4_as)
co2_as = np.log10(co2_as)
ch4_ds = np.log10(ch4_ds)
co2_ds = np.log10(co2_ds)

# Manually calculate mean and standard deviation, then standardize
def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std

# Standardize training data
pressure, pressure_mean, pressure_std = standardize(pressure)
temp, temp_mean, temp_std = standardize(temp)
ch4_x, ch4_x_mean, ch4_x_std = standardize(ch4_x)
co2_x, co2_x_mean, co2_x_std = standardize(co2_x)
ch4_u, ch4_u_mean, ch4_u_std = standardize(ch4_u)
co2_u, co2_u_mean, co2_u_std = standardize(co2_u)
ch4_d, ch4_d_mean, ch4_d_std = standardize(ch4_d)
co2_d, co2_d_mean, co2_d_std = standardize(co2_d)
ch4_as, ch4_as_mean, ch4_as_std = standardize(ch4_as)
co2_as, co2_as_mean, co2_as_std = standardize(co2_as)
ch4_ds, ch4_ds_mean, ch4_ds_std = standardize(ch4_ds)
co2_ds, co2_ds_mean, co2_ds_std = standardize(co2_ds)

y_ch4_std = ch4_d_std
y_co2_std = co2_d_std
y_ch4_m = ch4_d_mean
y_co2_m = co2_d_mean

# Extract test data without transformations
pressure_test = df2.iloc[:, 0].values.reshape(-1, 1)
temp_test = df2.iloc[:, 1].values.reshape(-1, 1)
ch4_x_test = df2.iloc[:, 2].values.reshape(-1, 1)
co2_x_test = df2.iloc[:, 3].values.reshape(-1, 1)
ch4_u_test = df2.iloc[:, 4].values.reshape(-1, 1)
co2_u_test = df2.iloc[:, 5].values.reshape(-1, 1)
ch4_d_test = df2.iloc[:, 6].values.reshape(-1, 1)
co2_d_test = df2.iloc[:, 7].values.reshape(-1, 1)
ch4_as_test = df2.iloc[:, 8].values.reshape(-1, 1)
co2_as_test = df2.iloc[:, 9].values.reshape(-1, 1)
ch4_ds_test = df2.iloc[:, 10].values.reshape(-1, 1)
co2_ds_test = df2.iloc[:, 11].values.reshape(-1, 1)
ch4_up_test = df2.iloc[:, 12].values.reshape(-1, 1)
co2_up_test = df2.iloc[:, 13].values.reshape(-1, 1)

a = pd.DataFrame(ch4_d_test, columns=['Actual'])
a.to_csv('ch4_actual.csv', index=False)

a = pd.DataFrame(co2_d_test, columns=['Actual'])
a.to_csv('co2_actual.csv', index=False)

# Apply log10 transformation to the extracted test data
pressure_test = np.log10(pressure_test)
temp_test = np.log10(temp_test)
ch4_x_test = np.log10(ch4_x_test)
co2_x_test = np.log10(co2_x_test)
ch4_u_test = np.log10(ch4_u_test)
co2_u_test = np.log10(co2_u_test)
ch4_d_test = np.log10(ch4_d_test)
co2_d_test = np.log10(co2_d_test)
ch4_as_test = np.log10(ch4_as_test)
co2_as_test = np.log10(co2_as_test)
ch4_ds_test = np.log10(ch4_ds_test)
co2_ds_test = np.log10(co2_ds_test)
ch4_up_test = np.log10(ch4_up_test)
co2_up_test = np.log10(co2_up_test)


# Standardize test data using training set statistics
def apply_standardization(data, mean, std):
    return (data - mean) / std

pressure_test = apply_standardization(pressure_test, pressure_mean, pressure_std)
temp_test = apply_standardization(temp_test, temp_mean, temp_std)
ch4_x_test = apply_standardization(ch4_x_test, ch4_x_mean, ch4_x_std)
co2_x_test = apply_standardization(co2_x_test, co2_x_mean, co2_x_std)
ch4_u_test = apply_standardization(ch4_u_test, ch4_u_mean, ch4_u_std)
co2_u_test = apply_standardization(co2_u_test, co2_u_mean, co2_u_std)
ch4_d_test = apply_standardization(ch4_d_test, ch4_d_mean, ch4_d_std)
co2_d_test = apply_standardization(co2_d_test, co2_d_mean, co2_d_std)
ch4_as_test = apply_standardization(ch4_as_test, ch4_as_mean, ch4_as_std)
co2_as_test = apply_standardization(co2_as_test, co2_as_mean, co2_as_std)
ch4_ds_test = apply_standardization(ch4_ds_test, ch4_ds_mean, ch4_ds_std)
co2_ds_test = apply_standardization(co2_ds_test, co2_ds_mean, co2_ds_std)
ch4_up_test = apply_standardization(ch4_u_test, ch4_u_mean, ch4_u_std)
co2_up_test = apply_standardization(co2_u_test, co2_u_mean, co2_u_std)


# Combine features into a single array for training and testing
x_ch4_s = np.hstack((pressure, temp, ch4_u))  # Shape: [N, 3]
x_co2_s = np.hstack((pressure, temp, co2_u))  # Shape: [N, 3]

X_ch4_test = np.hstack((pressure_test, temp_test, ch4_up_test))  # Shape: [N, 3]
X_co2_test = np.hstack((pressure_test, temp_test, co2_up_test))  # Shape: [N, 3]

# Process target data
y_ch4 = ch4_d
y_co2 = co2_d

# Define custom kernels
kernel_ch4 = kernel
kernel_co2 = kernel

# Initialize GP models with the correctly shaped data
model_ch4 = gpflow.models.GPR(
    data=(x_ch4_s, y_ch4),
    kernel=kernel_ch4,
    noise_variance=1e-5
)

model_co2 = gpflow.models.GPR(
    data=(x_co2_s, y_co2),
    kernel=kernel_co2,
    noise_variance=1e-5
)

# Set likelihood variance to be trainable
gpflow.utilities.set_trainable(model_ch4.likelihood.variance, True)
gpflow.utilities.set_trainable(model_co2.likelihood.variance, True)

# Optimize models using Scipy
optimizer = gpflow.optimizers.Scipy()
# Continue model optimization
optimizer.minimize(
    model_ch4.training_loss, 
    model_ch4.trainable_variables, 
    options=dict(maxiter=3000000, ftol=1e-9), 
    method="L-BFGS-B"
)

optimizer.minimize(
    model_co2.training_loss, 
    model_co2.trainable_variables, 
    options=dict(maxiter=3000000, ftol=1e-9), 
    method="L-BFGS-B"
)

# Make predictions on the test data
y_ch4_pred, var_ch4 = model_ch4.predict_f(X_ch4_test)
y_co2_pred, var_co2 = model_co2.predict_f(X_co2_test)

# Convert predictions to numpy arrays
y_ch4_pred = y_ch4_pred.numpy()
y_co2_pred = y_co2_pred.numpy()

# Inverse transformation of predictions
# Scale back the predictions using the mean and std from training
y_ch4_pred = (y_ch4_pred * y_ch4_std) + y_ch4_m
y_co2_pred = (y_co2_pred * y_co2_std) + y_co2_m

# Convert back from log scale to original scale
y_ch4_pred = 10**y_ch4_pred
y_co2_pred = 10**y_co2_pred

# Convert variances back to the original scale by multiplying with the squared standard deviation
var_ch4 = var_ch4.numpy()  
var_co2 = var_co2.numpy() 

# Handle invalid values in variances (replace NaNs, zeros, or negatives)
var_ch4 = np.where((var_ch4 <= 0) | np.isnan(var_ch4), 0.00002, var_ch4)
var_co2 = np.where((var_co2 <= 0) | np.isnan(var_co2), 0.00002, var_co2)

# Scale back variances
var_ch4 = 10**(var_ch4 * (y_ch4_std ** 2))
var_co2 = 10**(var_co2 * (y_co2_std ** 2))

# Calculate standard deviation (sigma) from variance
sigma_ch4 = np.sqrt(var_ch4)
sigma_co2 = np.sqrt(var_co2)

# Identify the index of the highest uncertainty
index_ch4 = np.argmax(sigma_ch4)
index_co2 = np.argmax(sigma_co2)

# Save the first sigma means only once using the function
save_first_sigma_means(sigma_ch4, sigma_co2)

# Load the first sigma means for comparison
first_sigma_ch4, first_sigma_co2 = load_first_sigma_means()

# Set thresholds to compare uncertainties
ninety_first_sigma_ch4 = 0.2 * first_sigma_ch4
ninety_first_sigma_co2 = 0.2 * first_sigma_co2

# Determine which gas has more uncertainty
if sigma_ch4.mean() > ninety_first_sigma_ch4 or sigma_co2.mean() > ninety_first_sigma_co2:
    if sigma_ch4.mean() > sigma_co2.mean():
        print(sigma_ch4.mean())
        print(sigma_co2.mean())
        print("CH4 more uncertain")
        print("Not_Done")
        index = index_ch4
    else:
        print(sigma_ch4.mean())
        print(sigma_co2.mean())
        print("CO2 more uncertain")
        print("Not_Done")
        index = index_co2
else:
    print("Done")

# Output predictions
pred_ch4 = y_ch4_pred
pred_co2 = y_co2_pred

# Save the final predicted data
pd.DataFrame(pred_ch4, columns=['Predicted']).to_csv('pred_ch4_diff.csv', index=False)
pd.DataFrame(pred_co2, columns=['Predicted']).to_csv('pred_co2_diff.csv', index=False)

# Looking up the data to add to Prior
df_lookup = pd.read_csv('NewTest.csv')

# Extract relevant data for display
pressure = df_lookup.iloc[:, 0].values
temp = df_lookup.iloc[:, 1].values
ch4_x = df_lookup.iloc[:, 2].values
co2_x = df_lookup.iloc[:, 3].values
ch4_u = df_lookup.iloc[:, 4].values
co2_u = df_lookup.iloc[:, 5].values
ch4_d = df_lookup.iloc[:, 6].values
co2_d = df_lookup.iloc[:, 7].values
ch4_as = df_lookup.iloc[:, 8].values
co2_as = df_lookup.iloc[:, 9].values
ch4_ds = df_lookup.iloc[:, 10].values
co2_ds = df_lookup.iloc[:, 11].values

# Print selected data based on identified uncertainty index
print(
    pressure[index], temp[index], ch4_x[index], co2_x[index], 
    ch4_u[index], co2_u[index], ch4_d[index], co2_d[index],
    ch4_as[index], co2_as[index], ch4_ds[index], co2_ds[index]
)

# Remove the row at the identified index from the TestData
df2.drop(index, inplace=True)

# Save the updated TestData.csv
df2.to_csv('tmp.csv', index=False)
