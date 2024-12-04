import pandas as pd
import numpy as np

# Load Excel file
file_path = "hist.xlsx"  # Replace with your file path
pre = pd.read_excel(file_path, sheet_name='pre')
temp = pd.read_excel(file_path, sheet_name='temp')
pop = pd.read_excel(file_path, sheet_name='pop')
gpcd = pd.read_excel(file_path, sheet_name='gpcd')

# Set constants
start_year = 1985
end_year = 2015

# 1. Preprocessing Population Data
# Interpolate yearly population from the 'pop' sheet
pop = pop.set_index('Year')  # Ensure YEAR is the index
pop_interpolated = pop.reindex(range(start_year, end_year + 1))
pop_interpolated['Population'] = pop_interpolated['Population'].interpolate(method='linear')

# 2. Filter GPCD Data for the Time Range
gpcd = gpcd[(gpcd['year'] >= start_year) & (gpcd['year'] <= end_year)]
gpcd = gpcd.set_index('year')

# 3. Calculate Annual GPD
# Annual GPD = GPCD * Population * 365
annual_gpd = gpcd['GPCD'] * pop_interpolated['Population'] * 365
annual_gpd.name = 'ANNUAL_GPD'

# 4. Precipitation Data - Calculate Monthly Fractions
# Filter precipitation for the time range
pre = pre[(pre['YEAR'] >= start_year) & (pre['YEAR'] <= end_year)]
pre.set_index('YEAR', inplace=True)

# Calculate monthly precipitation fractions
monthly_precip = pre.loc[:, 'JAN':'DEC']
annual_precip = pre['ANNUAL']
monthly_fractions = monthly_precip.div(annual_precip, axis=0)

# 5. Estimate Monthly GPD
monthly_gpd = monthly_fractions.mul(annual_gpd, axis=0)

# 6. Calculate Monthly GPCD
# Monthly GPCD = Monthly GPD / Population
monthly_gpcd = monthly_gpd.div(pop_interpolated['Population'], axis=0)

# 7. Save Results to Excel
output_path = "gpcd_gpd_results.xlsx"
with pd.ExcelWriter(output_path) as writer:
    annual_gpd.to_frame().to_excel(writer, sheet_name='Annual GPD')
    monthly_gpd.to_excel(writer, sheet_name='Monthly GPD')
    monthly_gpcd.to_excel(writer, sheet_name='Monthly GPCD')

print(f"Results saved to {output_path}")
