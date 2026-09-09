K_phial_FEM = 3420  # N/mm
K_tot_EXP = 1700    # N/mm

R = K_phial_FEM/K_tot_EXP
K_petg_exp = R / (R-1) * K_tot_EXP
compare_stiffnesses = K_petg_exp / K_phial_FEM

print(f"{'R (stiffness ratio K_phial/K_tot)':<40} = {R:8.3f}")
print(f"{'K_petg_exp (PETG adaptor stiffness)':<40} = {K_petg_exp:8.2f} N/mm")
print(f"{'K_petg_exp / K_phial_FEM':<40} = {compare_stiffnesses:8.3f}")