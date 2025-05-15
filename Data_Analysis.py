import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

plt.ion()

from Monte_Carlo_functions import   magnetization,plot_Thermal_two_point_r_sigma, plot_Thermal_two_point_t_sigma, display_spin_field, export_two_point_data_txt,compute_two_point_t_profile,compute_two_point_epsilon_r_profile, compute_two_point_epsilon_t_profile,compute_two_point_r_profile

field1 = np.loadtxt("Results/40x2000_field_configurations/field_config_{jj}_{ii}.txt".format(jj = 1,ii = 2000), dtype=int)
#display_spin_field(field1).show()

j = 1
mag = []
for i in range(2000, 7000, 100):
    field1 = np.loadtxt("Results/40x2000_field_configurations/field_config_{jj}_{ii}.txt".format(jj=j, ii=i), dtype=int)
    #display_spin_field(field1).show()
    #plot_Thermal_two_point_r_sigma(field1, 60, 40, f"test.png")
    #plot_Thermal_two_point_t_sigma(field1, 39, 40, f"testt.png")
    #export_two_point_data_txt(compute_two_point_epsilon_t_profile(field1, 39), "Results/40x2000_field_configurations/tp_sigma_data/tp_e_t_{jj}_{ii}.txt".format(jj=j, ii=i))
    mag.append(magnetization(field1))

print("Magnetization : ", np.mean(mag))