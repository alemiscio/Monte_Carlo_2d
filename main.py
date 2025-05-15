import numpy as np

from Monte_Carlo_functions import  random_spin_field, ising_step, magnetization, plot_Thermal_two_point_r_sigma, plot_Thermal_two_point_t_sigma


# --- Main simulation ---

N, M = 40, 5000 #Time+ Space
beta = 0.44068679
h = 0.0

n_steps = 6000
n_steps_min = 2000 - 1

for hh in range(4,20):

    for j in range(2):
        print(f"Simulation number {j+1} started...")
        field = random_spin_field(N, M)
        #field = balanced_spin_field(N,M)
        mags = []
        #print(f"Step {0}, Magnetization: {magn2etization(field):.6f}")

        for i in range(n_steps+1):
            field = ising_step(field, beta,1/hh)
            #print(i,sep=" ", end=" ", flush=True)
            #print(f"Step {i}, Magnetization: {magnetization(field):.6f}")
            mags.append(magnetization(field))
            if i % 100  == 0 and i > n_steps_min :
                print(f"Step {i}, Magnetization: {magnetization(field):.6f}")
                #img = display_spin_field(field)
                #img.show()
                #plot_two_point_sigma(field, 50)
                plot_Thermal_two_point_r_sigma(field,60,40,f"Ising_correlator_r_beta_40.png",i)
                plot_Thermal_two_point_t_sigma(field, 39, 40, f"Ising_correlator_t_beta_40.png", i)
                np.savetxt("field_config_{hhh}_{jj}_{ii}.txt".format(ii = i,jj = j, hhh = hh), field, fmt="%d")
            #plt.clf()
        #plot_magnetization(mags)