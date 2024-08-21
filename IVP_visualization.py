import sys
import os
import os.path as osp
import numpy as np
from glob import glob
import pyvista as pv
import descriptors_utils as dut
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------------------------------------------------
# rootDir=r'D:/InletProfileStudy/VTPfiles/2024_root'  # path to synthetic files or sampled profiles
# Fig_Dir = r'D:/InletProfileStudy/SSM/Output_2024/SINE_P5/'
# IVP_number = 'TBAD25'
# IVP_path = osp.join(rootDir,IVP_number)
# vtp_path = osp.join(IVP_path, '*.vtp')


rootDir=r'D:\InletProfileStudy\SSM\Output_2024\SH_P5\sampled_profiles\prof_mode2_coeff_0.5'
vtp_path = osp.join(rootDir, '*.vtp')

# -----------------------------------------------------------------------------------------------------------------------
## Read synthetic profiles

input_vtps = pv.read(sorted(glob((vtp_path))))



# ## Flow rate
# # Compute flow rate and convert to l/min
# flow_rate = dut.compute_flowrate(input_vtps)['Q(t)'] * 60000  # l/min

# # Plot the flow rate
# plt.figure(figsize=(10, 6))
# plt.plot(flow_rate, label='Flow Rate (l/min)', color='b', marker='o')

# # Add labels and title
# plt.xlabel('Time (frames)')
# plt.ylabel('Flow Rate (l/min)')
# plt.title('Flow Rate Over Time')
# plt.legend()

# # Show the plot
# plt.show()
#plt.savefig(osp.join(Fig_Dir, 'flow_rate.png'))


pl = pv.Plotter()
pl.add_mesh(input_vtps[3], clim=[0,1], scalars='Velocity', cmap='jet')
#pl.add_mesh(interp_planes[3], clim=[0,1.4], scalars='Velocity', cmap='jet')
pl.camera_position = 'xy'
pl.camera.zoom(1.3)
pl.remove_scalar_bar()
pl.show()

# plotter = pv.Plotter()
# plotter.add_mesh(input_vtps[5].warp_by_vector(factor=0.06), scalars='Velocity', clim=[0, 0.2])
# plotter.show()




