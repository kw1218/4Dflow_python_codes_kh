import sys
import os
import os.path as osp
import numpy as np
from glob import glob
import pyvista as pv
import descriptors_utils as dut
from matplotlib import pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------------------------------------------------
rootDir=r'D:\InletProfileStudy\SSM\Output_2024\Circular\synthetic_cohort_first8modes'  # path to synthetic files or sampled profiles
Out_Dir = r'D:/InletProfileStudy/SSM/Input/flow_csv'
IVP_number = '100'
IVP_path = osp.join(rootDir,IVP_number)
vtp_path = osp.join(IVP_path, '*.vtp')
Output_csv = False
os.makedirs(Out_Dir, exist_ok=True)


# rootDir=r'D:\InletProfileStudy\ReadandScale\scaling\HM_02_syn09'
# vtp_path = osp.join(rootDir, '*.vtp')

# -----------------------------------------------------------------------------------------------------------------------
## Read synthetic profiles

input_vtps = pv.read(sorted(glob((vtp_path))))



## Flow rate
# Compute flow rate and convert to l/min
#flow_rate = dut.compute_flowrate(input_vtps)['Q(t)'] * 60000  # l/min
flow_rate = dut.compute_flowrate(input_vtps)['Q(t)'] * 20 
max_flow_rate = np.max(flow_rate)
frame_max_flow_rate = np.argmax(flow_rate)

# Plot the flow rate
plt.figure(figsize=(10, 6))
plt.plot(flow_rate, label='Flow Rate (l/min)', color='b', marker='o')


# Write out flow rate to a csv file
if Output_csv:
    flow_rate_ls =  flow_rate / 60000  # convert to l/s 
    df_flow = pd.DataFrame(flow_rate_ls)
    df_flow.to_csv(osp.join(Out_Dir, 'flow_rate.csv'), index=False)



# Add labels and title
plt.xlabel('Time (frames)')
plt.ylabel('Flow Rate (l/min)')
plt.title('Flow Rate Over Time')
plt.legend()

# # Show the plot
plt.show()
#plt.savefig(osp.join(Fig_Dir, 'flow_rate.png'))


pl = pv.Plotter()
pl.add_mesh(input_vtps[int(frame_max_flow_rate)], clim=[0,0.8], scalars='Velocity', cmap='jet')
#pl.add_mesh(interp_planes[3], clim=[0,1.4], scalars='Velocity', cmap='jet')
pl.camera_position = 'xy'
pl.camera.zoom(1.3)
pl.remove_scalar_bar()
pl.set_background('white')
pl.show_axes()
pl.show()

plotter = pv.Plotter()
plotter.add_mesh(input_vtps[int(frame_max_flow_rate)].warp_by_vector(factor=0.5), scalars='Velocity',clim=[0.1, 0.8], cmap='jet')
scalar_bar = plotter.add_scalar_bar(title='Velocity', n_labels=2,shadow=True, italic=True)
scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)  # Set title color (black)
scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)  # Set label color (black)
plotter.camera_position = 'xy'
plotter.set_background('white')
plotter.show_axes()  # Show the axes
plotter.show()




