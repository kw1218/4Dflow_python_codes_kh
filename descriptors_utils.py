import numpy as np
import pyvista as pv
import math


## for flow morphology

def normvec(v):
    return v / np.linalg.norm(v)

def compute_flowrate(vtps):
    flowRate = []
    for i in range(len(vtps)):
        dummyPD = vtps[0]
        normal = dummyPD.compute_normals()['Normals'].mean(0)
        dummyPD['Velocity'] = vtps[i]['Velocity']
        dummyPD = dummyPD.point_data_to_cell_data(pass_point_data=True)
        Q = np.sum(np.dot(dummyPD['Velocity'], normal) * dummyPD.compute_cell_sizes()['Area'])
        flowRate.append(Q)
    flowRate = np.array(flowRate)
    if flowRate[np.argmax(np.abs(flowRate))] < 0:
        flowRate *= -1

    out = {'Q(t)': flowRate, 'Q_mean': np.mean(flowRate), 'Q_max': np.max(flowRate)}
    return out

def compute_retrograde_flow_fraction(flowRate):
    retro_mask = np.zeros(len(flowRate))
    retro_mask[np.where(flowRate < 0)] = 1
    fwd_mask = np.zeros(len(flowRate))
    fwd_mask[np.where(flowRate > 0)] = 1

    Q_r, Q_f = flowRate * retro_mask, flowRate * fwd_mask
    rff_t = np.abs(np.trapz(Q_r)) / (np.abs(np.trapz(Q_f)) + np.abs(np.trapz(Q_r))) * 100

    out = {'rfi': rff_t}
    return out


def compute_positive_peak_velocity(profs):
    # peak positive velocity
    peak_vels = [np.max(np.linalg.norm(profs[k]['Velocity'], axis=1)) for k in range(len(profs))]
    max_peak_vel = np.max(peak_vels)
    out = {'ppv(t)': peak_vels, 'ppv_max': max_peak_vel, 'ppv_systole': peak_vels[np.argmax(compute_flowrate(profs)['Q(t)'])], 'ppv_mean': np.mean(peak_vels)}
    return out

def compute_flow_displacement(profs):
    # plane diameter
    areas = [profs[k].compute_cell_sizes()['Area'].sum() for k in range(len(profs))]
    perims = [profs[k].extract_feature_edges().connectivity(largest=True).compute_arc_length()['arc_length'].sum() for k
              in range(len(profs))]
    diams = [4 * areas[k] / perims[k] for k in range(len(profs))]
    # flow displacement
    max_vel_coord = [profs[k].points[np.argmax(np.linalg.norm(profs[k]['Velocity'], axis=1))] for k in range(len(profs))]
    center_coord = [profs[k].points.mean(0) for k in range(len(profs))]
    fdm = [100 * (np.linalg.norm(max_vel_coord[k] - center_coord[k]) / (0.5 * diams[3])) for k in range(len(profs))]
    out = {'fdm(t)': fdm, 'fdm_systole': fdm[np.argmax(compute_flowrate(profs)['Q(t)'])], 'fdm_mean': np.mean(fdm), 'fdm_max': np.max(fdm)}
    return out

def compute_flow_dispersion(profs):
    areas = [profs[k].compute_cell_sizes()['Area'].sum() for k in range(len(profs))]
    peak_vels = [np.max(np.linalg.norm(profs[k]['Velocity'], axis=1)) for k in range(len(profs))]

    # flow dispersion
    cell_pds = [profs[k].point_data_to_cell_data(pass_point_data=True) for k in range(len(profs))]
    top15_idx = [np.where(np.linalg.norm(cell_pds[k].cell_data['Velocity'], axis=1) > 0.75 * peak_vels[k]) for k in
                 range(len(profs))]
    top15_areas = [cell_pds[k].compute_cell_sizes()['Area'][top15_idx[k]].sum() for k in range(len(profs))]
    fdi = [100 * top15_areas[k] / areas[k] for k in range(len(profs))]
    out = {'fdi(t)': fdi, 'fdi_systole': fdi[np.argmax(compute_flowrate(profs)['Q(t)'])], 'fdi_mean': np.mean(fdi), 'fdi_max': np.max(fdi)}
    return out


def compute_flow_jet_angle(profs):
    # flow jet angle
    normal = np.abs(profs[0].compute_normals()['Normals'].mean(0) * -1)
    #mean_vels = [np.array(np.mean(profs[k]['Velocity'], axis=0))**2 for k in range(len(profs))]
    mean_vels = [normvec(np.array(np.mean(profs[k]['Velocity'], axis=0))) for k in range(len(profs))]
    fja = [np.rad2deg(math.acos(np.dot(mean_vels[k], normal))) for k in range(len(profs))]
    fja = np.array(fja)
    indices = np.where(fja<90)[0]
    fja_systole = fja[indices]
    out = {'fja(t)': fja, 'fja_systole': fja[np.argmax(compute_flowrate(profs)['Q(t)'])], 'fja_max': np.max(fja), 'fja_mean': np.mean(fja), 'fja_systole_mean': np.mean(fja_systole)}
    return out

def compute_flow_jet_angle_kh(profiles):
    """
    Calculate the flow jet angle for a series of velocity profiles.
    
    Args:
        profiles (list): List of profiles containing 'Velocity' vectors for each cross-section.

    Returns:
        dict: A dictionary containing flow jet angles over time, the maximum angle, the mean angle, and the angle during systole.
    """
    # Extract normal vector from the first profile
    normal_vector = np.mean(profiles[0].compute_normals()['Normals'], axis=0)*-1
    normal_unit_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the normal vector

    # Calculate the flow jet angles for each profile
    flow_jet_angles = []
    for profile in profiles:
        # Calculate the mean velocity vector and normalize it
        mean_velocity_vector = np.mean(profile['Velocity'], axis=0)
        mean_velocity_unit_vector = mean_velocity_vector / np.linalg.norm(mean_velocity_vector)
        
        # Calculate the angle using the dot product
        cosine_angle = np.dot(mean_velocity_unit_vector, normal_unit_vector)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip the cosine value to avoid numerical issues
        flow_jet_angles.append(np.degrees(angle))  # Convert to degrees

    # Find the angle during systole (assuming systole corresponds to maximum flow rate)
    flow_rates =compute_flowrate(profiles)['Q(t)']
    systole_index = np.argmax(flow_rates)
    
    # Create a dictionary with the calculated values
    results = {
        'fja(t)': flow_jet_angles,
        'fja_systole': flow_jet_angles[systole_index],
        'fja_max': np.max(flow_jet_angles),
        'fja_mean': np.mean(flow_jet_angles)
    }
    
    return results

# secondary flow degree
def compute_secondary_flow_degree(profs):
    nA = np.array([0, 0, 1])
    normal = np.abs(profs[0].compute_normals()['Normals'].mean(0)*-1)
    mean_vecs = [np.mean(profs[k]['Velocity'], 0) for k in range(len(profs))]
    normal_vecs = [np.abs(np.dot(mean_vecs[k], normal) * nA) for k in range(len(profs))]
    normal_vec_norms = [np.linalg.norm(normal_vecs[k]) for k in range(len(profs))]
    parallel_vecs = [mean_vecs[k] - normal_vecs[k] for k in range(len(profs))]
    parallel_vec_norms = [np.linalg.norm(parallel_vecs[k]) for k in range(len(profs))]
    sfd = [parallel_vec_norms[k] / normal_vec_norms[k] for k in range(len(profs))]
    out = {'sfd(t)': sfd, 'sfd_systole': sfd[np.argmax(compute_flowrate(profs)['Q(t)'])], 'sfd_mean': np.mean(sfd), 'sfd_max': np.max(sfd)}
    return out


def compute_flow_descriptors(profs):
    Q = compute_flowrate(profs)
    rfi = compute_retrograde_flow_fraction(Q['Q(t)'])
    ppv = compute_positive_peak_velocity(profs)
    fdm = compute_flow_displacement(profs)
    fdi = compute_flow_dispersion(profs)
    fja = compute_flow_jet_angle(profs)
    sfd = compute_secondary_flow_degree(profs)
    hfi = compute_helical_flow_index(profs)
    all_feats = [Q, rfi, ppv, fdm, fdi, fja, sfd, hfi]
    descriptors = {}
    for d in all_feats:
        descriptors.update(d)
    return descriptors


def compute_helical_flow_index(profs):
    Q = compute_flowrate(profs)
    ps = np.argmax(Q)
    vort = [profs[k].compute_derivative(vorticity=True) for k in range(len(profs))]
    #HFI_sum = [np.sum(vort[k]['Velocity'] * vort[k]['vorticity']) for k in range(len(profs))]
    HFI_mean = [100 * np.mean(np.abs(np.sum(vort[k]['Velocity'] * vort[k]['vorticity'], 1))) for k in range(len(profs))]
    HFI_systole = HFI_mean[ps]
    out = {'hfi(t)': HFI_mean, 'hfi_systole': HFI_systole, 'hfi_mean': np.mean(HFI_mean), 'hfi_max': np.max(HFI_mean)}
    return out



