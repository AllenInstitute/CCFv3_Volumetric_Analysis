from __future__ import division, print_function, absolute_import
import math
import numpy as np
import SimpleITK as sitk
import nrrd
import scipy
from scipy.interpolate import RegularGridInterpolator as RegularGridInterpolator
from structure_meshes import get_normals, get_midpoints, get_volume, apply_composite_transformation
import pickle
import pandas as pd
import argparse
import h5py
import sqlite3
import time





# This function calls the other functions given ID and finds volume
def calculate_volume():

    # Get deformation field 
    dfmfld = mcc.get_deformation_field(ID,header_path='./{}/dfmfld.mhd'.format(ID),voxel_path='./{}/dfmfld.raw'.format(ID))

    # Create interpolator
    axes = [np.arange(x) * spacing[ii] for ii, x in enumerate(dfmfld.shape[:-1])]
    interpolator = RegularGridInterpolator(axes,dfmfld)

    # Get affine parameters 
    aff_params = mcc.get_affine_parameters(ID,direction='trv',file_name='./{}/aff_param.txt'.format(ID))
    aff_params = aff_params.T
    aff_params = np.vstack((aff_params,np.array([0,0,0,1])[None,:]))


    # List for structure volumes
    struct_vols = []

    # Find volume of each structure
    for struct in hemi_structs:
        try:
            # print('finding volume')
            with h5py.File('./data/surface_meshes/vertices/{}.hdf5'.format(struct)) as f:
                vertices = f['vertices'][()]
            with h5py.File('./data/surface_meshes/faces/{}.hdf5'.format(struct)) as f:
                faces = f['faces'][()]

            # dis_verts = apply_composite_transformation(nimg,aff_params,spacing,vertices)
            normals = get_normals(vertices,faces)
            midpoints = get_midpoints(vertices,faces)
            volume = get_volume(midpoints,normals,vertices,faces)

            struct_vols.append({'Structure_ID': struct,\
             'Volume': volume})
        except:
            pass

    for struct in medi_structs:
        try:
            # print('finding volume')
            with h5py.File('./data/surface_meshes/vertices/{}.hdf5'.format(struct)) as f:
                vertices = f['vertices'][()]
            with h5py.File('./data/surface_meshes/faces/{}.hdf5'.format(struct)) as f:
                faces = f['faces'][()]

            # dis_verts = apply_composite_transformation(nimg,aff_params,spacing,vertices)
            normals = get_normals(vertices,faces)
            midpoints = get_midpoints(vertices,faces)
            volume = get_volume(midpoints,normals,vertices,faces)

            struct_vols.append({'Structure_ID': struct,\
             'Volume': volume})
            
        except:
            pass

    dfmfld = np.flip(dfmfld,axis=0)

    # Create interpolator
    axes = [np.arange(x) * spacing[ii] for ii, x in enumerate(dfmfld.shape[:-1])]
    interpolator = RegularGridInterpolator(axes,dfmfld)

    # Find volume of each structure
    for struct in hemi_structs:
        try:
            # print('finding volume')
            with h5py.File('./data/surface_meshes/vertices/{}.hdf5'.format(struct)) as f:
                vertices = f['vertices'][()]
            with h5py.File('./data/surface_meshes/faces/{}.hdf5'.format(struct)) as f:
                faces = f['faces'][()]

            # normals_reg = get_normals(vertices,faces)
            # surface_area_reg = get_surfaceArea(normals_reg)

            dis_verts = apply_composite_transformation(dfmfld,aff_params,spacing,vertices)
            normals = get_normals(dis_verts,faces)
            surface_area = get_surfaceArea(normals)
            

            struct_vols.append({'Image_Series_ID': ID, 'Structure_ID': struct,\
             'Surface_Area': surface_area, 'Side': 'r'})
        except:
            pass



        
    # store volumes   
    conn = sqlite3.connect("structure_analysis_float.db") 

    df = pd.DataFrame(struct_vols)
    save = True
    count = 0
    while(save):
        count += 1
        try:
            df.to_sql('volumes',conn,if_exists='append')
            save = False
        except:
            if count > 20:
                break
            time.sleep(1)

    conn.close()



# Get ID from input args         
parser = argparse.ArgumentParser()
parser.add_argument('id', type = int, help='experiment ID')
args = parser.parse_args()
ID = args.id

# Structure IDs
# These are the structures we are interested in
with open('non_crossing_structures') as f:
    hemi_structs = pickle.load(f)
with open('mid_crossing_structures') as f:
    medi_structs = pickle.load(f)

mcc = MouseConnectivityCache(resolution=25)

spacing = (25,25,25)


if not os.path.exists('./{}/'.format(ID)):
    os.mkdir('./{}/'.format(ID))


calculate_volume(ID)