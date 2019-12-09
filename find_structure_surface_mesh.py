from __future__ import division, print_function, absolute_import
import numpy as np
import nrrd
from skimage.measure import marching_cubes_classic, correct_mesh_orientation
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache, MouseConnectivityApi
import h5py
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('id', type = int, help='Structure ID')
parser.add_argument('vout', type = str, help='Vertices Directory')
parser.add_argument('fout', type = str, help='Faces Directory')
args = parser.parse_args()

# Get surface of 3D structure
def get_surface(mask, spacing):
    vertices,faces = marching_cubes_classic(mask,level=0,spacing = spacing)
    faces = correct_mesh_orientation(mask,vertices,faces,spacing)
    return vertices, faces

# Load reference space
mcc = MouseConnectivityCache(ccf_version=MouseConnectivityApi.CCF_2017,resolution=25)
spacing = (25,25,25)
rsp = mcc.get_reference_space()

# Structure IDs
struct = args.id

# Get output directories
vertices_output_dir = args.vout
faces_output_dir = args.fout

# List of structures
structure_list = h5py.File('./data/structure_ids.hdf5','r')
midline_crossing_structures = structure_list.get('midline_crossing_structures')[()]
remaining_structures = structure_list.get('non_crossing_structures')[()]
structure_list.close()

# Check if structure is a midline structure
if struct in midline_crossing_structures:
    status = 1
elif struct in remaining_structures:
    status = 2
else:
    status = 0
    print('{} is not in the list of structre IDs'.format(struct))
del midline_crossing_structures, remaining_structures


# Find structure mesh
if status == 1:
    try:
        # Make mask for sturcture
        mask = rsp.make_structure_mask([struct])
        mask.swapaxes(0,2)

        volume = 0.0

        # if mask exists, get vertices,faces
        if mask.max() > 0:
            vertices,faces = get_surface(mask,spacing)
    except:
        pass


elif status == 2:
    try:
        # load hemisphere mask
        hemisphere_mask,meta = nrrd.read('./data/hemisphere_mask.nrrd')
        hemisphere_mask = hemisphere_mask.swapaxes(0,2)

        # Make mask for each structure
        mask = rsp.make_structure_mask([struct])
        mask = mask.swapaxes(0,2)
        mask = mask*hemisphere_mask
        mask = mask.astype(float)
        volume = 0.0

        # if mask exists, get vertices,faces
        if mask.max() > 0:
            vertices,faces = get_surface(mask,spacing)
    except:
        pass


# Save results
if len(vertices)>0:
    vout = h5py.File(os.path.join(vertices_output_dir,'{}.hdf5'.format(struct)),'w')
    vout.create_dataset('vertices',data=vertices);
    vout.close()

    fout = h5py.File(os.path.join(faces_output_dir,'{}.hdf5'.format(struct)),'w')
    fout.create_dataset('faces',data=faces)
    fout.close()