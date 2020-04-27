from __future__ import division, print_function, absolute_import
import math

from skimage.measure import marching_cubes_classic, correct_mesh_orientation
import numpy as np


def get_surface(mask, spacing, level=0):
    '''Obtain faces describing a surface
    
    Parameters
    ----------
    mask : numpy ndarray
        3d array.
    spacing : tuple of numeric
        resolution of mask in microns.
    level : numeric, optional
        Value of the the isosurface to be approximated.
        
    Returns
    -------
    vertices : numpy ndarray
        num_vertices X num_dimensions. Corners of triangles that define faces.
    faces :  numpy ndarray of int
        num_faces X num_vertices_per_face. Values index into vertices.
    
    '''
    
    vertices, faces = marching_cubes_classic(mask, level=level, spacing=spacing)
    faces = correct_mesh_orientation(mask, vertices, faces, spacing=spacing)
    
    return vertices, faces


def get_normals(vertices, faces):
    '''Obtain unit vectors normal to a collection of surface faces
    
    Parameters
    ----------
    vertices : numpy ndarray
        num_vertices X num_dimensions. Corners of triangles that define faces.
    faces :  numpy ndarray of int
        num_faces X num_vertices_per_face. Values index into vertices.
        
    Returns
    -------
    numpy ndarray : 
        num_faces X num_dimensions. Normal vectors (not necessarily unit 
        length).
        
    Notes
    -----
    If you got the faces directly from get_surface the normal vectors will be 
    oriented outwards. Otherwise, no guarantees.
    
    '''
    
    first_vectors = vertices[faces][:, 1, :] - vertices[faces][:, 0, :]
    second_vectors = vertices[faces][:, 2, :] - vertices[faces][:, 0, :]
    
    return np.cross(second_vectors, first_vectors)



def get_midpoints(vertices, faces):
    '''Obtain midpoints of surface faces
    
    Parameters
    ----------
    vertices : numpy ndarray
        num_vertices X num_dimensions. Corners of triangles that define faces.
    faces :  numpy ndarray of int
        num_faces X num_vertices_per_face. Values index into vertices.
        
    Returns
    -------
    numpy ndarray : 
        num_faces X num_dimensions. Coordinates of midpoints.
    
    '''

    return vertices[faces].mean(axis=1).squeeze()


def get_volume(midpoints, normals, vertices, faces):
    '''Obtain the volume of a mask by summing volumes of tetrahedra drawn from 
    its faces.
    
    midpoints : numpy ndarray
        Each row describes the center point of a face.
    normals : numpy ndarray
        Each row corresponds to a face of the mask and describes that face's 
        outward-pointing unit normal vector.
    vertices : numpy ndarray
        num_vertices X num_dimensions. Corners of triangles that define faces.
    faces :  numpy ndarray of int
        num_faces X num_vertices_per_face. Values index into vertices.
        
    Returns
    -------
    volume : float
        Volume of mask
    
    '''

    num_faces, num_dimensions = midpoints.shape

    signs = np.sign(np.multiply(midpoints, normals).sum(axis=1))
    
    pos_dets = np.fabs(np.linalg.det(vertices[faces][signs > 0, :, :])).sum()
    neg_dets = np.fabs(np.linalg.det(vertices[faces][signs < 0, :, :])).sum()
    
    return np.fabs(pos_dets - neg_dets) / math.factorial(num_dimensions)
    
    
def build_interpolator(displacement_field, spacing):
    '''Constructs an interpolator from a displacement field
    
    Parameters
    ----------
    displacement_field : numpy ndarray
        Array of vectors denoting a local displacement transformation.
    spacing : tuple of numeric
        Grid resolution in microns.
        
    Returns
    -------
    RegularGridInterpolator : 
        When called with an array of points -> returns linearly interpolated 
        displacements originating from those points.
    
    '''

    axes = [np.arange(x) * spacing[ii] for ii, x 
            in enumerate(displacement_field.shape[:-1])]

    return RegularGridInterpolator(axes, displacement_field, fill_value=0)
                                   

def apply_displacement_transformation(displacement_interpolator, spacing, 
                                      points):
    '''Apply a displacement transformation to a points list
    
    Parameters
    ----------
    displacement_interpolator : RegularGridInterpolator
        When called with an array of points -> returns linearly interpolated 
        displacements originating from those points.
    spacing : tuple of numeric
        Grid resolution in microns.
    points : numpy ndarray
        One point per row. Positions in columns.
    
    Returns
    -------
    numpy_ndarray : 
        Transformed points. num_points X num_dimensions.
    
    '''

    points = np.array(points)
    
    if points.ndim == 1:
        points = points[None, :]
    
    points = points * np.array(spacing)
    return points + displacement_interpolator(np.split(points, 
                                                       points.shape[0], 
                                                       axis=0)).squeeze()
                                                       
                                                       
def apply_affine_transformation(affine_params, points):
    '''Applies an affine transformation to each point in a set.
    
    Parameters
    ----------
    affine_params : numpy ndarray
        (ndims + 1) X (ndims + 1) array of transformation parameters. 
    points : numpy ndarray
        num_points X num_dimensions.
        
    Returns
    -------
    numpy_ndarray : 
        Transformed points. num_points X num_dimensions.
        
    Notes
    -----
    The alignment3ds are denominated in microns and come in two flavors. Trv
    maps ccf-space to mouse space. Tvr is the inverse.
    
    '''

    points = np.array(points)
    if points.ndim == 1:
        points = points[None, :]

    points = np.hstack((points, np.ones(points.shape[0])[:, None]))
    return np.dot(affine_params, points.T).T[:, :3]
    
    
def apply_composite_transformation(displacement_field, affine_params, 
                                   spacing, points):
    '''Applies a displacement transform followed by an affine transform to a 
    set of points.
    
    
    Parameters
    ----------
    displacement_interpolator : RegularGridInterpolator
        When called with an array of points -> returns linearly interpolated 
    affine_params : numpy ndarray
        (ndims + 1) X (ndims + 1) array of transformation parameters. 
    spacing : tuple of numeric
        Grid resolution in microns.
    points : numpy ndarray
        num_points X num_dimensions.
    
    Returns
    -------
    numpy_ndarray : 
        Transformed points. num_points X num_dimensions.
    
    
    '''
    
    displacement_interpolator = build_interpolator(displacement_field, 
                                                   spacing)
    
    points = apply_displacement_transformation(displacement_interpolator, 
                                               spacing, points)
                                               
    return apply_affine_transformation(affine_params, points)