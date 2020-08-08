from nurbs_fit.utils.GVF import EnforceMirrorBoundary
from numpy import gradient
from scipy.ndimage.filters import laplace as del2

def GVF3D(f, mu, iterations, verbose=False):
    """
    % This function calculates the gradient vector flow (GVF)
    % of a 3D image f.
    %
    % inputs:
    %   f : The 3D image
    %   mu : The regularization parameter. Adjust it to the amount
    %        of noise in the image. More noise higher mu
    %   iterations: The number of iterations. 
    %               sqrt(nr of voxels) is a good choice
    %
    % outputs:
    %   u,v,w : The GVF
    %
    % Function is written by Erik Smistad, Norwegian University 
    % of Science and Technology (June 2011) based on the original 
    % 2D implementation by Xu and Prince
    """

    # Normalize 3D image to be between 0 and 1
    f = (f-f.min())/(f.max()-f.min());

    # Enforce the mirror conditions on the boundary
    f = EnforceMirrorBoundary(f);

    # Calculate the gradient of the image f
    [Fx, Fy, Fz] = gradient(f);
    magSquared = Fx*Fx + Fy*Fy + Fz*Fz;
    
    # Set up the initial vector field
    u = Fx;
    v = Fy;
    w = Fz;
    
    for i in range(iterations):
        if verbose:
            print('\rGVF iter: ' + str(i+1) + '/' + str(iterations),
                  end='', flush=True)

        # Enforce the mirror conditions on the boundary
        u = EnforceMirrorBoundary(u);
        v = EnforceMirrorBoundary(v);
        w = EnforceMirrorBoundary(w);

        # Update the vector field
        u = u + mu*6*del2(u) - (u-Fx)*magSquared;
        v = v + mu*6*del2(v) - (v-Fy)*magSquared;
        w = w + mu*6*del2(w) - (w-Fz)*magSquared;

    if verbose:
        print('')

    return [u,v,w]

def EnforceMirrorBoundary(f):
    """
    % This function enforces the mirror boundary conditions
    % on the 3D input image f. The values of all voxels at 
    % the boundary is set to the values of the voxels 2 steps 
    % inward
    """
    
    [N, M, O] = np.shape(f);

    xi = np.arange(1, M-2);
    yi = np.arange(1, N-2);
    zi = np.arange(1, O-2);

    # Corners
    f[[0, N-1], [0, M-1], [0, O-1]] = f[[2, N-3], [2, M-3], [2, O-3]]

    # Edges
    f[np.ix_([0, N-1], [0, M-1], zi)] = \
        f[np.ix_([2, N-3], [2, M-3], zi)]
        
    f[np.ix_(yi, [0, M-1], [0, O-1])] = \
        f[np.ix_(yi, [2, M-3], [2, O-3])]
    f[np.ix_([0, N-1], xi, [0, O-1])] = \
        f[np.ix_([2, N-3], xi, [2, O-3])]

    # Faces
    f[np.ix_([0, N-1], xi, zi)] = \
        f[np.ix_([2, N-3], xi, zi)];
    f[np.ix_(yi, [0, M-1], zi)] = \
        f[np.ix_(yi, [2, M-3], zi)];
    f[np.ix_(yi, xi, [0, O-1])] = \
        f[np.ix_(yi, xi, [2, O-3])];   
    
    return f 