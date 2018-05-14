import numpy as np;
import re;
import os;
import h5py;
from PIL import Image;
import time;
import struct;
from scipy.spatial import ConvexHull;
import scipy;
YIQ2RGB = np.array([[1.0,0.9469,0.6236],[1.0,-0.2748,-0.6357],[1.0,-1.1,1.7]],dtype=np.float32);

def sphere_to_YIQ(ipts):
    pts = ipts.copy();
    IQ = pts[:,:,0:2];
    IQ[:,:,0] = np.arctan2(pts[:,:,1],pts[:,:,0]) / np.pi;
    IQ[:,:,1] = np.sqrt( np.sum( np.square( pts[:,:,0:3] ) , axis=2 ) );
    IQ[:,:,1] = np.sqrt( IQ[:,:,1] );
    IQ[:,:,1] = ( np.arccos( pts[:,:,2] / IQ[:,:,1] ) - ( np.pi / 2 ) ) / np.pi*2;
    IQ[:,:,0]*=0.5957;
    IQ[:,:,1]*=0.5226;
    Y = 0.6*np.ones([pts.shape[0],pts.shape[1],1],np.float32);
    YIQ = np.concatenate([Y,IQ],2).reshape([pts.shape[0],pts.shape[1],3,1]);
    YIQ2RGB_exp = np.zeros([pts.shape[0],pts.shape[1],1,1]) + YIQ2RGB;
    rgb = np.matmul(YIQ2RGB_exp,YIQ);
    rgb = np.where(rgb < 0, 0, rgb);
    rgb = np.where(rgb > 1.0,1.0,rgb);
    return rgb.reshape([pts.shape[0],pts.shape[1],3]);

def sphere_to_ellip(x3D,yGT):
    assert int(x3D.shape[0]) == int(yGT.shape[0]),"The batch size is not equal";
    outy = yGT.copy();
    for i in range(x3D.shape[0]):
        outy[ i , ... ] = sphere_to_ellip_sub( x3D[ i , ... ] , yGT[ i , ... ] );
    return outy;
        
def sphere_to_ellip_sub(x3D,yGT):
    affine = np.matmul( np.transpose( yGT ) , yGT );
    val,vec = scipy.linalg.eigh( affine );
    val /= float(yGT.shape[0]);
    valrt = np.sqrt( val );
    #print valrt; 
    proj = np.matmul( x3D , vec );
    valrt = valrt.reshape( ( 1 , 3 ) );
    scale = valrt / np.abs( proj );
    scale = np.amin( scale , axis=1 );
    #print scale
    scale = scale.reshape( ( -1 , 1 ) );
    proj *= scale*2.0;
    return np.matmul( proj , np.transpose(vec) );
    
def triangulateSphere(pts):
    hull_list = [];
    for i in range(pts.shape[0]):
        pt = pts[i,...];
        hull = ConvexHull(pt);
        for j in range(hull.simplices.shape[0]):
            simplex = hull.simplices[j,:];
            triangle = pt[simplex,:];
            m = triangle[0,:];
            p0p1 = triangle[1,:] -  triangle[0,:];
            p1p2 = triangle[2,:] -  triangle[1,:];
            k = np.cross(p0p1,p1p2);
            if np.dot(m,k) < 0:
                tmp = hull.simplices[j,1];
                hull.simplices[j,1] =  hull.simplices[j,2];
                hull.simplices[j,2] =  tmp;
        hull_list.append(hull);
    return hull_list;

def triangulateSphereIdx(pts):
    hull = ConvexHull(pts);
    return hull.simplices;
        
def randsphere(m=None):
    OUTPUTPOINTS = 1024;
    PTS_DIM = 3;
    if m is None:
        m = OUTPUTPOINTS;
    pts = np.zeros([m,PTS_DIM],np.float32);
    for i in range(m):
        r2 = 2.0;
        while r2 > 1.0:
            u = np.random.uniform(-1.0,1.0);
            v = np.random.uniform(-1.0,1.0);
            r2 = u*u+v*v;
        pts[i,0] = 2.0*u*np.sqrt(1-r2);
        pts[i,1] = 2.0*v*np.sqrt(1-r2);
        pts[i,2] = 1.0-2.0*r2;
    return pts;

def randsphere2(m=None):
    pts = np.random.normal(size=[m,3]).astype(np.float32);
    pts += 1e-8;
    pts_norm = np.sqrt(np.sum(np.square(pts),axis=1,keepdims=True));
    pts /= pts_norm;
    return pts;

def randsphere3(m=None):
    PTS_DIM = 3;
    pts = np.zeros([m,PTS_DIM],np.float32);
    pts_init = randsphere2(3);
    pts[0:3,:] = pts_init[0:3,:];
    for i in range(3,m):
        center = np.mean(pts[0:i,:],axis=0);
        tmppts = randsphere2(m);
        dist = np.sum(np.square(tmppts - center),axis=1);
        pts[i,:]=tmppts[np.argmax(dist),:];
    return pts;


def randsphere4(m=None):
    pts = np.zeros([m,3],np.float32);
    n = np.linspace(1,m,m);
    n += np.random.normal();
    tmp = 0.5*(np.sqrt(5)-1)*n;
    theta = 2.0*np.pi*(tmp - np.floor(tmp));
    pts[:,0] = np.cos(theta);
    pts[:,1] = np.sin(theta);
    pts[:,2] = 2.0*(n - n.min()) / (n.max()-n.min()) - 1.0;
    scale = np.sqrt(1 - np.square(pts[:,2]));
    pts[:,0] *= scale;
    pts[:,1] *= scale;
    return pts;
    

def getface(tri_lst):
    f = [];
    for tri in tri_lst:
        f.append(tri.simplices.copy());
    return f;

def rand_n_sphere(n,m=None):
    OUTPUTPOINTS = 1024;
    PTS_DIM = 3;
    if m is None:
        pts = np.zeros([n,OUTPUTPOINTS,PTS_DIM],np.float32);
    else:
        pts = np.zeros([n,m,PTS_DIM],np.float32);
    for i in range(n):
        pts[i,:,:] = randsphere(m);
    data_dict={'x3D':pts,'x3D_final':pts};
    return data_dict;

def rand_1_sphere(n,m=None):
    OUTPUTPOINTS = 1024;
    PTS_DIM = 3;
    if m is None:
        pts = np.zeros([n,OUTPUTPOINTS,PTS_DIM],np.float32);
    else:
        pts = np.zeros([n,m,PTS_DIM],np.float32);
    sphere = randsphere(m);
    for i in range(n):
        pts[i,:,:] = sphere;
    data_dict={'x3D':pts,'x3D_final':pts}; 
    return data_dict;

def edge_interp(pts,fidx):
    edge_dict = {};
    eidx_lst = [];
    for i in range(int(fidx.shape[0])):
        fidx0 = fidx[i,0];
        fidx1 = fidx[i,1];
        fidx2 = fidx[i,2];
        if not "%d_%d"%(fidx0,fidx1) in edge_dict.keys():
            edge_dict["%d_%d"%(fidx0,fidx1)] = len(eidx_lst);
            edge_dict["%d_%d"%(fidx1,fidx0)] = len(eidx_lst);
            eidx_lst.append((fidx0,fidx1));
        if not "%d_%d"%(fidx1,fidx2) in edge_dict.keys():
            edge_dict["%d_%d"%(fidx1,fidx2)] = len(eidx_lst);
            edge_dict["%d_%d"%(fidx2,fidx1)] = len(eidx_lst);
            eidx_lst.append((fidx1,fidx2));
        if not "%d_%d"%(fidx2,fidx0) in edge_dict.keys():
            edge_dict["%d_%d"%(fidx2,fidx0)] = len(eidx_lst);
            edge_dict["%d_%d"%(fidx0,fidx2)] = len(eidx_lst);
            eidx_lst.append((fidx2,fidx0));
    old_vnum = pts.shape[0];
    rfidx = np.zeros([4*fidx.shape[0],3],dtype=np.int32);
    for i in range(int(fidx.shape[0])):
        fidx0 = fidx[i,0];
        fidx1 = fidx[i,1];
        fidx2 = fidx[i,2];
        fidx01 = edge_dict["%d_%d"%(fidx0,fidx1)] + old_vnum;
        fidx12 = edge_dict["%d_%d"%(fidx1,fidx2)] + old_vnum;
        fidx20 = edge_dict["%d_%d"%(fidx2,fidx0)] + old_vnum;
        rfidx[4*i,:] = np.array([fidx0,fidx01,fidx20],dtype=np.int32);
        rfidx[4*i+1,:] = np.array([fidx1,fidx12,fidx01],dtype=np.int32);
        rfidx[4*i+2,:] = np.array([fidx2,fidx20,fidx12],dtype=np.int32);
        rfidx[4*i+3,:] = np.array([fidx12,fidx20,fidx01],dtype=np.int32);
    eidx = np.zeros([len(eidx_lst),2],dtype=np.int32);
    for i,p in enumerate(eidx_lst):
        eidx[i,0]=p[0];
        eidx[i,1]=p[1];
    return rfidx,eidx;

def rand_sphere_interp(n,m=None,level=3):
    if m is None:
        m = 256;
    pts = np.zeros([n,m,3],np.float32);
    sphere = randsphere4(m);
    for i in range(n):
        pts[i,:,:] = sphere;
    hulllst = triangulateSphere(sphere.reshape([1,-1,3]));
    interpidx = [];
    flst = [];
    flst.append(hulllst[0].simplices);
    for l in range(level):
        fidx,eidx = edge_interp(sphere,flst[-1]);
        flst.append(fidx);
        interpidx.append(eidx);
        interp_sphere = sphere[eidx,:];
        interp_sphere = np.mean(interp_sphere,axis=1);
        interp_sphere_norm = np.sqrt(np.sum(np.square(interp_sphere),axis=1,keepdims=True));
        interp_sphere /= interp_sphere_norm;
        sphere = np.concatenate([sphere,interp_sphere],axis=0);
    fidx,eidx = edge_interp(sphere,flst[-1]);
    interpidx.append(eidx);
    data_dict = {};
    data_dict['x3D'] = pts;
    data_dict['eidx'] = interpidx;
    data_dict['fidx'] = flst;
    ptsfinal = np.zeros([n,sphere.shape[0],3],np.float32);
    for i in range(n):
        ptsfinal[i,:,:] = sphere;
    data_dict['x3D_final'] = ptsfinal;
    return data_dict;

def laplace(n,fidx):
    lplidx = np.zeros([n,6],dtype=np.int32) - 1;
    lplcnt = np.zeros([n],dtype=np.int32);
    lplw = np.zeros([n,6],dtype=np.float32);
    for i in range(fidx.shape[0]):
        vidx0 = fidx[i,0];
        vidx1 = fidx[i,1];
        vidx2 = fidx[i,2];
        vs0 = lplidx[vidx0,:];
        vs1 = lplidx[vidx1,:];
        vs2 = lplidx[vidx2,:];
        if not vidx1 in vs0 and lplcnt[vidx0] < 6:
            lplidx[vidx0,lplcnt[vidx0]] = vidx1;
            lplw[vidx0,lplcnt[vidx0]] = 1.0;
            lplcnt[vidx0] += 1;
        if not vidx2 in vs0 and lplcnt[vidx0] < 6:
            lplidx[vidx0,lplcnt[vidx0]] = vidx2;
            lplw[vidx0,lplcnt[vidx0]] = 1.0;
            lplcnt[vidx0] += 1;
        if not vidx0 in vs1 and lplcnt[vidx1] < 6:
            lplidx[vidx1,lplcnt[vidx1]] = vidx0;
            lplw[vidx1,lplcnt[vidx1]] = 1.0;
            lplcnt[vidx1] += 1;
        if not vidx2 in vs1 and lplcnt[vidx1] < 6:
            lplidx[vidx1,lplcnt[vidx1]] = vidx2;
            lplw[vidx1,lplcnt[vidx1]] = 1.0;
            lplcnt[vidx1] += 1;
        if not vidx0 in vs2 and lplcnt[vidx2] < 6:
            lplidx[vidx2,lplcnt[vidx2]] = vidx0;
            lplw[vidx2,lplcnt[vidx2]] = 1.0;
            lplcnt[vidx2] += 1;
        if not vidx1 in vs2 and lplcnt[vidx2] < 6:
            lplidx[vidx2,lplcnt[vidx2]] = vidx1;
            lplw[vidx2,lplcnt[vidx2]] = 1.0;
            lplcnt[vidx2] += 1;
    assert lplcnt.min() > 0;
    lplw /= lplcnt.reshape((n,1)).astype(np.float32);
    return lplidx,lplw;

def rand_sphere_interp2(n,m=None,level=3):
    if m is None:
        m = 256;
    pts = np.zeros([n,m,3],np.float32);
    sphere = randsphere4(m);
    for i in range(n):
        pts[i,:,:] = sphere;
    hulllst = triangulateSphere(sphere.reshape([1,-1,3]));
    interpidx = [];
    flst = [];
    flst.append(hulllst[0].simplices);
    data_dict = {};
    data_dict['lplidx'] = [];
    data_dict['lplw'] = [];
    lplidx,lplw = laplace(int(sphere.shape[0]),flst[-1]);
    data_dict['lplidx'].append(lplidx);
    data_dict['lplw'].append(lplw);
    for l in range(level):
        fidx,eidx = edge_interp(sphere,flst[-1]);
        flst.append(fidx);
        interpidx.append(eidx);
        interp_sphere = sphere[eidx,:];
        interp_sphere = np.mean(interp_sphere,axis=1);
        interp_sphere_norm = np.sqrt(np.sum(np.square(interp_sphere),axis=1,keepdims=True));
        interp_sphere /= interp_sphere_norm;
        sphere = np.concatenate([sphere,interp_sphere],axis=0);
        lplidx,lplw = laplace(int(sphere.shape[0]),flst[-1]);
        data_dict['lplidx'].append(lplidx);
        data_dict['lplw'].append(lplw);
    _,eidx = edge_interp(sphere,flst[-1]);
    interpidx.append(eidx);
    data_dict['x3D'] = pts;
    data_dict['eidx'] = interpidx;
    data_dict['fidx'] = flst;
    ptsfinal = np.zeros([n,sphere.shape[0],3],np.float32);
    for i in range(n):
        ptsfinal[i,:,:] = sphere;
    data_dict['x3D_final'] = ptsfinal;
    return data_dict;
    
def rand_sphere_nointerp(n,m=None):
    pts = np.zeros([n,m,3],np.float32);
    sphere = randsphere4(m);
    for i in range(n):
        pts[i,:,:] = sphere;
    hulllst = triangulateSphere(sphere.reshape([1,-1,3]));
    flst = [];
    flst.append(hulllst[0].simplices);
    data_dict = {};
    _,eidx = edge_interp(sphere,flst[-1]);
    data_dict['x3D'] = pts;
    data_dict['x3D_final'] = pts;
    data_dict['eidx'] = [];
    data_dict['eidx'].append(eidx);
    data_dict['fidx'] = flst;
    data_dict['lplidx'] = [];
    data_dict['lplw'] = [];
    lplidx,lplw = laplace(int(sphere.shape[0]),flst[-1]);
    data_dict['lplidx'].append(lplidx);
    data_dict['lplw'].append(lplw);
    return data_dict;