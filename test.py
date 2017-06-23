# -*- coding: utf-8 -*-
"""
Unit testing for mha.py

Use SimpleITK to generate test data.

Given SimpleITK, the usefulness of mha.py could be questioned, but
mha.py is far lighter, and requires no externally compiled libraries
(besides numpy).
"""
# For maximum usefulness, should support both 2 and 3 dimensional images
# with these tags. This will allow reading itk data, and importing most
# binary blocks in a useful way.
#
# data types to support
#
#
# The following tags are essential
# ObjectType = Image
# NDims = 3
# DimSize = 256 256 64
# ElementType = MET_USHORT
# ElementDataFile = image.raw  OR LOCAL OR LIST (but we can not support list for the moment)
#
# Optional flags
# To skip the header bytes in the image data file, use
# HeaderSize = X
# If you know there are no trailing bytes 
# HeaderSize = -1
#
# Coordinate system information
# ElementSpacing = X Y Z
# ElementSize = X Y Z
# ElementByteOrderMSB = True/False
# TransformMatrix = -0.172397 -0.768278 -0.58504 -0.128086 0.607108 -0.75969 -0.95708 0.0570597 0.207019
# Offset = 79.8114 -44.1469 -59.1385
# CenterOfRotation = 0 0 0
#sitkUInt8	Unsigned 8 bit integer
#sitkInt8	Signed 8 bit integer
#sitkUInt16	Unsigned 16 bit integer
#sitkInt16	Signed 16 bit integer
#sitkUInt32	Unsigned 32 bit integer
#sitkInt32	Signed 32 bit integer
#sitkUInt64	Unsigned 64 bit integer
#sitkInt64	Signed 64 bit integer
#sitkFloat32	32 bit float
#sitkFloat64	64 bit float
#sitkComplexFloat32	complex number of 32 bit float
#sitkComplexFloat64	complex number of 64 bit float
#sitkVectorUInt8	Multi-component of unsigned 8 bit integer
#sitkVectorInt8	Multi-component of signed 8 bit integer
#sitkVectorUInt16	Multi-component of unsigned 16 bit integer
#sitkVectorInt16	Multi-component of signed 16 bit integer
#sitkVectorUInt32	Multi-component of unsigned 32 bit integer
#sitkVectorInt32	Multi-component of signed 32 bit integer
#sitkVectorUInt64	Multi-component of unsigned 64 bit integer
#sitkVectorInt64	Multi-component of signed 64 bit integer
#sitkVectorFloat32	Multi-component of 32 bit float
#sitkVectorFloat64	Multi-component of 64 bit float
#sitkLabelUInt8	RLE label of unsigned 8 bit integers
#sitkLabelUInt16	RLE label of unsigned 16 bit integers
#sitkLabelUInt32	RLE label of unsigned 32 bit integers
#sitkLabelUInt64	RLE label of unsigned 64 bit integers


import numpy as np
import SimpleITK as sitk
import mha

def EqualityCheck(description,a,b,tolerance=0):
    if tolerance==0:
        if(a==b):
            return True
    else:
        d=np.array(a)-np.array(b)
        if max(np.abs(d))<tolerance:
            return True
        else:
            print "Specific differences"
            print d
    print description,"differs by more than",tolerance
    print a
    print b
    return False

def CheckMhaEqualImage(m,img,vector=False):
    if not EqualityCheck("Origin",m.offset,img.GetOrigin(),1.e-6):
        return False
    if not EqualityCheck("Spacing",m.spacing,img.GetSpacing(),1.e-6):
        return False
    dim=int(np.sqrt(len(m.direction_cosines)))
    tmp=np.ravel(np.array(m.direction_cosines).reshape((dim,dim)).T)
    if not EqualityCheck("Direction",tmp,img.GetDirection(),1.e-6):
        return False
    img2 = sitk.GetImageFromArray(m.data,isVector=vector)
    Z=sitk.GetArrayFromImage(img)
    if not EqualityCheck("Hashes",sitk.Hash(img),sitk.Hash(img2)):
        return False
    Z=sitk.GetArrayFromImage(img)
    if m.data.dtype != Z.dtype:
        print "Numpy types differ mha",m.data.dtype,"itk",Z.dtype
        return False
    if not np.array_equal(m.data,Z):
        print "Numpy arrays differ"
        return False
    return True
    
def GetRotationMatrixExampleForDimension(dim):
    c=np.cos(np.deg2rad(30))
    s=np.sin(np.deg2rad(30))
    if dim==2:
        return (c,-s,s,c)
    elif dim==3:
        return (c,0,-s,0,1,0,s,0,c)
    else:
        raise Exception("No rotation matrix defined for dimension %d"%dim)

def TestMhaToAndFrom_Scalar(dim,inType):
    # generate image
    img1 = sitk.GaussianSource( inType,  [100+i*10 for i in range(dim)], sigma=[10+i for i in range(dim)], mean = [50. -i*5 for i in range(dim)] )
    img1.SetOrigin([float(i+1)+float(i+2)/10. for i in range(dim)])
    img1.SetSpacing([float(i+1)/10. for i in range(dim)])
    img1.SetDirection(GetRotationMatrixExampleForDimension(dim))
    sitk.WriteImage(img1,"testwrite.mha")
    m=mha.new("testwrite.mha")
    if not CheckMhaEqualImage(m,img1):
        print "Test failed on read image"
        return 
    m.write_mha("testwrite2.mha")
    img2=sitk.ReadImage("testwrite2.mha")
    if not CheckMhaEqualImage(m,img2):
        print "Test failed on write image"
        return 

def TestMhaToAndFrom_Complex(dim,inType):
    # generate image
    img1 = sitk.GaussianSource( inType,  [128+i*16 for i in range(dim)], sigma=[10+i for i in range(dim)], mean = [50. -i*5 for i in range(dim)] )
    img1.SetOrigin([float(i+1)+float(i+2)/10. for i in range(dim)])
    img1.SetSpacing([float(i+1)/10. for i in range(dim)])
    img1.SetDirection(GetRotationMatrixExampleForDimension(dim))
    img2=sitk.ForwardFFT(img1)
    sitk.WriteImage(img2,"testwrite.mha")
    m=mha.new("testwrite.mha")
    if not CheckMhaEqualImage(m,img2):
        print "Test failed on read image"
        return 
    m.write_mha("testwrite2.mha")
    img3=sitk.ReadImage("testwrite2.mha")
    if not CheckMhaEqualImage(m,img3):
        print "Test failed on write image"
        return 


def TestMhaToAndFrom_Vector(dim,inScalarType,inVectorType,components):
    # generate image
    sz=[128+i*16 for i in range(dim)]
    img1=sitk.Image(sz,inVectorType,components)
    Z=sitk.GetArrayFromImage(img1)
    for c in range(components):
        img = sitk.GaussianSource( inScalarType,  [128+i*16 for i in range(dim)], sigma=[10+i+c for i in range(dim)], mean = [50. -i*5 -c*5 for i in range(dim)] )
        Z[:,:,:,c]=sitk.GetArrayFromImage(img)
    img1=sitk.GetImageFromArray(Z)
    img1.SetOrigin([float(i+1)+float(i+2)/10. for i in range(dim)])
    img1.SetSpacing([float(i+1)/10. for i in range(dim)])
    img1.SetDirection(GetRotationMatrixExampleForDimension(dim))
    sitk.WriteImage(img1,"testwrite.mha")
    m=mha.new("testwrite.mha")
    if not CheckMhaEqualImage(m,img1):
        print "Test failed on read image"
        return 
    m.write_mha("testwrite2.mha")
    img3=sitk.ReadImage("testwrite2.mha")
    if not CheckMhaEqualImage(m,img3):
        print "Test failed on write image"
        return 


dim=3
print "================================================"
print "Test Scalar case"
print "================================================"
for t in [sitk.sitkUInt8, 
          sitk.sitkInt8,
          sitk.sitkUInt16,
          sitk.sitkInt16,
          sitk.sitkUInt32,
          sitk.sitkInt32,
          sitk.sitkUInt64,
          sitk.sitkInt64,
          sitk.sitkFloat32,
          sitk.sitkFloat64
          ]:
    img=sitk.Image(1,2,t)
    s=img.GetPixelIDTypeAsString()
    print "Processing dimension %d and type %s"%(dim,s)
    try:
        TestMhaToAndFrom_Scalar(dim,t)
    except Exception as e:
        print "Exception thrown",e

print "================================================"
print "Test Complex case"
print "================================================"
for t in [sitk.sitkFloat32,
          sitk.sitkFloat64
          ]:
    img=sitk.ForwardFFT(sitk.Image(16,16,16,t))
    s=img.GetPixelIDTypeAsString()
    print "Processing dimension %d and type %s"%(dim,s)
    try:
        TestMhaToAndFrom_Complex(dim,t)
    except Exception as e:
        print "Exception thrown",e

print "================================================"
print "Test Vector case"
print "================================================"
for c in [2,3,4]:
    for st,vt in [(sitk.sitkUInt8,sitk.sitkVectorUInt8), 
              (sitk.sitkInt8,sitk.sitkVectorInt8),
              (sitk.sitkUInt16,sitk.sitkVectorUInt16),
              (sitk.sitkInt16,sitk.sitkVectorInt16),
              (sitk.sitkUInt32,sitk.sitkVectorUInt32),
              (sitk.sitkInt32,sitk.sitkVectorInt32),
              (sitk.sitkUInt64,sitk.sitkVectorUInt64),
              (sitk.sitkInt64,sitk.sitkVectorInt64),
              (sitk.sitkFloat32,sitk.sitkVectorFloat32),
              (sitk.sitkFloat64,sitk.sitkVectorFloat64)
              ]:
        img=sitk.Image((1,2),st)
        sst=img.GetPixelIDTypeAsString()
        img=sitk.Image((1,2),vt,c)
        svt=img.GetPixelIDTypeAsString()
        print "Processing dimension %d, components %d, and type %s/%s"%(dim,c,sst,svt)
        try:
            TestMhaToAndFrom_Vector(dim,st,vt,c)
        except Exception as e:
            print "Exception thrown",e
