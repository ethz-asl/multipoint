import math
import numpy as np

def get_timestamp(msg):
    '''
    Get the timestamp of a ROS message with a header.

    Parameters
    ----------
    msg : ros message
        The input ros message. Must contain a header with the timestamp

    Returns
    -------
    timestamp : float
        The timestamp in seconds
    '''

    return msg.header.stamp.secs + msg.header.stamp.nsecs * 10**-9

def quat2euler(qx, qy, qz, qw):
    '''
    Compute the euler angles based on the input quaternion

    Parameters
    ----------
    qx : float
        x-component of the quaterion
    qy : float
        y-component of the quaterion
    qz : float
        z-component of the quaterion
    qw : float
        w-component of the quaterion

    Returns
    -------
    roll : float
        Roll angle [rad]
    pitch : float
        Pitch angle [rad]
    yaw : float
        Yaw angle [rad]
    '''

    roll = np.arctan2(2.0*(qw*qx+qy*qz),1.0-2.0*(qx*qx+qy*qy))
    pitch = np.arcsin(2*(qw*qy-qz*qx))
    yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

    return roll, pitch, yaw

def get_camera_matrix(intrinsics):
    '''
    Convert the flat intrinsics vector to the camera matrix.

    Parameters
    ----------
    intrinsics : list/array
        1-D list or array containing the intrinsics.
        It should have the following form:
        [fx, fy, cx, cy]

    Returns
    -------
    mag: numpy.array
        The 3x3 camera matrix
    '''

    mat = np.eye(3)
    mat[0,0] = intrinsics[0]
    mat[1,1] = intrinsics[1]
    mat[0,2] = intrinsics[2]
    mat[1,2] = intrinsics[3]

    return mat

def isRotationMatrix(R, eps = 1e-6):
    '''
    Checks if a matrix is a valid rotation matrix.

    Parameters
    ----------
    R : numpy.array
        3x3 rotation matrix

    Returns
    -------
    is_rot: bool
        Indicates if the input is a rotation matrix
    '''

    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)

    return n < eps

def rotationMatrixToEulerAngles(R, eps = 1e-6) :
    '''
    Calculates rotation matrix to euler angles (Bryan-Tait)
    Source: http://nghiaho.com/?page_id=846

    Parameters
    ----------
    R : numpy.array
        3x3 rotation matrix

    Returns
    -------
    angles: numpy.array
        The Bryan-Tait angles [rad]
    '''

#     assert(isRotationMatrix(R))

    if abs(R[0,2] - 1.0) < eps:
        roll = 0.0
        pitch = math.pi * 0.5
        yaw = math.atan2(R[1,0], R[0,0])
    elif abs(R[0,2] + 1.0) < eps:
        pitch = -math.pi * 0.5
        yaw = math.atan2(R[1,0], R[0,0])
    else:
        roll = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], math.sqrt(R[2,1] * R[2,1] +  R[2,2] * R[2,2]))
        yaw = math.atan2(R[1,0], R[0,0])

    return np.array([roll, pitch, yaw])
