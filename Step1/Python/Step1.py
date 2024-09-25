import numpy as np
import matplotlib.pyplot as plt
import os

################## READ CSV DATA ##################

# Reads the mocap data from the csv file
# input:    none
# returns:  t, x, y, z, roll, pitch, yaw 
#           all vectors of length N where N is number of data points
def read_data():
    # read csv file
    lab2_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(lab2_path, 'mocap_data.csv')
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    
    # each column is a different variable
    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    z = data[:,3]
    roll_deg = data[:,4]
    pitch_deg = data[:,5]
    yaw_deg = data[:,6]
    
    # Important! Convert degrees to radians
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    # convert to 1xN vectors
    t = np.transpose(t)
    x = np.transpose(x)
    y = np.transpose(y)
    z = np.transpose(z)
    roll = np.transpose(roll)
    pitch = np.transpose(pitch)
    yaw = np.transpose(yaw)
    
    # return the data for further processing
    return t, x, y, z, roll, pitch, yaw

################## PLOTTING FUNCTIONS ##################
# Plots the position (x,y,z) vs time
# input:   t = N-vector for time
#          x = N-vector for x position
#          y = N-vector for y position
#          z = N-vector for z position
# Step 1a
def plot_position(t, x, y, z):
    plt.figure(1, figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title("Position vs Time")

    plt.subplot(3, 1, 2)
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')

    plt.subplot(3, 1, 3)
    plt.plot(t, z)
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')

    plt.tight_layout()
    # plt.show()
    
# Plots the velocity (vx,vy,vz) vs time
# input:   t = N-vector for time
#          vx = N-vector for x velocity
#          vy = N-vector for y velocity
#          vz = N-vector for z velocity
# Step 1b
def plot_velocity(t, vx, vy, vz):
    plt.figure(2, figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, vx)
    plt.xlabel('Time (s)')
    plt.ylabel('V_x (m/s)')
    plt.title("Velocity vs Time")

    plt.subplot(3, 1, 2)
    plt.plot(t, vy)
    plt.xlabel('Time (s)')
    plt.ylabel('V_y (m/s)')

    plt.subplot(3, 1, 3)
    plt.plot(t, vz)
    plt.xlabel('Time (s)')
    plt.ylabel('V_z (m/s)')

    plt.tight_layout()
    # plt.show()
    
# Plots the angle-axis representation of the orientation vs time
# input:   t = N-vector for time
#          omega_axis = 3xN vector of axes of rotation
#          where ||omega_axis|| = theta
# Step 1c
def plot_angle_axes(t, thetas, omega_axes_1, omega_axes_2, omega_axes_3):
    plt.figure(3, figsize=(15, 10))

    plt.subplot(4, 1, 4)
    plt.plot(t, thetas)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\theta$ (rad)')

    plt.subplot(4, 1, 1)
    plt.plot(t, omega_axes_1)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\omega$_1')
    plt.title("Orientation vs Time")

    plt.subplot(4, 1, 2)
    plt.plot(t, omega_axes_2)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\omega$_2')

    plt.subplot(4, 1, 3)
    plt.plot(t, omega_axes_3)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\omega$_3')

    plt.tight_layout()
    # plt.show()
    
# Plots the body angular velocity vs time
# input:   t = N-vector for time
#          omega_body = 3xN angular velocity vector in the body frame
# Step 1d
def plot_body_angular_velocity(t, omega_body):
    omega_1, omega_2, omega_3 = omega_body[0,:], omega_body[1,:], omega_body[2,:]

    plt.figure(4, figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, omega_1)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\omega_1^b$')
    plt.title('Angular Velocity vs Time')

    plt.subplot(3, 1, 2)
    plt.plot(t, omega_2)
    plt.xlabel("Time (s)")
    plt.ylabel(r'$\omega_2^b$')

    plt.subplot(3, 1, 3)
    plt.plot(t, omega_3)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\omega_3^b$')

    plt.tight_layout()
    plt.show()
    
# Plots the error metric vs time
# input:   t = N-vector for time
#          error = N-vector to plot
# Step 1e
def plot_error_metric(t, error):
    plt.figure(5, figsize=(15, 10))

    plt.plot(t, error)
    plt.xlabel('Time (s)')
    plt.ylabel('d^2')
    plt.title("Squared Distance to Target Orientation")


################## CALCULATIONS ##################
    
# Calculate the velocity of the drone in the inertial frame
# input:    t, x, y, z (all N vectors)
# returns:  vx, vy, vz (all N vectors)
# step 1b
def calculate_velocity(t, x, y, z):
    vx = np.diff(x) / np.diff(t)
    vy = np.diff(y) / np.diff(t)
    vz = np.diff(z) / np.diff(t)

    # Append the last value to keep the same length
    vx = np.append(vx, vx[-1])
    vy = np.append(vy, vy[-1])
    vz = np.append(vz, vz[-1])

    return vx, vy, vz
    
# Convert a 3-vector to a 3x3 skew symmetric matrix (hat operator)
# input:    W = 3x3 skew symmetric matrix
# returns:  v = 3-vector such that W = hat(v)
# Step 1c
def VecToso3(v):
    W = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return W

# an alias for VecToso3
# Step 1c
def hat(v):
    return VecToso3(v)

# Convert a 3x3 skew symmetric matrix to a 3-vector (vee operator)
# input:    W = 3x3 skew symmetric matrix
# returns:  v = 3-vector such that W = hat(v)
# Step 1c
def so3ToVec(W):
    v = np.array([W[2, 1], W[0, 2], W[1, 0]])
    return v
    

# an alias for so3ToVec
# Step 1c
def vee(W):
    return so3ToVec(W)



# convert euler angles to rotation matrix
# input:    alpha, beta, gamma (yaw, pitch, roll), all scalars in radians
# returns:  R = 3x3 rotation matrix 
def calculate_R(alpha, beta, gamma):
    R = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)],
                  [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)],
                  [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])
    return R

# Calculates the angle-axis representation of a single rotation matrix, R
# input:    R = expm (theta * hat(omega_axis)), 3x3 rotation matrix
# returns:  return hat(theta * omega_axis), 3x3 skew-symmetric matrix
def MatrixLog3(R):
    # case (a): when Ri = I (trace is close to 3)
    if np.isclose(np.trace(R), 3):
        theta = 0
        omega_1 = 0 # undefined, set to 0
        omega_2 = 0
        omega_3 = 0
        
    # case (b): when trace(R) = -1 (is close to -1)
    elif np.isclose(np.trace(R), -1):
        theta = np.pi

        # now, there are three cases to consider, 
        # based on the value of the diagonal entries

        # case (b.1): when r33 != -1
        # note: in Python, R[2,2] = r33 because indexing starts from 0 instead of 1
        if not np.isclose(R[2,2], -1): #not rotating along z-axis
            omega_1 = R[0, 2] / np.sqrt(2 * (1 + R[2, 2]))
            omega_2 = R[1, 2] / np.sqrt(2 * (1 + R[2, 2]))
            omega_3 = np.sqrt(1 + R[2, 2]) / np.sqrt(2)
        
        # case (b.2): when r22 != -1
        elif not np.isclose(R[1,1], -1): #Not rotating along y-axis
            omega_1 = R[0, 1] / np.sqrt(2 * (1 + R[1, 1]))
            omega_2 = np.sqrt(1 + R[1, 1]) / np.sqrt(2)
            omega_3 = R[2, 1] / np.sqrt(2 * (1 + R[1, 1]))

        # case (b.3): when r11 != -1
        else: #not rotating along x-axis
            omega_1 = np.sqrt(1 + R[0, 0]) / np.sqrt(2)
            omega_2 = R[1, 0] / np.sqrt(2 * (1 + R[0, 0]))
            omega_3 = R[2, 0] / np.sqrt(2 * (1 + R[0, 0]))
        
    # case (c): the "typical" case
    else:
        theta = np.arccos((np.trace(R) - 1) / 2)
        omega_1 = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
        omega_2 = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
        omega_3 = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
        
    # omega_axis vector
    omega_axis = np.array([omega_1, omega_2, omega_3])
    
    # Return the skew-symmetric matrix using the hat operator
    return hat(theta * omega_axis)


# Calculate the angle-axis representation of an exponential coordinate for rotation
# input:    expc3 = omega_axis * theta, a 3-vector exponential coordinate. R = expm(hat(expc3))
# returns:  omega_axis = normalized 3-vector
#           theta = scalar
def AngleAxis3(expc3):
    theta = np.linalg.norm(expc3)
    if np.isclose(theta, 0): #prevent div by 0 error
        return np.zeros(3), 0
    else:
        omega_axis = expc3 / theta
        return omega_axis, theta
    return omega_axis, theta

# Calculate the angle-axis representation for each rotation matrix
# input:    Rs = 3x3xN rotation matrices
# returns:  thetas = N-vector of angles
#           omega_axes_1 = N-vector of omega_1
#           omega_axes_2 = N-vector of omega_2
#           omega_axes_3 = N-vector of omega_3
# Hint: all you really have to do is convert R = Rs[:,:,i] into its exponential coordiante expc3
# Step 1c
def calculate_angle_axes(Rs):
    N = np.size(Rs,2)
    thetas = np.zeros(N)
    omega_axes_1 = np.zeros(N)
    omega_axes_2 = np.zeros(N)
    omega_axes_3 = np.zeros(N)

    for i in range(N):
        expc3 = so3ToVec(MatrixLog3(Rs[:, :, i]))
        omega_axis, theta = AngleAxis3(expc3)
        thetas[i] = theta
        omega_axes_1[i] = omega_axis[0]
        omega_axes_2[i] = omega_axis[1]
        omega_axes_3[i] = omega_axis[2]

    return thetas, omega_axes_1, omega_axes_2, omega_axes_3

# Calculate the angular velocity of the metafly in the body frame
# input:    t = N-vector for time
#           Rs = 3x3xN rotation matrices
# returns:  omega_body = 3xN angular velocity vector in the body frame
# Step 1d
def calculate_body_angular_velocity(t, Rs):
    N = np.size(Rs,2)
    # reminder: R(ti)^T = Rs[:,:,i].T
    # reminder: R(ti+1) = Rs[:,:,i+1]
    # reminder: matrix multiplication is denoted with @
    omega_body = np.zeros((3, N))

    for i in range(N-1):
        # Compute the change in rotation matrix from time step i to i+1
        R_diff = Rs[:, :, i+1] @ Rs[:, :, i].T

        # Calculate the logarithm of the rotation matrix difference
        omega_hat = MatrixLog3(R_diff)

        # Compute the angular velocity
        omega_body[:, i] = so3ToVec(omega_hat) / (t[i+1] - t[i])

    # For the last time step, we can simply set the angular velocity to be the same as the second-to-last step
    omega_body[:, -1] = omega_body[:, -2]
    return omega_body
    

# Calculate the error metric
# input:    Rs = 3x3xN rotation matrices
#           Rds = 3x3xN desired rotation matrices
# returns:  error = N-vector
# Step 1e
def calculate_error(Rs, Rds):
    N = Rs.shape[2]
    error = np.zeros(N)

    for i in range(N):
        # Compute the product R(t)^\top * R_d
        Rt_Rd = Rs[:, :, i].T @ Rds[:, :, i]

        # Compute the error metric using the formula 2 * Tr(I - R(t)^\top * R_d)
        error[i] = 2 * np.trace(np.eye(3) - Rt_Rd)

    return error
    

################## MAIN FUNCTION ##################

def main():
    
    # (a) read the csv file
    t, x, y, z, roll, pitch, yaw = read_data()
    N = np.size(t)
    plot_position(t, x, y, z) # uncomment when you are ready
    
    # (b) calculate velocity
    vx, vy, vz = calculate_velocity(t, x, y, z) # uncomment when you are ready
    plot_velocity(t, vx, vy, vz) # uncomment when you are ready
    
    # calculate all of the rotation matrices
    # (this is done for you :)
    Rs = np.zeros((3,3,N))
    for i in range(N):
        Rs[:,:,i] = calculate_R(yaw[i], pitch[i], roll[i])

    # (c) calculate the angle-axis representation for each R = expm(expc3)
    thetas, omega_axes_1, omega_axes_2, omega_axes_3 = calculate_angle_axes(Rs) # uncomment when you are ready
    plot_angle_axes(t, thetas, omega_axes_1, omega_axes_2, omega_axes_3) # uncomment when you are ready
    
    # (d) calculate the angular velocity in the body frame
    omega_body = calculate_body_angular_velocity(t, Rs) # uncomment when you are ready
    plot_body_angular_velocity(t, omega_body) # uncomment when you are ready
    
    # (e) calculate the error metric
    # We set Rd = the last rotation matrix in R
    # (this is done for you :)
    Rds = np.zeros_like(Rs)
    for i in range(np.size(Rs,2)):
        Rds[:,:,i] = Rs[:,:,-1]

    error = calculate_error(Rs, Rds) # uncomment when you are ready
    plot_error_metric(t, error) # uncomment when you are ready
    

if __name__ == "__main__":
    main()
