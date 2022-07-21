"""
    Code for running planning and control software using DER library for soft robot path following. 
    Paper under review
    Copyright Soft Machines Lab 2022
    (license TBD)
"""
from re import S
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import spatial
from multiprocessing import Queue
from queue import Empty
import time


class Robot():
    def __init__(self, model, x, y, theta, xdot=0, ydot=0, thetadot=0, state_queue = [], mode = 'brute'):
        
        # set of possible actions
        self.actions = ['A','B','C','D','E','F','G','H','I']

        # model from DER simulations - 12 elements per simulation
        # ['x0','y0','t0','xdot0','ydot0','tdot0','xf','yf','tf','xdotf','ydotf','tdotf']
        self.model = np.load(model)
        self.model[:,8] = self.model[:,8]*np.pi/180
        self.model[:,-1] = self.model[:,-1]*np.pi/180
        #self.model = np.load('simfrog_numpy_data_2021-11-30.npy')
        # actions corresponding to each simulation
        self.sim_actions = np.load('simfrog_numpy_actions_2022-01-26.npy')
        #self.sim_actions = np.load('simfrog_numpy_actions_2021-11-30.npy')

        # initialize state
        self.x = x
        self.y = y
        self.theta = theta
        self.xdot = xdot
        self.ydot = ydot
        self.thetadot = thetadot
        
        self.mode = mode

        # queue for polling robot state from camera
        self.state_queue = state_queue

        # need to create a dict with self.actions as the keys and the action strings as the values
        action_strings = ['1100', '1111', '0111','1011','0110','1001','0100','1000','0000']
        self.action_dict = dict(zip(self.actions, action_strings))

    def predict(self, state, action, closest_state, ind):
        assert(action in self.actions)
        
        # Brute force planner that simply takes estimates action's effect based on closest point in the library
        if self.mode == 'brute':
            new_state = self.model[closest_state[0],:]
            next_actions = self.sim_actions[closest_state[0]]
            new_state = new_state[next_actions==action,:]
            new_state = new_state[:,6:]

        # Stock Locally Weight Regression from pg 20-21 of Atkeson et al 1997
        elif self.mode == 'lwr':
            time1 = time.time()
            model = self.model[ind,:]
            next_actions = self.sim_actions[ind]
            action_data = model[next_actions==action,:]
            #action_data = self.model[self.sim_actions==action,0:6]
            
            query = self.world_to_body(state[0:6]).T
            x = action_data[:,3:6] - query[3:6] 
            
            weights = np.diag(np.array(np.sqrt(self.kernel((x[:,0])**2 + (x[:,1])**2 + (x[:,2])**2)))).astype('float32') #would be subtracted for distance but already subtracted query
            # # we could also scale the effect of certain features
            time2 = time.time() - time1
            query = np.hstack((query[3:6],np.ones((1))))
            x = np.hstack((x,np.ones((len(action_data),1)))).astype('float32')
            time3 = time.time() - time2 - time1
            y = action_data[:,6:].astype('float32')
            
            y = model[next_actions==action,6:]
            
            z = np.matmul(weights, x)
            #z = np.einsum('ij,jk', weights, x)
            
            v = np.matmul(weights, y)
            
            
            lamb = np.diag(np.repeat(1,4)*1e-6)# 4 = num_columns in x + 1, ridge
            zpseudo = np.linalg.inv((z.T @ z)+lamb) @ z.T
            new_state = (query @ zpseudo @ v).reshape(6,1).T
            
            
            
            #print(time2)
        #time2 = time.time() - time1
        return self.body_to_world(state, new_state[0:1,:])


    def kernel(self, distance):
        return 1/(1+distance**(2))


    def world_to_body(self,state):
        theta = state[0,2]# - np.pi/2
        Vxb = state[0,3]*np.cos(theta) + state[0,4]*np.sin(theta)
        Vyb = -state[0,3]*np.sin(theta) + state[0,4]*np.cos(theta)
        return np.array([state[0,0], state[0,1], theta, Vxb, Vyb, state[0,5]])

    def body_to_world(self, state, new_state):

        
        x,y,theta = state[0,0:3]
        theta = theta - np.pi/2
        r_xfw = x + new_state[0,0]*np.cos(theta) - new_state[0,1]*np.sin(theta)
        r_yfw = y + new_state[0,0]*np.sin(theta) + new_state[0,1]*np.cos(theta)
        theta_fw = theta + new_state[0,2]
        V_xfw = new_state[0,3]*np.cos(theta) - new_state[0,4]*np.sin(theta)
        V_yfw = new_state[0,3]*np.sin(theta) + new_state[0,4]*np.cos(theta)
        omega_fw = new_state[0,5]
        return np.array([[r_xfw, r_yfw, theta_fw + np.pi/2, V_xfw, V_yfw, omega_fw]])

    def poll(self):# for hardware, this is gonna poll camera instead
        state = self.state_queue.get()
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.xdot =  state[3]
        self.ydot = state[4]
        self.thetadot = state[5]
    
    def state(self):
        return np.array([[self.x, self.y, self.theta, self.xdot, self.ydot, self.thetadot]])
        

    def __str__(self):
        return f"x:{self.x}, y:{self.y}, theta:{self.theta}, xdot:{self.xdot}, ydot:{self.ydot}, thetadot:{self.thetadot}"



class FrogPlanner():
    def __init__(self, frog, path, serial_port, search_depth=3):
        self.frog = frog
        self.path = path
        self.serial_port = serial_port
        self.search_depth = search_depth
        self.next_action = None
        self.action_queue = Queue()
        self.glob = []


    def plan(self):
        path = self.path
        
        for i in range(500):
            state = np.zeros((9,6))
            for j in range(9):
                self.frog.poll()
                state[j,:] = self.frog.state()
            state = np.mean(state, axis=0)
            state = state.reshape((1,-1))
            self.best_action_list = []
            self.best_state_list = []
            best_action, cost, costs = self.evaluate_actions(state, self.search_depth)
            _, cost,costs = self.evaluate_actions(state, 0)
            print('dist cost = ' + str(costs[0]))
            print('ang cost = ' + str(costs[1]))
            print('prog cost = ' + str(costs[2]))
            pose_vec = np.squeeze([np.cos(state[0,2]), np.sin(state[0,2])])
            print('ang = ' + str(pose_vec))
            print('angd = ' + str(path.angd))
            print("\n")
            self.best_action_list.append
            self.next_action = best_action
            self.execute()
        print((self.frog.state(), path.p3))

    def evaluate_actions(self, state, depth):
        if depth == 0:
            
            cost,dists = self.path.cost(state)

            return None, cost, dists

        best_cost = np.inf
        best_action = None
        x = (self.frog.world_to_body(state[0:6]).T - self.frog.model[:,0:6])/self.frog.world_to_body(state[0:6]).T
        
        dist = np.sqrt((x[:,3])**2 + (x[:,4])**2 + (x[:,5])**2)
        closest_state = np.where(dist == dist.min())

        if self.frog.mode == 'lwr':
            k = 9*100
        
            ind = np.argpartition(dist, k)[:k]
        else:
            ind = []

                
        for action in self.frog.actions:
            next_state = self.frog.predict(state, action, closest_state, ind)
            _, cost,dists = self.evaluate_actions(next_state, depth-1)
            cost = cost*(depth*0.3+1)
            if cost < best_cost:
                best_cost = cost
                best_action = action
                self.best_action_list.append(best_action)
                self.best_state_list.append(next_state)
        
        return best_action, best_cost, dists
    def get_action_queue(self):
        return self.action_queue

    def execute(self):
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except Empty:
                pass

        # Put state variables in queue
        self.action_queue.put_nowait(self.next_action)
        



class Bezier():
    def __init__(self, conversion, center, width, height, state, num_pts, shape='sp'):
        self.init_state = np.array([state[0], state[1]])# + np.array([[640/2,480/2]])
        self.init_orientation = state[2]
        self.conversion = conversion
        self.center = center
        self.shape = shape
        starting_pos = np.rint(self.center).astype(int)
        self.l = self.length(starting_pos, self.init_orientation)
        self.width = width
        self.height = height
        self.p0 = self.init_state 
        self.p1 = np.array([height*0.9, width*0.1]).T
        self.p2 = np.array([width*1, height*0.9]).T
        self.p3 = np.array([width*0.1, height*0.3]).T
        vec0=self.derivative(0)
        self.theta0 = np.arctan2(vec0[1], vec0[0])
        self.curve_points, ts = self.discretize(num_pts)
        self.angd = []
        

    def curve(self,t):
        return (1-t)**3*self.p0 + \
                3*(1-t)**2*t*self.p1 + \
                3*(1-t)*t**2*self.p2 + \
                t**3*self.p3

    def derivative(self, t):
        return 3*(1-t)**2*(self.p1-self.p0) + \
               6*(1-t)*t*(self.p2-self.p1) + \
               3*t**2*(self.p3-self.p2)

    def discretize(self, num_points):
        ts = np.linspace(0,1,num_points).reshape(-1, 1)
        points = np.zeros((num_points,4))
        l_e = 0.85
        if self.shape == 'sp':
            points[:,0:2] = self.curve(ts) 
            points[:,2:4] = self.derivative(ts)
        elif self.shape == 'sq':
            l_e = 0.7
            width = 4*640*l_e
            height = 4*480*l_e
            quarter_index = np.rint(num_points/4).astype(int)
            points[:quarter_index,1] = height*ts[:quarter_index,0]
            points[:quarter_index,3] = 1.0
            points[quarter_index:quarter_index*2,0] = width*(ts[quarter_index:quarter_index*2,0] - ts[quarter_index,0])
            points[quarter_index:quarter_index*2,1] = height*ts[quarter_index,0]
            points[quarter_index:quarter_index*2,2] = 1.0
            points[quarter_index*2:quarter_index*3,1] = points[quarter_index,1] -height*(ts[quarter_index*2:quarter_index*3,0] - ts[quarter_index*2,0])
            points[quarter_index*2:quarter_index*3,0] = width*(ts[quarter_index*2,0] - ts[quarter_index,0])
            points[quarter_index*2:quarter_index*3,3] = -1.0
            points[quarter_index*3:quarter_index*4,0] = points[quarter_index*2,0] - width*(ts[quarter_index*3:quarter_index*4,0] - ts[quarter_index*3,0])
            points[quarter_index*3:quarter_index*4,2] = -1.0
            
            points[:,:2] = (points[:,:2] + [50, 50])/self.conversion
            
        elif self.shape == 'el':
            tf = ts[-1,0]
            l_e = 1.1
            ellipse_ang = ts*2*np.pi/tf*0.95
            x = self.l*l_e/4*(1.0 - np.cos(ellipse_ang))
            xdot = self.l*l_e/4*np.sin(ellipse_ang)
            y = -self.l*l_e/2/2*np.sin(ellipse_ang)
            ydot = -self.l*l_e/2/2*np.cos(ellipse_ang)

            t = self.init_orientation
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

            rotated_p = R @ np.hstack((x, y)).T + np.array([[self.init_state[0]], [self.init_state[1]]])
            rotated_dp = R @ np.hstack((xdot, ydot)).T
            points[:,0] = rotated_p[0,:]
            points[:,1] = rotated_p[1,:]
            points[:,2] = rotated_dp[0,:]
            points[:,3] = rotated_dp[1,:]
        
        elif self.shape == 'si':
            x = ts*self.l*l_e
            xdot = self.l*l_e*np.ones((num_points,1))
            t = self.init_orientation
            f = 10
            y = self.l/10*np.sin(f*ts)
            ydot = f*self.l/10*np.cos(f*ts)
            
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            rotated_p = R @ np.hstack((x, y)).T + np.array([[self.init_state[0]], [self.init_state[1]]])
            rotated_dp = R @ np.hstack((xdot, ydot)).T
            points[:,0] = rotated_p[0,:]
            points[:,1] = rotated_p[1,:]
            points[:,2] = rotated_dp[0,:]
            points[:,3] = rotated_dp[1,:]
        
        elif self.shape == 'li':
            x = ts*self.l*l_e
            xdot = np.ones((num_points,1))*self.l*l_e
            t = self.init_orientation
            f = 50
            y = np.zeros((num_points,1))
            ydot = np.zeros((num_points,1))
            
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            rotated_p = R @ np.hstack((x, y)).T + np.array([[self.init_state[0]], [self.init_state[1]]])
            rotated_dp = R @ np.hstack((xdot, ydot)).T
            points[:,0] = rotated_p[0,:]
            points[:,1] = rotated_p[1,:]
            points[:,2] = rotated_dp[0,:]
            points[:,3] = rotated_dp[1,:]

        return points, ts

    def length(self, starting_pos, theta):
        x = starting_pos[0]
        y = starting_pos[1]
        t1 = np.arctan(480/640)
        t2 = np.pi - t1
        t3 = np.pi + t1
        t4 = 2*np.pi - t1
        if theta < t1:
            l = np.abs((640 - x)/np.cos(2*np.pi - theta))
        elif theta > t1 and theta < t2:
            l = np.abs(y/np.sin(theta))
        elif theta > t2 and theta < t3:
            l = np.abs(x/np.cos(np.pi - theta))
        elif theta > t3 and theta < t4:
            l = np.abs((480 - y)/np.sin(theta - np.pi))
        else:
            l = np.abs((640 - x)/np.cos(2*np.pi - theta))

        l = l/self.conversion

        return l
        

    

    def plot(self):
        points = self.get_curve()
        plt.figure()
        plt.scatter(points[:,0], points[:,1])
        # plt.scatter([self.p0[0], self.p1[0], self.p2[0], self.p3[0]],
        #             [self.p0[1], self.p1[1], self.p2[1], self.p3[1]],
        #             color='r')
        plt.show()
    
    def get_curve(self):
        return self.curve_points


    def ang_diff(self, pose_vec, desired_vec):
        dot = np.dot(pose_vec, desired_vec)
        
        mag = np.linalg.norm(pose_vec) * np.linalg.norm(desired_vec)
        return np.abs(np.arccos(dot/mag))


    def cost(self, pose, num_points=100):
        dist_weight = 500.0
        ang_weight = 50
        progression_weight = 150*2
        
        curve_points, ts = self.discretize(num_points)


        
        dists = spatial.distance.cdist(curve_points[:,:2], pose[0:1,:2])
        
        dist_cost = dist_weight*np.min(dists)

        mindex = np.argmin(dists)
        
        min_t = ts[mindex,0]

        progression_cost = progression_weight * (1-min_t)

        closest_point = curve_points[mindex]
        
        pose_vec = np.squeeze([np.cos(pose[0,2]), np.sin(pose[0,2])])
        desired_vec = [closest_point[2], closest_point[3]]
        self.angd = desired_vec/np.linalg.norm(desired_vec)
        ang_cost = ang_weight * (self.ang_diff(pose_vec, desired_vec))
        


        return dist_cost + ang_cost + progression_cost, [dist_cost, ang_cost, progression_cost]

def executor(planner,activation_time,period,last_action_queue):
    fl = ' 0'
    fr = ' 1'
    bl = ' 2'
    br = ' 3'

    to_microcontroller_msg = f'{"p "+str(activation_time)}\n'
    planner.serial_port.write(to_microcontroller_msg.encode('UTF-8'))

    while True:
        action = planner.get_action_queue().get()
        print(action)
        last_action_queue.put(action)
        msg1 = 'h'
        action_sequence = planner.frog.action_dict[action]
        print(action+' - '+action_sequence)
        
        if action_sequence[0] == '1':
            msg1 = msg1 + fr
        if action_sequence[1] == '1':
            msg1 = msg1 + fl
        if action_sequence[0:2] == '00':
            pass
        else:
            to_microcontroller_msg = f'{msg1}\n'
            planner.serial_port.write(to_microcontroller_msg.encode('UTF-8'))
        time.sleep(activation_time/1000)
        msg2 = 'h '
        if action_sequence[2] == '1':
            msg2 = msg2 + bl
        if action_sequence[3] == '1':
            msg2 = msg2 + br
        if action_sequence[2:4] == '00':
            pass
        else:
            to_microcontroller_msg = f'{msg2}\n'
            planner.serial_port.write(to_microcontroller_msg.encode('UTF-8'))
        time.sleep(period - 2*activation_time/1000)
        



if __name__ == "__main__":
    width = 600
    height = 400
    state = [234.577425645519, -324.91754761909897, 2.862985398826103, 46696.20477905567, -64679.78023962873, 569.9207930820085]
    state_queue = Queue()
    state_queue.put(state)
    path = Bezier(width, height, state, 100, 'el')
    #path.plot()
    frog = Robot(state[0],state[1],state[2],state[3],state[4],state[5],state_queue,'lwr')
    planner = FrogPlanner(frog, path, [], 10)
    planner.plan()
    # plt.show()
