import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import spatial



class Robot():
    def __init__(self, x, y, theta, xdot=0, ydot=0, thetadot=0):
        self.actions = [1,2,3,4,5]
        self.model = pd.DataFrame(data=np.array([[1, 0, 0, 0,  0, 10,        0, 0, 0, 0],
                                                 [2, 0, 0, 0, -4,  6,  np.pi/4, 0, 0, 0],
                                                 [2, 1, 0, 0, -4,  7,  np.pi/4, 0, 0, 0.1],#extra data for testing
                                                 [3, 0, 0, 0,  4,  6, -np.pi/4, 0, 0, 0],
                                                 [4, 0, 0, 0,  0,  0,  np.pi/2, 0, 0, 0],
                                                 [5, 0, 0, 0,  0,  0, -np.pi/2, 0, 0, 0]]),
                                  columns = ["action", "xdot", "ydot", "thetadot", "dx","dy","dtheta","dxdot","dydot","dthetadot"])
        self.x = x
        self.y = y
        self.theta = theta
        self.xdot = xdot
        self.ydot = ydot
        self.thetadot = thetadot

    def predict(self, state, action, mode = 'state'):
        assert(action in self.actions)
        relevant_data = self.model[self.model['action']==action]
        N = len(relevant_data)
        query = np.array([state['xdot'], state['ydot'], state['thetadot']]).T


        x = relevant_data[['xdot','ydot', 'thetadot']]-query
        weights = np.diag(np.array(np.sqrt(self.kernel((x['xdot'])**2 + (x['ydot'])**2 + (x['thetadot'])**2)))) #would be subtracted for distance but already subtracted query
        # we could also scale the effect of certain features
        query = np.hstack((query,np.ones((1,1))))
        x = np.array(x)
        x = np.hstack((x,np.ones((N,1))))
        y = np.array(relevant_data[['dx', 'dy', 'dtheta', 'dxdot', 'dydot', 'dthetadot']])
        z = weights @ x
        v = weights @ y
        lamb = np.diag(np.repeat(1,4)*1e-6)#4  = num_columns in x + 1, ridge 
        zpseudo = np.linalg.inv((z.T @ z)+lamb) @ z.T
        delta = query @ zpseudo @ v
        if mode=='delta':
            return delta
        if mode=='state':
            new_state = pd.DataFrame(np.array([[self.x+delta[0][0],
                                                self.y+delta[0][1],
                                                self.theta+delta[0][2],
                                                self.xdot+delta[0][3],
                                                self.ydot+delta[0][4],
                                                self.thetadot+delta[0][5]]]),
                            columns = ['x', 'y', 'theta', 'xdot', 'ydot', 'thetadot'])
            return new_state

    def kernel(self, distance):
        return 1/(1+distance**(2))





    def update(self, state):# for hardware, this is gonna poll camera instead
        self.x = state['x']
        self.y = state['y']
        self.theta = state['theta']
        self.xdot =  state['xdot']
        self.ydot = state['ydot']
        self.thetadot = state['thetadot']

    def state(self):
        return pd.DataFrame(np.array([[self.x, self.y, self.theta, self.xdot, self.ydot, self.thetadot]]),
                            columns = ['x', 'y', 'theta', 'xdot', 'ydot', 'thetadot'])

    def __str__(self):
        return f"x:{self.x}, y:{self.y}, theta:{self.theta}, xdot:{self.xdot}, ydot:{self.ydot}, thetadot:{self.thetadot}"



class FrogPlanner():
    def __init__(self, frog, path, search_depth=3):
        self.frog = frog
        self.path = path
        self.search_depth = search_depth
        self.next_action = None


    def plan(self):
        path = self.path
        state = self.frog.state()
        for i in range(100):
            best_action, cost = self.evaluate_actions(state, self.search_depth)
            self.next_action = best_action
            self.execute()
        print((self.frog.state(), path.p3))

    def evaluate_actions(self, state, depth):
        if depth == 0:
            return None, self.path.cost(state)

        best_cost = np.inf
        best_action = None
        for action in self.frog.actions:
            next_state = self.frog.predict(state, action)
            _, cost = self.evaluate_actions(next_state, depth-1)
            if cost < best_cost:
                best_cost = cost
                best_action = action
        return best_action, best_cost


    def execute(self):
        self.frog.update(self.frog.predict(self.frog.state(), self.next_action))# need a hardware execute function for real hardware



class Bezier():
    def __init__(self, width, height):
        self.p0 = np.array([width*0.1,height*0.1])
        self.p1 = np.array([width*0.9, height*0.1])
        self.p2 = np.array([width*1, height*0.9])
        self.p3 = np.array([width*0.1, height*0.3])
        vec0=self.derivative(0)
        self.theta0 = np.arctan2(vec0[1], vec0[0])

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
        points = np.empty((num_points,4))
        ts = np.linspace(0,1,num_points)
        for i,t in enumerate(ts):
            points[i,0:2] = self.curve(t)
            points[i,2:4] = self.derivative(t)
        return points, ts

    def plot(self, num_points):
        points,_ = self.discretize(num_points)
        plt.plot(points[:,0], points[:,1])
        plt.scatter([self.p0[0], self.p1[0], self.p2[0], self.p3[0]],
                    [self.p0[1], self.p1[1], self.p2[1], self.p3[1]],
                    color='r')


    def ang_diff(self, pose_vec, desired_vec):
        dot = np.dot(pose_vec, desired_vec)
        mag = np.linalg.norm(pose_vec) * np.linalg.norm(desired_vec)
        return np.arccos(dot/mag)


    def cost(self, pose, num_points=100):
        dist_weight = 1
        ang_weight = 5
        progression_weight = 2

        curve_points, ts = self.discretize(num_points)

        dists = spatial.distance.cdist(curve_points[:,:2], pose[['x','y']])
        dist_cost = dist_weight*np.min(dists)

        mindex = np.argmin(dists)
        min_t = ts[mindex]

        progression_cost = progression_weight * (1-min_t)

        closest_point = curve_points[mindex]
        pose_vec = np.squeeze([np.cos(pose['theta']), np.sin(pose['theta'])])
        desired_vec = closest_point[2:]
        ang_cost = ang_weight * self.ang_diff(pose_vec, desired_vec)


        return dist_cost + ang_cost + progression_cost


if __name__ == "__main__":
    width = 600
    height = 400
    path = Bezier(width, height)
    path.plot(100)
    frog = Robot(path.p0[0], path.p0[1], path.theta0, 0, 0, 0)
    planner = FrogPlanner(frog, path)
    planner.plan()
    plt.show()
