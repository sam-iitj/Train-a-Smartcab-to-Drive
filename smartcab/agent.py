import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np 

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        #self.q_matrix = np.random.randint(low=0, high=10, size=(14, 4))
        self.q_matrix = np.zeros((14, 4))
        self.gamma = 0.8   # Discount Factor 
        self.alpha = 0.5   # Learning Rate
        self.epsilon = 0.3  # Epsilon Value to trade off between exploration and exploitation. 

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def Qmax(self, state):
        return max(self.q_matrix[state, :])

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
         
        # TODO: Update state
        learning_agent_environment = self.env.agent_states[self]
        current_state = learning_agent_environment["location"] 
        heading = learning_agent_environment["heading"]
        destination = learning_agent_environment["destination"]
        current_manhattan_distance = self.env.compute_dist(current_state, destination)
        
        dist_X = (current_state[0] - destination[0])%6
        dist_Y = (current_state[1] - destination[1])%8

        if inputs["light"] == "green" and (self.next_waypoint == "left" and inputs["left"] == None and inputs["right"] == None and inputs["oncoming"] == None) or \
                                          (self.next_waypoint == "forward" and inputs["left"] == None and inputs["right"] == None) or \
                                          (self.next_waypoint == "right"):
            position_in_q_matrix = dist_X + dist_Y 
        else:
            position_in_q_matrix = self.q_matrix.shape[0] - 1

        print("Current Distance from destination : " + str(self.env.compute_dist(current_state, destination)))

        # TODO: Select action according to your policy
        if random.random() < self.epsilon:
            action = random.choice((None, 'forward', 'left', 'right'))
            action_index = (None, 'forward', 'left', 'right').index(action)
            self.epsilon -= 0.001                       # Decaying the value of epsilon, as we move towards the exploitation phase from exploration phase slowly. 
            print("Probabilistic action taken, with epsilon value : " + str(self.epsilon))
        else: #Choose action based on policy
            action = (None, 'forward', 'left', 'right')[list(self.q_matrix[position_in_q_matrix, :]).index(max(self.q_matrix[position_in_q_matrix, :]))]
            action_index = (None, 'forward', 'left', 'right').index(action)
            print("Deterministic action taken, with epsilon value : " + str(self.epsilon))

        print("Current action taken : " + str(action))

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Find the next state after the current action on the current state. 
        learning_agent_environment_new_state = self.env.agent_states[self]    
        new_state = learning_agent_environment_new_state["location"]   
        new_heading = learning_agent_environment_new_state["heading"]

        current_manhattan_distance = self.env.compute_dist(new_state, destination)
        
        newdist_X = (new_state[0] - destination[0])%6
        newdist_Y = (new_state[1] - destination[1])%8

        if inputs["light"] == "green" and (self.next_waypoint == "left" and inputs["left"] == None and inputs["right"] == None and inputs["oncoming"] == None) or \
                                          (self.next_waypoint == "forward" and inputs["left"] == None and inputs["right"] == None) or \
                                          (self.next_waypoint == "right"):
            newposition_in_q_matrix = newdist_X + newdist_Y
        else:
            newposition_in_q_matrix = self.q_matrix.shape[0] - 1

        # Find the index of the action in the new state which maximizes the Q value 
        newaction = (None, 'forward', 'left', 'right')[list(self.q_matrix[newposition_in_q_matrix, :]).index(max(self.q_matrix[newposition_in_q_matrix, :]))]
        newaction_index = (None, 'forward', 'left', 'right').index(newaction)

        # TODO: Learn policy based on state, action, reward
        self.q_matrix[position_in_q_matrix, action_index] = self.q_matrix[position_in_q_matrix, action_index] + \
                                                            self.alpha * ( reward + self.gamma * self.q_matrix[newposition_in_q_matrix, newaction_index] - \
                                                            self.q_matrix[position_in_q_matrix, action_index])
        print("Q_matrix ")
        print(self.q_matrix)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
