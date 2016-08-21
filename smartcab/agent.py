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
        self.q_matrix = np.random.rand(4, 4)
        self.gamma = 0.9

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        print("Next waypoint " +  str(self.next_waypoint))
         
        # TODO: Update state
        if inputs["light"] == "green":
          self.state = 0                    # "OK" state 
        elif inputs["light"] == "red":
          self.state = 1                    # "Stop!" state
        elif deadline < 0:
          self.state = 2                    # "Deadline Missed!" state   
        else:
          self.state = 3                    # "None" state
        
        # TODO: Select action according to your policy
        #action =  (None, 'forward', 'left', 'right')[random.randint(0, 3)]
        action = (None, 'forward', 'left', 'right')[list(self.q_matrix[self.state, :]).index(max(self.q_matrix[self.state, :]))]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Let's find the next state after the current action 
        next_state = None
        if deadline - 1 < 0:
          next_state = 2
        elif inputs["light"] == "green":
          next_state = 0
        elif inputs["light"] == "red":
          next_state = 1

        # Find the max a' Q(s', a') value  for the next state. 
        max_entry = max(self.q_matrix[next_state, :])
        
        self.q_matrix[self.state, (None, 'forward', 'left', 'right').index(action)] = reward + self.gamma * max_entry 
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
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
