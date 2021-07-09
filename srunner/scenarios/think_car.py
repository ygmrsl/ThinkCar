#!/usr/bin/env python

# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Simple freeride scenario. No action, no triggers. Ego vehicle can simply cruise around.
"""

import py_trees
import carla

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (HandBrakeVehicle,
                                                                      StopVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               InTriggerDistanceToVehicle,
                                                                               DriveDistance)


class ThinkCarScenario(BasicScenario):

  

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,timeout=10000000):
        """
        Setup all relevant parameters and create scenario
        """
        
        # Timeout of scenario in seconds
        self.timeout = timeout
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 1.0
        control.brake = 0.0
        control.hand_brake = False
        self.control = control
        
        super(ThinkCarScenario, self).__init__("ThinkCarScenario",
                                       ego_vehicles,
                                       config,
                                       world,
                                       debug_mode,
                                       criteria_enable=criteria_enable)
        ego_vehicles[0].set_autopilot(False)
        ego_vehicles[0].apply_control(control)

        
        
        
    def _initialize_actors(self, config):
        for other_actor in config.other_actors: #create thinkcar vehicles
            transform_actor = other_actor.transform
            transform = carla.Transform(carla.Location(transform_actor.location.x,transform_actor.location.y,transform_actor.location.z),transform_actor.rotation)
            actor = CarlaDataProvider.request_new_actor(other_actor.model, transform)

            
            if other_actor.model.find("vehicle.") != -1:
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 0.0
                control.hand_brake = True
                actor.apply_control(control)
                
            

            
            self.other_actors.append(actor)
            
                        

    def _setup_scenario_trigger(self, config):
        """
        """
        return None
    def change_control(self, control):  # pylint: disable=no-self-use
        """
        This is a function that changes the control based on the scenario determination
        :param control: a carla vehicle control
        :return: a control to be changed by the scenario.
        Note: This method should be overriden by the user-defined scenario behavior
        """
        print("change control",control)
        return control
    
    def _create_behavior(self):
        """
        """
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        ego_vehicle_crash = InTriggerDistanceToVehicle(
            self.ego_vehicles[0],
            self.other_actors[0], #van
            5,
            name="Waiting for ego vehicle crash")

        ego_stop = StopVehicle(self.ego_vehicles[0],1.0,name="ego stop")
        
        sequence.add_child(ego_vehicle_crash)
        sequence.add_child(ego_stop)
        sequence.add_child(Idle())
        
        
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        for ego_vehicle in self.ego_vehicles:
            collision_criterion = CollisionTest(ego_vehicle)
            criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
