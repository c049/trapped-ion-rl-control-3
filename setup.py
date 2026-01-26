# Author: Ben Brock 
# Created on May 08, 2023 

from setuptools import setup, find_packages

setup(
    name="quantum_control_rl_server",
    version="1.0",
    description="Fork of v-sivak/quantum-control-rl, frozen version where the server (RL agent) and client (experiment or sim) communicate over tcpip",
    author="Volodymyr Sivak, Henry Liu, Ben Brock",
    author_email="bbrock89@gmail.com",
    url="https://github.com/bbrock89/quantum_control_rl_server",
    packages = ["quantum_control_rl_server"],
    requires=[],
)
