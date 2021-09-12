# Predicting Pollution

Prediction Pollution is a Repository that was developed to analyse vehicle pollution ranging from rural areas to built up cities using OpenStreetMap data to extract real life road scenarios and simulate vehicle traffic. With data extracted directly from the simulation, a neural network was trained to recognise road conditions that could be extracted from real-life crowdsourcing navigation applications. The neural network was then applied to unseen simulations to reroute vehicle traffic to reduce vehicle pollution exposure without sacrificing the quality of routes in respect to time travelled, distance travelled & junctions encountered.

# The Results
Using the routing method proposed in conjunction with Dijkstra, the determined vehicle pollution-optimal routes showed vast improvements over an abstract version of current vehicle routing. The results are shown within the table below. 

Note:
The dynamic method of pollution-optimal routing was outperformed by static pollution-optimal routing. This was determined to be caused by short trip durations and updating of the vehicle route too regularly. The simple solution would be to increase trip durations & and increase the time between route updates, however due to the low CPU utilisation of the simulation software (SUMO: Simulation Of Urban Mobility) and the time constraints, this was not deemed possible.

