# Predicting Pollution

Prediction Pollution is a Repository that was developed to analyse vehicle pollution ranging from rural areas to built up cities using OpenStreetMap data to extract real life road scenarios and simulate vehicle traffic. With data extracted directly from the simulation, a neural network was trained to recognise road conditions that could be extracted from real-life crowdsourcing navigation applications. The neural network was then applied to unseen simulations to reroute vehicle traffic to reduce vehicle pollution exposure without sacrificing the quality of routes in respect to time travelled, distance travelled & junctions encountered.

# The Results
Using the routing method proposed in conjunction with Dijkstra, the determined vehicle pollution-optimal routes showed vast improvements over an abstract version of current vehicle routing. The results are shown within the table below. 

Note:
The dynamic method of pollution-optimal routing was outperformed by static pollution-optimal routing. This was determined to be caused by short trip durations and updating of the vehicle route too regularly. The simple solution would be to increase trip durations & and increase the time between route updates, however due to the low CPU utilisation of the simulation software (SUMO: Simulation Of Urban Mobility) and the time constraints, this was not deemed possible.

The results have been analysed in detail which indicate that the proposed routing method can reduce strain on infrastructure while reducing high pollution zones and lowering overall pollution exposure.

# How To Use
All Programming Files are located within /Scripts/.

A breakdown of each script is listed below in order of execution:
1) 1DataExtraction.py : A script that would automatically run simulations and extract data to unprocessed CSV Files.
2) 2CSVFileProcessor.py : A script that would automatically process each CSV file extracted from simulation while retaining simulation information.
3) 3ClassificationGenerator.py : A script that would classify data into classes based on severity of pollution levels, essential for training the neural network.
4) 4TrainNetwork.py : A script that would, you guessed it, train the neural network based on the classified csv data.
5) 5TestSimulation.py : A script that would implement neural network and single vehicle control within a simulation scenario and extract data to be reviewed.
6) 6InterpretData.py : A script that would interpret the results, to be manually interpretted and stored within spreadsheets for comparisons.

Automation.py was an attempt to automate the conversion of maps from OpenStreetMaps to XML files which are compatiable with SUMO.
EmissionsModel.h5 is a TensorFlow2 neural network model, this is the most accurate model that was trained.

# Future Development
Future development is currently limited until SUMO implements more advanced weather models that can influence pollution levels consistantly. There is however, a bypass that can be implemented. 

The suggested bypass:
Extract RoadID's and corresponding NodeID's for each Road and determine the distance between each road. From there, manually add a % of one roads emission values to nearby roads, for each road. However this can lead to inflated pollution values and a uniform dominoes effect that can greatly increase every roads pollution level in an unrealistic manor unless regulated correctly. This model would only account for the wind aspect of air pollution, which significance can vary depending on environment and number of vehicles.

# Notes
Due to github size limits, examples of simulations have not been included. Use requires intermediate ability with SUMO and manually altering config files for each scenario.
