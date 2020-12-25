# Allsky-StarCounter
### Count the stars in an Allsky-image using different techniques and methods.

Needs to be changed depending on the type of input data but framework is there.<br/>
<br/>
final1_starfind.py -- Star-Finder Method using DAOFIND (Stetson 1987) in Photoutils Library<br/>
final2_thresh.py -- Determining star count using threshold algorithm (detail in Bachelorthesis)<br/>
final3_nn.py -- Determining star count using simple neural network (detail in Bachelorthesis)<br/>
<br/>

For Starfinder it is important to determine the right parameters. Use this matrix for that:<br/>
<img src="images/method1_starfinder_params.png" width = 400> <br/>
For Threshold it is important to determine the right threshold. Use this threshold vs stars determined plot for that:<br/>
<img src="images/method2-thresholdstars_closer.png" width = 400> <br/>
For NN it is important to determine the right epochs/elarning rate. Use this matrix for that:<br/>
<img src="images/nn_results1.png" width = 400> <br/>
