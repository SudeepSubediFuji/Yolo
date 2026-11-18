run_stage1 =>
last epooch info 25,26 and 27 suggest here that the model is not learning but memorizing as train/cls_loss shows the decreasing trend where as val/cls_loss shows decreasing trend with precision seems to be increasing and the mAP50 and mAP50~90 is also at increasing precision trend. The data that was really weird here is the learning rate.
In the graph for l1_loss(learning rate) it seems that at around the 20 epoch mark the l1_loss was increasing. the same 

25,232.528,1.37593,0.81159,0.00639,0.86393,0.71425,0.80751,0.51197,1.48019,0.90353,0.00802,2.38095e-05,2.38095e-05,2.38095e-05
26,241.157,1.37023,0.80479,0.00634,0.86393,0.71442,0.80882,0.51239,1.48931,0.89063,0.00805,1.98432e-05,1.98432e-05,1.98432e-05
27,249.824,1.37671,0.79985,0.0064,0.86802,0.71458,0.80884,0.51366,1.48441,0.89303,0.00802,1.5877e-05,1.5877e-05,1.5877e-05



run_stage_2 =>
The final precision metrics/precision seems to have increase but mA50~90 mark has gone down. There were alot of false data given so that it can also learn new characteristics but that seems that data has also shown up and down we can see the val/cls_loss ,box_loss has also seen to be decreasing trend in train but has gone up in the val. it has started memorizing rather than learning at this point(2nd finetuning stage) .
Mean average precision has also seen in downward in the final layer(head) same can be said to the mAP 50~90. Overall not a good data i think here.


25,272.1,1.31212,0.70389,0.00597,0.88931,0.71843,0.82906,0.53045,1.44698,0.84849,0.00789,2.38952e-05,2.38952e-05,2.38952e-05
26,282.531,1.31134,0.69399,0.00602,0.86142,0.74519,0.83061,0.52832,1.44676,0.85068,0.00787,1.99325e-05,1.99325e-05,1.99325e-05
27,292.902,1.31046,0.69432,0.00596,0.86636,0.74175,0.83046,0.52921,1.44777,0.85272,0.00786,1.59698e-05,1.59698e-05,1.59698e-05
28,303.56,1.30771,0.69132,0.00593,0.87944,0.73558,0.83033,0.5277,1.44953,0.85669,0.00787,1.20071e-05,1.20071e-05,1.20071e-05


In the result graph when comparing we can see that metrics/precision has gone wild as the precision has been fluctuating real hard . Mertics recall was smother as it should not have if the precision flucuate this hard then the recall should have also have the fluctuation data .But overall metrics/mAP50 and mAP50-95 shows very good curve meaning that it has read the real face training data had given good result but the false data has made the fluctuation of precision. The val box loss at validation seems to be getting bigger in the last epoch where train shows downward trend only(This also make me thinks it not learning it is rather memorizing or getting confused).