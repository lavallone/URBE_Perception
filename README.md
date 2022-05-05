# VISIOPE_project



## ABSTRACT

**Edge computing for Autonomous Driving: road signs detection and recognition in a light-weight model**

Edge computing, in particular Edge AI, is the deployment of AI applications on embedded devices. It's called "edge" because all the computations are done near the user at the edge of the network, close to where the data are located, rather than centrally in a cloud computing facility or data center. Due to the maturation of neural networks and to the advances reached in compute infrastructure, it's now possible to deploy AI applications "on the edge" to solve real-world problems, since AI algorithms are capable of understanding language, sights, faces, objects and other analog forms of unstructured information. 

Organizations from every industry are looking nowadays to increase automation and intelligence in their systems to improve efficiency and safety. To help them, computer programs need to execute tasks repeatedly and in a secure manner. Such jobs would be impractical to deploy on the cloud computing due to issues related to latency, bandwidth and privacy. That's why today the interest in edge computing is rising more and over. For machines to replicate human intelligence, it's required a deep neural network to be trained. After training, the network becomes an “inference engine” that can answer real-world questions, and it's exactly the inference time (that is the time of the network to make a prediction) that can play a fundamental role in an AI application. So, in all those real-time applications where the inference time is crucial, making the computation on the edge is needed. An example can be find in the field of Autonomous Driving.

As it can be imagined, the computational system behind a self-driving car is huge and extremely complex; it integrates many technologies, including sensing (lidars, cameras, radars), localization, decision making 
and perception (objects recognition and tracking). This is very challenging, since the design goal of autonomous driving edge computing systems is to guarantee the safety of the vehicles themselves and they need to process an enormous amount of data in real time with extremely tight latency constraints. 

For instance, if an autonomous vehicle travels at 50 km/h, and thus about 30 m of braking distance, this requires the autonomous driving system to predict potential dangers up to a few seconds before they occur.
Imagine a situation in which the vehicle has few instants to recognize a STOP road sign; the right recognition could prevent the passengers from doing a car accident or not.

Therefore, the faster the autonomous driving edge computing system performs these complex computations (inference time), the safer the autonomous vehicle is.

This project will then focus on the perception capability of an autonomous car, in particular on an object detection and recognition submodule that can be found in a real "autonomous driving algorithms system".

The objects that are going to be detected and classified are the traffic signs. As well as road signs recognition plays an important role as in the safety of each of our lives, it is in autonomous cars. 

More over is a challenging real-world problem because they seems to be easily detectable (they follow standard shapes and colors etc.), but nevertheless, that are some factors as the fact that are placed in outside environments, with different weather conditions, illumination variations, perspectives, occlusions, that makes traffic sign recognition a complex task to solve.

Fortunately in recent years, as it's known, we have seen the rapid development of deep learning technology, which achieves significant results in object detection and recognition. Especially thanks to convolutional neural networks. These networks can be huge in terms of learnt parameters, that can easily reach the millions of them. Both this aspect and the level of operations complexity involved in them, leads to a very high inference time, that is the most important aspect when dealing with object recognition in real-time (that is, while the car is driving autonomously). 

So, the main two goals of this project will be to: first, reach a good performance on the detection and recognition task and second, decrease the inference time by manipulating and simplify somehow the  neural network under consideration. To do so we will have to find and/or create a dataset regarding the topic, then building our light-weight network (to have as few operations involved as possible, but keeping a reasoning accuracy on the dataset) and finally training it. After all, having the "best" model ready, it will be time to make some real experiments on different embedded devices in order to be able to measure its inference time and the Frame Per Seconds (FPS) that the model can compute. Only with these final measurements we'll be able to decree the quality of the model as a component of an autonomous driving subsystem.
