# Super-Resolution-Image<br>
This is an algorithm about how to get super-resolution image from a single image, which is described in the paper "Glasner, Bagon, Irani - 2009 - Super-resolution from a single image"
I wrote the program in c++, and parallelized it with MPI and GPU. However, because of the limitation of my experience and knowledge in Computer Vision, I didn't finish all the steps described in the paper, so it's effect is not as good as described in the paper.
How to use it?<br>
First, you should install OpenCV.<br>
Then, compile it, using g++ for serial version, mpic++ for the MPI version, and nvcc for the GPU version, blahblahblah...
Run it with the command: ./sr baby.png<br>
