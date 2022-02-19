Report for homework #1 of HPC course
===

# Graphs

+ Comparasing of time consuming for **regular matrix - vector** multiplication
![](points/reg-vec/img.png)

+ Comparasing of time consuming for **CRS matrix - vector** multiplication
![](points/crs-vec/img.png)

+ Comparasing of time consuming for **regular matrix - matrix** multiplication

![](points/reg-reg/img.png)

+ Comparasing of time consuming for **CRS matrix - matrix** multiplication
> For some reason, serial execution is better than parallel one. I think the reason is that crs-crs multiplication needs a lot synchronization (`#pramga omp atomic`). 

![](points/crs-crs/img.png)

+ Comparasing of time consuming for **regular matrix - matrix** multiplication on GPU

![](points/gpu/img.png)
 

# Code

Code can be found here:

