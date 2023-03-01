# Fast Graphlet Transformation with CUDA
Implements <a href="https://github.com/fcdimitr/fglt">fglt</a> using CUDA.

<h2>Dependencies</h2>
<ul>
  <li>NVIDIA CUDA Toolkit and nvcc compiler. Can be installed by running:
  
    sudo apt-get install nvidia-cuda-toolkit
    
  </li>
</ul>

<h2>Running</h2>
<ul>
  <li>Clone repo</li>
  <li>To run locally compile using:
  
    nvcc src/main.cu src/mtx.cu src/fglt.cu -O3 -o main

  And run by giving an .mtx file as argument.
  e.g.
  
    ./main auto.mtx
 
  </li>
  <li>To run in HPC copy everything to your HPC account. You can either use try.sh, auto.sh, great.sh, del.sh or all.sh. Move any of them to the top directory and run:
  
    sbatch <filename>.sh
    
with the corresponding .mtx file in the top directory.
    
  e.g.
  
    sbatch auto.sh

  </li>
</ul>
