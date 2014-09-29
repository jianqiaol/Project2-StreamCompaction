Project-2
=========

A Study in Parallel Algorithms : Stream Compaction

# PART 2 : NAIVE PREFIX SUM
### Questions 
* Compare this version to the serial version of exclusive prefix scan. Please
  include a table of how the runtimes compare on different lengths of arrays.
* Plot a graph of the comparison and write a short explanation of the phenomenon you
  see here.

![](https://raw.githubusercontent.com/jianqiaol/Project2-StreamCompaction/master/project2.png)



# PART 3 : OPTIMIZING PREFIX SUM
In the previous section we did not take into account shared memory.  In the
previous section, we kept everything in global memory, which is much slower than
shared memory.

## PART 3a : Write prefix sum for a single block
Shared memory is accessible to threads of a block. Please write a version of
prefix sum that works on a single block.  

## PART 3b : Generalizing to arrays of any length.
Taking the previous portion, please write a version that generalizes prefix sum
to arbitrary length arrays, this includes arrays that will not fit on one block.

### Questions
* Compare this version to the parallel prefix sum using global memory.
* Plot a graph of the comparison and write a short explanation of the phenomenon
  you see here.

# PART 4 : ADDING SCATTER
First create a serial version of scatter by expanding the serial version of
prefix sum.  Then create a GPU version of scatter.  Combine the function call
such that, given an array, you can call stream compact and it will compact the
array for you.  Finally, write a version using thrust. 

### Questions
* Compare your version of stream compact to your version using thrust.  How do
  they compare?  How might you optimize yours more, or how might thrust's stream
  compact be optimized.

# EXTRA CREDIT (+10)
For extra credit, please optimize your prefix sum for work parallelism and to
deal with bank conflicts.  Information on this can be found in the GPU Gems
chapter listed in the references.  

# SUBMISSION
Please answer all the questions in each of the subsections above and write your
answers in the README by overwriting the README file.  In future projects, we
expect your analysis to be similar to the one we have led you through in this
project.  Like other projects, please open a pull request and email Harmony.

# REFERENCES
"Parallel Prefix Sum (Scan) with CUDA." GPU Gems 3.
