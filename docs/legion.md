# Legion

Legion is an asynchronous, task-based distributed execution engine based on the following concepts:

* Tasks
* Regions
* Partitioning
* Control Replication
* Coherence
* Mapping

### Tasks
Tasks are the basic units of the Legion parallel computation, their execution is non-preemptible and they can have multiple variants for each kind of processor (CPU vs GPU) that can be used, as well as for each memory layout of the arguments. Every task should be assigned a task ID and registerd with the runtime before the runtime starts. A particular task is designated as the **top-level task**, and it will be invoked by the start method. Tasks can call any C++ function, including those allocating/deallocating memory, but they cannot use packages other than Legion to implement parallelism or concurrency.

**Task Registration**: To register a task with the runtime, we need to provide the name of the task, the task ID, the kind of processor that can run the task (or other constraints), and pass two booleans indicting whether the task is launchable as a single task and/or as part of an index launch

**Task arguments and signature**: A task's signature contains an object representing the task, a verctor of physical region instances (used to pass data between tasks), a context object, with the task's metadata, and a pointer to the runtime. Arguments should not contain C++ pointers, and tasks should not refer global variables

#### Subtasks
Subtasks are tasks called by other tasks. To execute a subtask, the parent creates a `TaskLauncher` object, which takes two arguments: the ID of the task to be launched and `TaskArgument` object holding any arguments (and their size), which will be passed by value. TaskArgument objects should only be used for passing small amounts of data and cannot contain pointers nor futures. 

**Executing a task**: to execute a task, the parent calls `runtime->execute_task`, with the `TaskLauncher` as an argument. This method is not blocking, meaning that all tasks are launched asynchronously by their parent. The parent task will, however, wait until all children return before terminating.

#### Futures
Futures are placeholders for the results returned by tasks, since tasks are executed asynchronously. If we read the value of a future, the future will be immediately evaluated, with the parent blocking until the value is returned. 

#### Points, Rectangles, Domains
Legion defines the points, rectangles and domains types for launching sets of tasks, as well as other related applications. Various operations are defined on these types.

**Point**: a point is a tuple of $n$ integers

**Rectangle**: a rectangle is a set of points. It is defined using two points A and B, and includes all points that are greater or equal to A and less or equal to B.

**Domains**: a domain is equivalent to a rectangle, but with less type checking.

#### Index Launches
An index launch is a mechanism to launch multiple tasks at once, with an improved efficiency. In general, index launches are only meant to execute multiple instances of the same task in parallel. Index launches make use of a `IndexLauncher` (counterpart of the `TaskLauncher`) and are executed with `runtime->execute_index_space` (instead of `runtime->execute_task`). To pass arguments to tasks launched as part of an index launch, Legion uses an `ArgumentMap`, mapping each point in the task index space to an argument for the corresponding task. Similarly, the exeutor does not return a single future but a `FutureMap`, mapping each point in the index space to the future returned by the corresponding task.

### Regions
Legion uses **regions** to handle large quantities of data (smaller amounts can be handled using arguments and futures). 

**Logical region**: a logical region is a table whose rows are defined by an **index space** an whose columns are defined by a **field space**. The number of indexes in an index space is defined by the user, whereas the total number of fields in a field space is bounded. Logical regions do not actually hold data.

**Physical instance**: each physical instance of a logical region holds a copy of the data to be stored by that region. Multiple copies can be placed on multiple machines for increased performance.

#### Creating a physical instance
The most common way to instantiate a logical region is to pass it as a requirement to a substask. Legion will then automatically create a physical instance for the logical region when launching the subtask. We can add a **region requirement** to a subtask through the `add_region_requirement` method of the `TaskLauncher`. A `RegionRequirement` consists of, in addition to the logical region to be instantiated, a **privilege** (`WRITE_DISCARD`, `READ_ONLY`, `READ_WRITE`, `WRITE`, `REDUCE`), a **coherence mode** (`ATOMIC`, `SIMULTANEOUS`, `RELAXED`) and the parent region (or the region itself if the region has no parent).  

### Partitioning
Regions can be partitioned in multiple ways, and to arbitrary levels of recursion. The partitioning hierarchy defines a **region tree**. Partitioning is done dynamically at runtime; applications can create and destroy partitions. Partitioning can be costly, so it has to be done judiciously. Partitions only name subsets of data, they don't allocate any storage themselves. 

**Aliased/disjoint:** a partition is called aliased if its subregions overlap. A partition whose subregions do not share elements is called disjoint

**Complete**: a partition where every element of the partitioned region is at least in one subregion is called complete

**Ghost regions/stencils**: overlapping boundaries of blocks?

**Projection functor**: we can pass subregions to tasks as part of an index launch through the use of two arguments. The first argument is the logical partition to be used, and the second one is the identifier of a projection functor, which determines how subregions are assigned to tasks. Users can provide their own projection functor. 

#### Equal partitions
Equal partitions are the simplest type; a region is simply split into subregions of approximately the same size. An index space is used to determine the number of subregions. The points in the index space corresponding to the subregions are called **colors**. The partitioning is applied to the index space, rather than the logical region. In this way, logical regions sharing the same index space can reuse the same partitioning.

Equal partitions are both complete and disjoint

By default, equal partitioning of a multi-dimensional region only partitions the first dimension.


#### Partition by Field

Partitioning by field consists of specifying for each element of a region which subregion it should be in. 

#### Partition by Restriction
A **blocked** partition is one where a region is divided into blocks of the same size. In applications with stencil, it is useful to include ghost cells asjacent to the block.

A partition by restriction takes a **transform** and an **extent** as inputs. The extent $E$ is a rectangle with the desired size of each subregion, whereas the transform $T$ is a $n \times m$ matrix, where $n$ is the number of dimensions of the color space, and $m$ is the number of dimensions of the region. Each point $p$ in the color space corresponds to the points defined by the rectangle $Tp + E$ in the corresponding subregion.

#### Set-based Partitions
Set-based partitions allow us to defined partitions that are dependent on other ones. For example, we can define a **difference partition**, obtained by the set difference (by color) between two index space partitions. Similary, we can define a **union partition** or **intersection partition**.

#### Image and Pre-Image Partitions
Image partitions allow us to create new partitions that are compatible with already computed partitions from another region, e.g. if we want to create a partition for the nodes of a graph and a partition for the graph's edges. The image partition and pre-image partition allow us to do that by defining a pointer relationship betwene two regions, which in turn induces a partition of one region given the already existing partitioning in another one.

### Control Replication
In Legion, root tasks are responsible for launching subtasks to utilize each parallel machine, but this design choice can cause bottlenecks whenever a root task needs to launch ~16-32 tasks or more. To solve the problem, Legion uses **control replication**, which consists in executing multiple copies of the parent task in parallel, and each copy launches a subset of the child tasks. 

The most common case is to control-replicate only the top-level task. To use control replication, the programmer only needs to mark the task as replicable. It is the mapper's role to decide whether to replicate it and how to shard the work of analyzing the subtasks across the instances of the replicated tasks.

### Coherence
The coherence settings control how sibling tasks can do concurrently with a region. There are four coherence modes: **exclusive** where each task has exclusive access to a region for the entirety of the task's execution time, **atomic**, where sibling tasks must execute atomically with respect to a region, **simultaneous**, which implements a shared memory abstraction where race conditions are allowed, **relaxed**, where no restriction is in place, and the application can organize the data however it wants. 

#### Simultaneous coherence
In the simultaneous coherence scenario, a **copy restriction** is added to the underlying region, so that there can only be one physical instance, and all tasks need to use it. Because of this restriction, the region cannot be copied to a GPUs. Two important concepts, when it comes to coherence, are are **phase barriers** and the **acquire/release** directives. 

Phase barriers have an *arrival count*, and when an operation arrives at the barrier, the counter is incremented. The operation arriving at the barrier does not stop its execution. Operations can *wait* at a phase barrier, in the sense that their execution does not start until the phase barrier is triggered. Phase barriers have up to $2^{32}$ generations; each operation arrives or waits at a specific generation. When a phase barrier triggers, it is advanced to the new generation, and the waiters at the next generation can start their execution.

The **acquire** directive removes the copy restriction on the underlying region, which means the region can be copied anywhere, including on GPU accelerators. It should be used by tasks that are certain to be the only ones using a region. On the other hand, the **release** directive flushes all updates, and reintroduces the copy restriction.

In the most common scenario, phase barriers and the acquire/release directives are used together: phase barriers are used to enforce the proper synchronization of acquire/release.


### Mapping