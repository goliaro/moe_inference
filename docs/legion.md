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

### Partitioning

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