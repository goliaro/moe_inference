# Legion

Legion is an asynchronous, task-based distributed execution engine based on the following concepts:
* Tasks
* Regions
* Partitioning
* Control Replication
* Coherence
* Mapping

### Tasks
Tasks are the basic units of computations,

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