# Inference Requests traces

To evaluate the systems performance (throughtput & latency) of a serving system, we need a real or synthesized trace of client requests. 

## Orca approach
The authors of the [ORCA paper](https://www.usenix.org/conference/osdi22/presentation/yu) performed the end-to-end evaluation of their serving system using a simulated workload, which they generated as follows. First, the arrival rate of the requests is modeled using a Poisson process with various values of `$\lambda$`. As a consequence, the arrival time `$t_k$` of the `$k$`-th request `$r_k$` can be simulated by sampling from the Erlang distribution: 
```math
T=\{t_1,..., T_N\} \sim Erlang(k, \lambda)
```
The PDF of the Erlang distribution is as follows: 
```math
p_{T_k} = P(T_k=t)=\frac{\lambda^kt^{k-1}e^{-\lambda t}}{(k-1)!}
```
More practically, we can simulate each request arrival time `$t_k$`, using just a uniform random number generator and the formula: 
```math
t_k=-(1/k)\sum_{i=1}^k\log{u_i}
```
where `$u_i$` is the `$i$`-th random number generated from the uniform distribution `$U \sim Uniform([0,1])$`

Next the number of input tokens `$n_k$` and the number of tokens to generate (`$g_k$`) as output in each request are determined by sampling from the following uniform distributions:
```math
n_k \sim Uniform([32,512])
```
```math
g_k \sim Uniform([1,128])
```

In summary, each request `$r_k$` is generated using the three parameters `$t_k$` (arrival time), `$n_k$` (\# input tokens) and `$g_k$` (\# output tokens). The authors did not specify which values they used for `$\lambda$`, nor how the input tokens were generated.

## Proposed approach

To evaluate the performance of our system, we could use a similar approach as above, but instead of generating input samples artificially with `$n_k$` tokens sampled from a uniform distribution, we could use a real dataset such as `wikitext-103` and larger ones. We could generate the input samples by grouping the dataset entries by number of tokens and then for each request, pick one entry (with or without replacement) from one of the groups. The group itself could be picked randomly according to a uniform distribution.
