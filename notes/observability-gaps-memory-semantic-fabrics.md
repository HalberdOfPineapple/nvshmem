# Observability Gaps in Memory-Semantic Scale-Up Networking

What FabricPerf measures, what it structurally cannot, and which of those blind spots are general properties of memory-semantic fabrics rather than quirks of one library. NVSHMEM is used as the lens because it exposes the raw primitives of the fabric (stores, loads, atomics, fences, multicast, bulk-copy engines, and network fall-backs) instead of one fixed protocol, so whatever it reveals about observability tends to transfer to NCCL, UALink, Scale-Up Ethernet and rocSHMEM as well.

Sources: FabricPerf (Huang and Wu, SIGCOMM '26, `papers/26_FabricPerf.pdf`), Demystifying NVSHMEM (Ma, Shen et al., arXiv 2606.05951, `papers/26_Demystifying_NVSHMEM_...pdf`), NVSHMEM source on branch `devel` at 7bb2e99 (3.9.0.dev0), and the companion note `data-movement-paths.md`. Written 2026-09-29.

---

## 0. The short version

FabricPerf turns "measure the scale-up network" into "profile the NCCL communication kernel": it timestamps five landmarks of NCCL's data-then-fence-then-flag protocol from inside the kernel, synchronises GPU clocks with a PTP-style handshake (GPTP), and models fabric traffic as memory traffic so CUPTI counters can be reused. That is a good first step, but it silently assumes six things that are true of NCCL ring kernels and false of memory-semantic fabrics in general:

1. **There is a "packet" delimited by a fence and announced by a flag.** Memory-semantic traffic has no such unit. The same bytes can travel as thread stores, as a bulk copy issued by the Tensor Memory Accelerator (TMA), as a copy-engine transfer with no SM involved, as a multicast store the switch fans out, as an atomic, or as an LL/LL128 message where data and flag share one 16- or 128-byte store and there is no fence at all.
2. **Both endpoints run code.** One-sided operations by definition do not execute anything at the target. A remote store, load or atomic "arrives" as a state change in another GPU's memory system. Arrival is not an event any software thread experiences, so there is no receiver-side timestamp to take, and any timestamp you do take from a polling loop is bounded by the polling period. This is the root of the two-clock problem the supervisor raised, but the problem is deeper than clock synchronisation: for loads and fetching atomics the round trip is the natural unit and needs no second clock, while for non-fetching stores no clock on the initiator can observe arrival at all.
3. **The fence marks "entered the fabric."** A system-scope fence is a *drain plus ordering* instruction. Its duration is a function of everything in flight, it serialises the pipeline, and it is itself a large fraction of what is being measured (1.5 to 3 µs against 5 to 13 µs latencies). FabricPerf treats it as an instant; it is better understood as a sensor of queue depth, the very thing FabricPerf says the missing NIC queues took away.
4. **Only stores matter.** Reads (`get`, `ld`) and atomics are request/response traffic whose bottleneck is usually the issuer's outstanding-request capacity, not fabric bandwidth (bulk `get` peaks at 141 GB/s against 313 GB/s for `put`; scalar `g` at 9 GB/s against 172 GB/s for `p` on H200). They also expose address-level contention that no per-channel measurement can see.
5. **The switch is a wire.** With NVLink SHARP a single `multimem.st` becomes N deliveries and a single `multimem.ld_reduce` becomes N reads plus an in-switch reduction. One sender timestamp, N arrivals, and an actor (the switch) with no clock and no probe point.
6. **The communication kernel is a separable kernel.** The trend that motivates device-initiated communication is fusing communication into compute kernels (DeepEP, fused GEMM + all-gather, megakernels). There the "transport layer" is inlined library code with no fixed instruction landmarks, or a copy engine with no kernel at all. Probing a known NCCL kernel at known instructions does not generalise.

Each of these becomes a research gap in section 2. Section 3 sketches directions that follow from them and are *not* "FabricPerf on NVSHMEM". Section 4 lists the NVSHMEM code paths that make these gaps concrete and measurable on Beta.

---

## 1. What FabricPerf actually measures

It is worth being precise, because the paper's framing ("packet-level", "transport layer") is more general than the mechanism.

**Observation unit.** A "packet" is one fence-delimited run of stores from one thread block (which FabricPerf equates with a "channel") followed by a control-plane write (the "tailer"). The paper admits in §7 that these are KB to MB units, far coarser than the hardware's flits. The 128-bit "MTU" in Figure 5 is the store width, not a packet size.

**Timestamps.** Five per packet (Figure 6): T1 receiver posts CLEAR, T2 sender observes CLEAR, T3 sender's first store, T4 sender's fence returns and flag is written, T5 receiver's poll loop sees the flag. Plus S2/S5 spin counts so that samples where the poller found the value on its first iteration (and therefore cannot bound arrival time from above) are discarded from latency, but kept for throughput. Probes attach at the SASS level with Neutrino to three landmarks: the `fence`, the loop branch (`bra`), and the first `st`.

**Clock synchronisation.** GPTP: one leader GPU, offset-only correction (GPU clocks have no `adjtime`), 128-bit messages over the fabric, resynchronised at every kernel because inter-kernel jitter is 36 µs while intra-kernel jitter is about 100 ns; triangulation error 106 ns average, 365 ns worst case. It assumes symmetric path delay in the two directions, as PTP does.

**Physical layer.** Fabric traffic is modelled as memory traffic (Table 2) so CUPTI's XBAR, DRAM and L2 counters apply. The model is specific to NCCL's staged buffers: network buffers are persistent and should hit in the LLC (last-level cache, L2 on NVIDIA GPUs), user buffers are streaming and should miss.

**Results.** Latency is not a fixed 6 µs; it trades off against channel count and thread count; channels are imbalanced (P99 twice P50); LLC hit rate collapses to ~0% at high bandwidth because network buffers are evicted. Two fixes: work stealing across channels, and PTX eviction-priority hints.

**Scope actually exercised.** NCCL 2.28 ring SendRecv, AllGather and ReduceScatter, varying `NCCL_CTAS`, on H100 NVL8 and GB200 NVL16. Store-oriented only; the `ld` case is a paragraph in §7. No atomics, no multicast, no TMA, no copy engine, no host-initiated path, no fused kernels.

**Limitations the authors state.** No network-layer visibility (routing, packetisation, L2 latency), no switch visibility, channel imbalance cannot be attributed to a cause "regrettably".

---

## 2. Research gaps

The criterion for inclusion: the gap must exist for *any* memory-semantic scale-up network, not just for NVSHMEM. NVSHMEM is cited as evidence that the phenomenon is real and as the place to experiment. Things that are NVSHMEM-specific and therefore *not* gaps: the symmetric-heap allocator, teams and `pSync` layout, the host/device dual API, bootstrap, the on-stream wrappers.

### Gap 1. There is no canonical "packet" to timestamp

FabricPerf's unit exists because NCCL's simple protocol has one. NVSHMEM shows that memory-semantic traffic has at least seven mechanisms with different observable events (see `data-movement-paths.md`, section B):

| Mechanism | Initiator | Where the bytes are moved | Software-visible "send" event | Software-visible "done" event |
|---|---|---|---|---|
| Thread `st.global` to peer VA | each SM thread | LSU → L2 → NVLink | the store instruction | none; `quiet` is only `__threadfence_system` (`nvshmemi_common_device.cuh:1185-1212`) |
| TMA `cp.async.bulk` | one elected thread | TMA unit, async proxy | the bulk instruction | `cp.async.bulk.wait_group` per issuing thread; ordered by `fence.proxy.async`, not `membar` |
| Multicast `multimem.st` | each thread | switch fans out to N GPUs | one store | none; N arrivals |
| Copy engine `cudaMemcpyAsync` | host | DMA engine, no SM | stream event | stream event; no kernel to probe at all |
| Remote atomic `atomicAdd_system`, `atom.*` | thread | read-modify-write at remote L2 | the atomic | fetching variants return a value (a round trip) |
| LL / LL128 | thread | 16 B or 128 B store carrying data and flag | the store | receiver polls the flag embedded in the data (`coll/fcollect.cuh:141-284`) |
| Proxy / IBGDA (fall-back) | thread writes request | CPU thread + NIC, or GPU-driven NIC | request written to ring | CPU/NIC completion counters (`proxy_device.cuh:50-130`) |

Three consequences:

- **No fixed instruction landmarks.** FabricPerf needs a `fence`, a `bra` and a first `st` in a known order. LL128 has no fence; TMA has no `st`; the copy engine has no kernel; multicast has one `st` for N deliveries.
- **The unit of measurement depends on the mechanism.** For thread stores the meaningful unit is a warp's coalesced wave; for TMA it is the bulk group; for LL it is the 16-byte element; for multicast it is the fan-out set.
- **The same call takes different mechanisms at run time.** `nvshmem_putmem` dispatches on peer reachability, TMA policy, alignment and build flags (`nvshmemi_common_device.cuh:1444-1472`). An observability tool that assumes one mechanism misattributes the others.

**NCCL's own device API makes the same point.** NCCL 2.28, the version FabricPerf profiles, also ships a device API (Hamidouche et al., "GPU-Initiated Networking for NCCL", arXiv 2511.15076) with three modes: LSA (Load/Store Accessible: plain loads and stores into peer windows, the application writes its own flags and fences), Multimem (`multimem.st` through NVLink SHARP), and GIN (one-sided `put` with a `signal` for remote completion and a `counter` for local completion, where the paper states ordering is provided by the signal "without requiring explicit fence operations" and operations are unordered by default). FabricPerf's experiments use only the host-launched collectives (nccl-tests with `NCCL_CTAS`, ring algorithm), whose Simple protocol is the one realisation of "data, then ordered notification" that has a fence instruction, a plain-store flag and a poll loop at fixed places in a known kernel. The abstract pattern survives across Simple, LL, NVSHMEM put-with-signal and GIN put-with-signal; the probe points do not.

**Research question.** What is the right observation unit for memory-semantic traffic, and where should observation attach when the initiator is not an SM thread? One candidate answer is elegant precisely because it is *not* protocol-based: classify every memory instruction by its *destination virtual address*. In every memory-semantic fabric the remote GPU's memory is mapped into a known VA window (NVSHMEM: `peer_heap_base_p2p[pe] + offset`, Demystifying §IV.C; NCCL symmetric windows and UALink do the same). So "is this store remote, and to which PE" is decidable from the address alone, for thread stores, TMA, multicast (multicast VA is a separate window) and atomics alike, without knowing the protocol. FabricPerf's landmark-based probing cannot handle fused kernels; address-based classification can.

### Gap 2. Arrival is not an event; the two-clock problem is a symptom

This is the supervisor's point, stated more generally.

In a two-sided network the receiver NIC executes something on arrival, so arrival has a natural timestamp. In a one-sided memory-semantic network, arrival is a state change in the target's L2 or HBM. No thread on the target runs. The only ways software learns of it are:

1. **A target-side poll loop** sees a value change. Its resolution is the loop period, which is one L2 load round trip (hundreds of ns) or, if the poll is over the fabric, a full remote load. FabricPerf's S5 handling discards samples where the poll found the value immediately, which biases the retained samples toward the slow tail and throws away exactly the fast cases that would show the fabric's floor.
2. **An initiator-side fetching operation** (load, `atomic_fetch_add`, `get`) returns. That is a round trip measured on one clock. It needs no synchronisation, but it folds in the return path.
3. **A later acknowledgement** written back by the target after it noticed. That is a three-hop measurement (store, poll, ack) on one clock and is what ping-pong latency tests do (`perftest/device/pt-to-pt/shmem_put_ping_pong_latency.cu`).

FabricPerf chooses (1) plus clock synchronisation. The costs are structural, not implementation details:

- **GPTP must be re-run per kernel** because inter-kernel jitter is 36 µs and there is no frequency discipline. A persistent kernel (DeepEP's dispatch, a megakernel) has no kernel boundary to resynchronise at, and drift of about 1.5 µs per run between nodes accumulates.
- **GPTP assumes symmetric delay** in the two directions between leader and follower. In a switched fabric with per-direction lanes, unequal loads and possibly different routes, the asymmetry is unknown and is *itself one of the quantities we would like to measure*. The reported 106 to 365 ns triangulation error is a consistency check among followers, not a bound on the asymmetry error.
- **Synchronisation traffic uses the fabric under test** and stalls the kernel for 5 to 10 µs per kernel.
- **Non-fetching stores are unobservable from the initiator** on the NVLink path. NVSHMEM's `quiet` for peer-reachable PEs is a single `__threadfence_system()`; it orders but does not report when the stores landed. There is no counter to read. Only the proxy and IBGDA paths have completion counters, and those are maintained by the CPU or NIC.

**Research question.** Can one-sided latency be observed *without* two clocks, or with a clock-sync design whose error is bounded independently of path asymmetry? Options to evaluate:

- **Round-trip decomposition with fetching atomics.** An `atomic_fetch_add` to the remote flag is a store whose completion is observable on the initiator. Comparing store-then-fetch against fetch alone separates one-way store latency from the return, under an asymmetry assumption that can itself be tested by swapping roles.
- **Timestamp-carrying stores.** The sender writes its `%globaltimer` into the payload (LL-style, 8 B data + 8 B flag). The receiver records its own `%globaltimer` when it sees it. Without sync this gives only *relative* latency changes over time on a fixed pair, but that is enough for tail analysis, imbalance and interference studies, which are the questions FabricPerf actually asks.
- **Self-calibrating symmetric probes.** Use the fabric's own symmetry: A→B and B→A one-way estimates share the offset with opposite sign, so their sum is offset-free. With multiple paths (direct, via C) one can solve for asymmetry rather than assume it away.
- **Ground truth from LL flags.** LL and LL128 give per-16-byte and per-128-byte arrival observations at the receiver (data and flag arrive atomically). Treating them as a measurement primitive rather than a protocol gives arrival resolution far finer than a fence-delimited message.

### Gap 3. Visibility, ordering and completion are different events, and the fence is a sensor, not a marker

For one store there are at least six distinct moments: issued by the thread; drained from the SM's store path; ordered with respect to other stores (what `fence` provides); entered the NVLink port; written into the remote L2; visible to a remote SM after its own acquire. FabricPerf collapses these into T3 (first store), T4 (fence returned) and T5 (remote poll succeeded), and equates T4 with "entered the fabric".

`fence.sys` / `__threadfence_system` is a drain-and-order instruction. Its latency is proportional to what is outstanding ahead of it, it blocks subsequent stores until drained, and FabricPerf measures it at 1.6 to 3.1 µs, a third of the end-to-end latency on GB200. Two observations follow:

- **The fence perturbs what it measures.** Probing "existing fences" is only non-invasive when the protocol already has one per packet. LL/LL128 and `put` without `quiet` have none. Adding one changes the pipeline.
- **The fence duration is a queue-depth signal.** FabricPerf laments that "packet queues of NICs expose precise arrival timing" and that fabrics have no equivalent. But the time a fence takes to return is a direct function of how many stores are in flight from that thread and, through cumulativity, from the block. Calibrated against a known injection rate, fence latency is a software-readable proxy for in-flight bytes, i.e. a queue occupancy sensor the fabric does not otherwise expose. NVSHMEM's `perftest/device/pt-to-pt/shmem_flush_bench.cu` and `shmem_nbi_issue_rate.cu` already measure the two ends of this (issue rate and drain cost) separately.

There is also **flow control**, which FabricPerf models at the protocol level (T1/T2, the CLEAR credit, "cwnd = 1"). One-sided puts have no receiver credit at all: the target buffer always exists, and back-pressure happens inside the memory system (store-buffer and NVLink credit stalls) where it is invisible. The observable symptom is the issuing thread stalling on a store instruction, which only instruction-level sampling (stall reasons in CUPTI/Nsight) can see, and not per destination.

**Research question.** Non-perturbing completion observation for stores on load/store fabrics: which existing side effects (fence latency, store-stall cycles, poll-iteration counts, TMA `wait_group` latency) carry information about the invisible in-flight state, and how well can they be calibrated to reconstruct occupancy and completion time?

### Gap 4. Reads and atomics are first-class traffic with different physics

FabricPerf is store-only and says `ld` "could be simpler as all timestamps are local". Locality of timestamps is the easy part; the hard part is that read and atomic traffic behaves unlike store traffic:

- **The issuer's outstanding-request capacity bounds throughput**, not the link. Demystifying §VII.B: on H200 NVLink, bulk `get` reaches 141 GB/s against 313 GB/s for `put`; scalar `g` reaches 9 GB/s against 172 GB/s for `p`, because each thread must receive the value before its next dependent operation. The interesting metrics are memory-level parallelism per thread/warp/SM, response reordering, and the asymmetry between small requests and large responses on the two directions of the link.
- **Atomics create address-level contention.** NVSHMEM signals are `atomicAdd_system` on a remote 8-byte word (`nvshmemi_signal_op`, `nvshmemi_common_device.cuh:1519-1560`); DeepEP uses remote atomics for token counts and credits. N senders hitting one address serialise at one L2 slice on one GPU: an incast at cache-line granularity. FabricPerf's per-channel view cannot see per-address contention, and its throughput model (bytes per second) is the wrong metric for a 8-byte RMW stream where the cost is serialisation, not bandwidth.
- **Read traffic crosses the fabric twice** (request and data), so the memory-traffic model of Table 2 needs a different accounting, and CUPTI's NVLink counters (bytes received/transmitted) cannot separate request from response.

**Research question.** Observability for request/response memory-semantic traffic: measuring outstanding-request limits and response ordering per SM, and per-address contention for atomics, ideally from the same address-based classification as Gap 1.

### Gap 5. In-fabric operations: one send, N arrivals, and a switch with no clock

NVLink SHARP (NVLS) lets a store to a multicast address be replicated by the NVSwitch to every GPU in the team, and lets a `multimem.ld_reduce` read the same address on all GPUs and return their sum (`coll/reduce.cuh:148-223`, `nvshmemi_mcast16_store_threadgroup` in `nvshmemi_common_device.cuh:222-236`). NCCL uses the same hardware. Everything in FabricPerf's model breaks:

- One sender timestamp maps to N arrival events on N clocks; "latency" becomes a distribution across receivers and *arrival skew* across the fan-out is the new tail metric.
- The switch performs work (replication, reduction) and has no probe point; its residency time can only be inferred by subtraction.
- Memory traffic at the sender no longer equals fabric traffic: one 16-byte store becomes N × 16 bytes of switch egress, so "fabric bandwidth = memory bandwidth" (the premise of FabricPerf's physical-layer design) is false by a factor of N.
- Completion semantics are unclear from software: when is a `multimem.st` "done"? When the switch accepted it, or when all N copies landed? `quiet` gives no answer.

**Research question.** Observability of in-network compute in memory-semantic fabrics: fan-out skew, reduction latency as a function of participant count and data type, and what completion means. This needs multi-way (N-clock) rather than pairwise synchronisation, or a clock-free design based on the reduction's return value.

### Gap 6. Contention is address-determined, and the "channel" is not a link

FabricPerf finds channel imbalance (P99 twice P50 from 12 channels onward) and cannot attribute it: "channels here correspond to cores/memory pipelines, rather than PHY link/port(s), with the mapping instructed by the network layer." In a memory-semantic fabric the path of a store is SM → load/store unit → crossbar → L2 slice (chosen by an address hash) → NVLink port (chosen by an address or route hash) → switch → remote L2 slice (address hash) → HBM channel (address hash). Every contention point is selected by *address*, not by flow. Two thread blocks writing to addresses that hash to the same L2 slice or port interfere regardless of which "channel" they are, and local compute traffic (a GEMM's HBM reads) shares the same L2 and crossbar. FabricPerf's LLC study is one instance of this interference; it is not a special case.

The address-to-path mapping is fixed once the symmetric heap is mapped (the VA window per peer is fixed at init), so the same buffer always takes the same ports. That is both the cause of persistent imbalance and the handle for measuring it: a controlled address sweep, correlated with per-link hardware counters (NVML NVLink counters, CUPTI `nvlink` metrics per link), can recover the hash and attribute traffic to physical links.

**The L1 is missing from the model, and "bypass" is asserted, not measured.** FabricPerf's Table 2 marks L1 as bypassed for every collective on the argument that network data is external to the core; no L1 counter is reported and the XBAR metric it uses is by definition the L1's miss output. Receive-side loads do bypass L1, but for coherence rather than locality: L1 is not coherent with remote writes, so every load of a remotely written buffer must be volatile, relaxed/acquire at gpu or system scope, or preceded by an acquire fence. NVSHMEM's LL receives use `ld.volatile.global` (`nvshmemi_common_device.cuh:1910`) and its wait loops read through `volatile` pointers; NCCL's receive path does the same. That rule is what forces each poll iteration to L2 and sets the poll period behind Gap 2. Local source reads, by contrast, are ordinary cached loads (`nvshmemi_memcpy_threadgroup`, `:327-345`) that merely stream once. Stores are write-through, so the L1-side effect that matters is not hits but the load/store unit's coalescing of per-thread 16-byte stores into the sector transactions that enter the crossbar and NVLink, which is where the fabric's actual packetisation is decided. TMA and multicast skip L1 through the async proxy for reasons unrelated to locality. And the SM-side structures the model omits (outstanding loads, store queue depth, MIO queues) are exactly what makes `get` issuer-bound and what a fence drains, so Gaps 3 and 4 cannot be expressed in an L2-plus-DRAM model at all. The unexplained 15% error on ReduceScatter's read flow, the one flow mixing a local and a network operand, is consistent with this omission.

**Research question.** Attributing memory-semantic traffic to physical paths (L2 slice, port, switch plane) from software-visible addresses, and quantifying compute/communication interference in the shared memory pipeline. The second half is the more important one: overlapping communication with computation is the whole reason device-initiated communication exists, and no current tool measures what the overlap costs the communication side.

### Gap 7. Reachability tiers, silent fall-backs and degraded paths

FabricPerf treats its NVL16 domain as flat and finds intra-node and inter-node latencies nearly equal, which is expected on a healthy single-hop switch. But the *path taken* by an operation is decided per peer at init and can silently change:

- NVSHMEM maps a peer's heap only if `cudaDeviceCanAccessPeer` and native atomics succeed (Demystifying §V.A); otherwise the same `put` goes to the proxy or IBGDA and crosses a NIC. On Beta the IMEX channel is broken, so `NVSHMEM_DISABLE_MNNVL=1` sends cross-node traffic over InfiniBand even though the NVLink rack exists (`dev-guide-ring-fabric-handle-error.md`; memory handle type selection at `src/host/mem/mem_heap.cpp:133-139`).
- The GB200 rack has two levels (four-GPU compute tray, then switch trays); FabricPerf's "consistency within fully-connected topology" does not tell whether a pair traverses one or two switch chips.
- NVLink is lossless via CRC and link-level replay; FabricPerf discounts loss entirely ("packet loss rate is often not a concern"). Replays, link-width degradation and lane errors appear as latency tails with no attribution. NVML exposes replay and CRC error counters per link, but nothing per kernel or per destination.

**Research question.** Per-operation path attribution (which transport, how many switch hops, which link) and detection of degraded reachability, so that tails can be attributed to a route rather than averaged into "channel imbalance".

### Gap 8. Observation cost at nanosecond time scales, and what to sample

Latencies are 1 to 10 µs; one store is tens of ns; a probe costs 20 ns to shared memory or 200 ns to global memory. FabricPerf keeps overhead at 1% by probing only the fence-delimited unit, which is exactly the coarse unit of Gap 1. Finer units (per warp store wave, per 16-byte LL element) cost proportionally more, and online mode already costs 9%.

Two things are missing: a **sampling theory** for memory-semantic traffic (what does "sample one packet in 100" mean when a "packet" is 1024 threads each issuing 16-byte stores?), and **passive observation** that reuses side effects already present: NVSHMEM's wait loops read `%globaltimer` when timeout polling is compiled in (`wait/nvshmemi_wait_until_apis.cuh:25-60`), the proxy path exports issue/complete counters, TMA exports `wait_group` completion, and every fence has a measurable duration.

**Research question.** Sampling and passive-observation designs whose overhead is independent of the number of issuing threads.

### Gap 9. SIMT aggregation semantics: whose timestamp is the packet's?

A "flow" in this world is 32 lanes of a warp issuing coalesced stores, hundreds of warps per block, dozens of blocks. The memory system may split or merge those stores; ordering among threads is undefined without fences; fence scope (`cta`, `gpu`, `sys`) changes what is ordered; TMA's async proxy has its own ordering (`fence.proxy.async`). FabricPerf timestamps once per block, from "the polling thread". That silently defines the packet's arrival as the time its *last* byte (the one the flag orders behind) is visible to *one* thread. Whether the first, median or last store's arrival is the right statistic is the load/store analogue of the packet-latency versus flow-completion-time distinction, and there is no established answer.

**Research question.** Defining and measuring per-warp and per-block completion distributions for a wave of stores, and relating them to per-thread stalls.

### Gap 10. The communication "layer" is inlined into compute kernels

FabricPerf's central claim is that the transport layer "is the lowest software layer that can be inspected". For NCCL that layer is a kernel with a stable instruction shape. For device-initiated communication it is not:

- NVSHMEM device APIs are header-only templates inlined into the user's kernel (`src/include/non_abi/device/...`); the "transport" is a few `st.global` instructions among the user's arithmetic.
- DeepEP calls internal helpers (`nvshmemi_ibgda_put_nbi_warp`, `nvshmemi_ibgda_amo_nonfetch_add`) directly and builds its own pipeline with warp specialisation (Demystifying §VIII).
- Fused GEMM + all-gather kernels (ParallelKittens, Mercury, TMA from shared-memory tiles as in `examples/tma-smem.cu`) issue communication from the middle of a compute loop.
- Host-initiated paths use copy engines with no kernel at all.

There is no fixed kernel to probe, and the same address is written by user code and library code. This is the strongest argument that a general tool cannot be "an NCCL-kernel profiler": it needs to classify memory instructions by *what they touch* (Gap 1's address-based decoding), or hardware support at the memory-system level.

**Research question.** Observability for communication that is fused into computation: attribution of memory-instruction latency and stalls to remote versus local destinations within one kernel, at overhead low enough to run on production fused kernels.

### Gap 11. Zero-copy delivery has no receiver-side per-packet event

FabricPerf derives throughput and packet rate by bin-counting receiver arrival timestamps, and its memory model assumes a persistent "network buffer" that the receiver copies out of (Table 2, flows ❸ to ❺). NVSHMEM `put` writes directly into the destination symmetric buffer, and NCCL's user-buffer registration does the same. Then there is no intermediate buffer, no per-packet receiver copy, and the only receiver event is the final flag or barrier. Packet rate at the receiver is unobservable in software; only hardware counters (NVLink bytes received) remain, and they are per link and per time window, not per operation. The LLC model also does not apply: there is no persistent buffer to keep resident.

**Research question.** Receiver-side throughput observability for zero-copy memory-semantic delivery.

---

## 3. Where this points

The gaps cluster into three themes that are each larger than "build a better FabricPerf":

**A. Clock-free or asymmetry-aware one-sided latency measurement (Gaps 2, 3, 5).** The primary open problem. Candidate designs: fetching-atomic round-trip decomposition; timestamp-carrying LL-style stores for relative latency; symmetric-pair probes that cancel offset; using fence and TMA `wait_group` latency as calibrated completion sensors; N-way skew measurement for multicast. The evaluation question is what precision each design achieves *without* GPTP and how each degrades under load asymmetry, which GPTP cannot even detect.

**B. Address-based, protocol-agnostic traffic classification (Gaps 1, 4, 6, 7, 10).** Every memory-semantic fabric maps remote memory into VA windows. Instrumenting memory instructions by destination window (with NVBit or Neutrino, or compiler-inserted) classifies stores, loads, atomics, multicast and TMA uniformly, works inside fused kernels, and yields per-destination, per-address-slice statistics that can be correlated with per-link hardware counters to recover path attribution. This is the piece that makes the "channel imbalance, cause unknown" finding answerable.

**C. Interference and completion in the shared memory pipeline (Gaps 3, 6, 8, 9, 11).** Communication and computation share L2, crossbar and HBM; back-pressure is invisible; completion is not reported. The research object is the store's life inside the memory system rather than "the network", and the metrics are occupancy, stall attribution, and completion distributions, measured through side effects rather than probes.

What we should *not* do: re-implement T1 to T5 on NVSHMEM's put/wait and call it a contribution. That would inherit every assumption above and add nothing but a different protocol.

---

## 4. NVSHMEM hooks that make the gaps concrete on Beta

For our own experiments, the places in the tree where each phenomenon can be observed or provoked. Nothing here has been run yet.

| Gap | Where in the source | What to do with it |
|---|---|---|
| 1, 10 | `src/include/non_abi/device/common/nvshmemi_common_device.cuh:1444-1472` (dispatch), `data-movement-paths.md` | Force each mechanism (`NVSHMEM_TMA_POLICY`, NVLS via collectives, copy engine via host API, proxy via `NVSHMEM_DISABLE_MNNVL`) and record which observable events exist for each |
| 2 | `perftest/device/pt-to-pt/shmem_put_ping_pong_latency.cu`, `shmem_atomic_ping_pong_latency.cu`, `shmem_p_ping_pong_latency.cu` | Baseline one-clock round trips; decompose with fetching atomics |
| 2, 8 | `src/include/non_abi/device/wait/nvshmemi_wait_until_apis.cuh:25-60` | Poll-loop structure and existing `%globaltimer` reads; measure poll-period resolution |
| 3 | `nvshmemi_quiet` / `nvshmemi_flush` at `nvshmemi_common_device.cuh:1185-1275`; `perftest/device/pt-to-pt/shmem_flush_bench.cu`, `shmem_nbi_issue_rate.cu` | Fence latency versus outstanding stores; calibrate as an occupancy sensor |
| 3, 7 | `src/include/non_abi/device/pt-to-pt/proxy_device.cuh:50-130` | Only path with software completion counters; compare with the counter-less NVLink path |
| 4 | `perftest/device/pt-to-pt/shmem_get_bw.cu`, `shmem_g_bw.cu`, `shmem_atomic_bw.cu`; `nvshmemi_signal_op` | Outstanding-request limits; atomic incast on one signal address with many senders |
| 5 | `src/include/non_abi/device/coll/reduce.cuh:148-223`, `fcollect.cuh` NVLS paths; `NVSHMEM_DISABLE_NVLS` | Fan-out skew and reduction latency versus team size |
| 6 | `peer_heap_base_p2p` layout (Demystifying Fig. 1); NVML per-link counters | Address sweep to recover address-to-link mapping; co-run a GEMM to measure interference |
| 7 | `src/host/mem/mem_heap.cpp:133-139`; `NVSHMEM_DISABLE_MNNVL`; memory `beta-imex-channel-broken` | Same put over NVLink versus IB fall-back; document how invisible the switch is to the application |
| 2, 11 | `src/include/non_abi/device/coll/fcollect.cuh:141-284` (LL128 pack/recv) | Use LL flags as fine-grained arrival ground truth |

Beta-specific constraints to remember: MNNVL is off on the tested nodes, so multi-node NVLink experiments wait on the IMEX fix; CUPTI hung on inter-node LLC metrics in FabricPerf's own runs, so expect the same on NVL72; the login node has no GPU, so all of this runs inside the container on a compute node.

---

## 5. One-page comparison

| Feature of memory-semantic scale-up networking | FabricPerf | Why it is general (NVSHMEM evidence) |
|---|---|---|
| Multiple transfer mechanisms per call (st, TMA, multicast, CE, atomics, LL) | one (fence-delimited st) | dispatch in `nvshmemii_put_nbi`; LL128 in collectives; NVLS in reduce/fcollect |
| One-sided arrival with no target-side code | receiver poll + GPTP, discard fast samples | `put` with no `wait`; `get`; `quiet` is a fence only |
| Clock sync error under path asymmetry | assumed symmetric | switched fabric with per-direction lanes; unmeasured |
| Persistent / fused kernels without kernel boundaries | resync per kernel | DeepEP, megakernels, inlined device API |
| Fence as drain + ordering (perturbs, and carries queue information) | treated as an instant | fence is 1/3 of latency; `flush_bench`, `nbi_issue_rate` |
| Hardware flow control invisible to software | protocol-level CLEAR credit | no receiver credit on one-sided put |
| Read / atomic traffic (request-response, issuer-bound, address contention) | not measured | `get` 141 vs `put` 313 GB/s; `g` 9 vs `p` 172 GB/s; signals are remote atomics |
| Multicast and in-switch reduction | not measured; memory model assumes 1:1 | `multimem.st`, `multimem.ld_reduce` |
| Address-hashed contention, compute/comm interference | LLC study only; imbalance unattributed | fixed VA window per peer; shared L2/XBAR |
| Silent path fall-backs, multi-hop topology, link replay | flat domain, loss ignored | MNNVL off on Beta → IB; NVL72 two-level; NVML replay counters |
| Zero-copy delivery (no receiver buffer, no per-packet receiver event) | model requires network buffer | `put` writes into the destination symmetric buffer |
| Whose timestamp is the packet's (SIMT aggregation) | one thread per block | warp waves, fence scopes, async proxy |

---

*Analysis from reading both papers in full and tracing the cited NVSHMEM paths on 2026-09-29. Performance numbers are quoted from the papers, not measured here. Companion notes: `data-movement-paths.md`, `dev-guide-ring-fabric-handle-error.md`.*
