#ifndef QUADCONTROL_CORE_THREADSCHEDULING_HH
#define QUADCONTROL_CORE_THREADSCHEDULING_HH

namespace quadruped {

// Thread-scheduler plumbing for helper threads. Threads created by the
// control thread inherit its SCHED_FIFO policy and its control-cpuset
// affinity (config/versions/go2/threads.toml): a priority-98 helper on the
// control CPUs can stall the 500 Hz loop (observed: a CSV logger cost
// 29.6 ms and latched the loop-stall estop) or starve the very watchdogs
// meant to protect it. Every long-lived helper thread demotes itself with
// these before doing any work.

// Un-pin the calling thread from any inherited CPU mask (all online CPUs).
// Returns false when the OS refused; always true on non-Linux.
bool unpinCurrentThreadToAllCpus();

// Put the calling thread on the normal, preemptible scheduler (SCHED_OTHER).
// Returns false when the OS refused; always true on non-Linux.
bool setCurrentThreadNormalScheduler();

}  // namespace quadruped

#endif  // QUADCONTROL_CORE_THREADSCHEDULING_HH
