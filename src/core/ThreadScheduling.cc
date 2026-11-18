#include "core/ThreadScheduling.hh"

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace quadruped {

bool unpinCurrentThreadToAllCpus() {
#ifdef __linux__
  cpu_set_t all_cpus;
  CPU_ZERO( &all_cpus );
  const int ncpu = static_cast<int>( sysconf( _SC_NPROCESSORS_ONLN ) );
  for ( int c = 0; c < ncpu; ++c ) CPU_SET( c, &all_cpus );
  return pthread_setaffinity_np( pthread_self(), sizeof( all_cpus ),
                                 &all_cpus ) == 0;
#else
  return true;
#endif
}

bool setCurrentThreadNormalScheduler() {
#ifdef __linux__
  sched_param sp{};
  return pthread_setschedparam( pthread_self(), SCHED_OTHER, &sp ) == 0;
#else
  return true;
#endif
}

}  // namespace quadruped
