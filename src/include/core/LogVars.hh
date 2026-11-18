#ifndef QUADCONTROL_CORE_LOGVARS_HH
#define QUADCONTROL_CORE_LOGVARS_HH

#include <cstddef>
#include <cstdint>

#include "rtcore/LogServer.hh"
#include "rtcore/ModuleManager.hh"

namespace quadruped {

// Tear down a module's LogServer variables. Revalidates the cached
// LogServer pointer first: core-module teardown can destroy the LogServer
// before the modules that logged into it go away (go2robot removes core
// modules before cleanupHardware deactivates the driver), so a raw cached
// pointer would be a use-after-free. After the call the pointer is null
// and the ids are zeroed, whether or not anything could be unregistered.
inline void unregisterLogVars( rtcore::ModuleManager* mgr,
                               rtcore::LogServer*& logserver,
                               uintptr_t* ids, std::size_t count ) {
  if ( !mgr || mgr->findModule( LOGSERVER_NAME, 0 ) != logserver ) {
    logserver = nullptr;
  }
  if ( logserver ) {
    for ( std::size_t i = 0; i < count; ++i ) {
      if ( ids[i] ) logserver->deleteVar( ids[i] );
      ids[i] = 0;
    }
    logserver = nullptr;
  } else {
    for ( std::size_t i = 0; i < count; ++i ) ids[i] = 0;
  }
}

}  // namespace quadruped

#endif  // QUADCONTROL_CORE_LOGVARS_HH
