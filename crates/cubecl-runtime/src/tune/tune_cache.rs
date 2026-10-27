#[cfg(persistence)]
use alloc::vec::Vec;

#[cfg(persistence)]
use cubecl_environment::persistence::StoreError;
#[cfg(persistence)]
use cubecl_environment::persistence::{CacheOption, Namespace, Store, StoreOptions};
#[cfg(serializable)]
use serde::{Deserialize, Serialize};

use super::{AutotuneError, AutotuneKey, AutotuneOutcome};
use alloc::string::String;
use cubecl_environment::collections::HashMap;

#[derive(Debug)]
pub(crate) enum CacheEntry {
    Done {
        checksum: ChecksumState,
        fastest_index: usize,
    },
    Pending,
}

#[derive(Debug)]
#[allow(dead_code)] // Some variants are not created when the cache isn't saved.
pub(crate) enum ChecksumState {
    Match,
    NoMatch,
    ToBeVerified(String),
}

/// Persistent cache key
#[cfg(persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Hash)]
pub struct PersistentCacheKey<K> {
    /// The autotune key identifying the operation.
    pub key: K,
    /// The checksum of the candidate list the key was tuned under: an answer
    /// to another list is an answer to another question.
    pub checksum: String,
}

/// Persistent cache entry
///
/// Only [`fastest_index`](Self::fastest_index) is read back: hydration seeds the in-memory cache
/// from it and nothing else. Everything below it is stored so a cache entry can be inspected after
/// the fact — why a kernel won, against which measurements, and under which bounds — which is the
/// question that cannot be answered from a live process once tuning is over. That is also why the
/// type is `pub`: reading an entry back is the point.
#[cfg(persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct PersistentCacheValue {
    /// Index of the fastest candidate operation.
    pub fastest_index: usize,
    /// Benchmarking results for all autotune candidates.
    pub results: Vec<AutotuneResult>,
    /// Optional input size bounds for which the autotune result applies.
    ///
    /// Defaulted, so entries written before this field existed still decode. Without it every
    /// cached key on every existing installation would fail to read and re-tune from scratch.
    #[serde(default)]
    pub bounds: Option<crate::tune::Bounds>,
    /// Optional execution time limit for the autotune process.
    ///
    /// Defaulted for the same reason as [`bounds`](Self::bounds).
    #[serde(default)]
    pub limit: Option<core::time::Duration>,
}

#[cfg_attr(serializable, derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// The result of an autotune job.
pub struct AutotuneResult {
    /// The outcome of the benchmark.
    pub outcome: Result<AutotuneOutcome, AutotuneError>,
}

impl AutotuneResult {
    pub(crate) fn error(error: AutotuneError) -> Self {
        Self {
            outcome: Err(error),
        }
    }
    pub(crate) fn success(outcome: AutotuneOutcome) -> Self {
        Self {
            outcome: Ok(outcome),
        }
    }
}

impl Eq for AutotuneResult {}
impl PartialEq for AutotuneResult {
    fn eq(&self, other: &Self) -> bool {
        match (&self.outcome, &other.outcome) {
            (Ok(lhs), Ok(rhs)) => lhs == rhs,
            (Ok(_), Err(_)) => false,
            (Err(_), Ok(_)) => false,
            // We don't have to check the error
            (Err(_), Err(_)) => true,
        }
    }
}

/// The kernels picked in this process, keyed by [`pick_key`]: a caller exports
/// them once tuned ([`autotune_picks`]) and seeds a later run of the same build with them
/// ([`seed_autotune_picks`]). An index names a kernel of that build's tunable sets, so picks carry over
/// only between runs of one build.
static PICKS: cubecl_environment::sync::Mutex<Option<HashMap<String, usize>>> =
    cubecl_environment::sync::Mutex::new(None);

fn pick_key(cache: &str, key: &impl core::fmt::Display) -> String {
    alloc::format!("{cache}|{key}")
}

/// Every kernel measured so far by the tuners of this process, and every seeded pick, as
/// `(<device>/<tuner>|<key>, index)`.
pub fn autotune_picks() -> alloc::vec::Vec<(String, usize)> {
    (PICKS.lock().iter().flatten())
        .map(|(key, &index)| (key.clone(), index))
        .collect()
}

/// Picks a tuner takes instead of measuring, for the keys [`autotune_picks`] exported from a run of the
/// same build.
pub fn seed_autotune_picks(picks: impl IntoIterator<Item = (String, usize)>) {
    PICKS.lock().get_or_insert_with(HashMap::new).extend(picks);
}

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub(crate) struct TuneCache<K> {
    /// The single in-memory home of tuning state, keyed for the per-launch
    /// lookup: tuned picks, in-flight tunes and checksum verdicts. Hydrated
    /// from the store, which retains nothing itself, and rebuilt when the
    /// environment switches.
    in_memory_cache: HashMap<K, CacheEntry>,
    /// Write-through persistence, or `None` when the persistent cache is
    /// disabled, so no cache file is ever touched. Lazy: entries live in
    /// [`Self::in_memory_cache`] once hydrated, not here.
    #[cfg(persistence)]
    persistent_cache: Option<Store<PersistentCacheKey<K>, PersistentCacheValue>>,
    /// The namespace the table is stored under, whether or not it is
    /// persisted: what a [record](super::TuneRecord) names it by.
    #[cfg(persistence)]
    table: String,
    /// Whether everything the store holds has been ingested into
    /// [`Self::in_memory_cache`]. What makes an ordinary miss cost a bool
    /// check rather than a walk; `false` until the first sync, and again
    /// after an environment switch.
    #[cfg(persistence)]
    hydrated: bool,
    /// The environment generation [`Self::in_memory_cache`] was built under;
    /// see [`cubecl_environment::environment::generation`].
    #[cfg(persistence)]
    generation: u32,
    /// The device and tuner the picks of this cache are exported under.
    id: String,
}

/// Result of the cache try
#[derive(Debug)]
pub enum TuneCacheResult {
    /// An operation is found.
    Hit {
        /// The index of the fastest operation to execute.
        fastest_index: usize,
    },
    /// The operation might be cached, but we don't know yet whether the checksum is valid.
    Unchecked,
    /// A tuning job is in flight for this key — the worker hasn't published a result yet.
    /// Callers that see this fall through to running the operation rather than blocking on
    /// the in-flight job.
    Pending,
    /// No operation is found yet.
    Miss,
}

impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn new(name: &str, device_id: &str) -> Self {
        #[cfg(persistence)]
        {
            use crate::config::RuntimeConfig;
            use alloc::format;

            let config = crate::config::CubeClRuntimeConfig::get();
            let namespace = Namespace::scoped("autotune", format!("{device_id}/{name}"));
            let table = namespace.as_str().into();

            if config.autotune.disable_cache {
                return TuneCache {
                    in_memory_cache: HashMap::new(),
                    id: format!("{device_id}/{name}"),
                    persistent_cache: None,
                    table,
                    hydrated: true,
                    generation: cubecl_environment::environment::generation(),
                };
            }

            // Sampled before the store opens, so a switch landing in between
            // reads as "rebuild", never as "this state belongs to the new
            // environment".
            let generation = cubecl_environment::environment::generation();
            let mut cache = TuneCache {
                in_memory_cache: HashMap::new(),
                id: format!("{device_id}/{name}"),
                persistent_cache: Some(Store::new(
                    StoreOptions::new()
                        .storage(namespace)
                        .cache(CacheOption::Lazy),
                )),
                table,
                hydrated: false,
                generation,
            };
            log::info!("Load autotune cache ...");
            let loaded = cache.sync_persistent();
            log::info!("Loaded {loaded} autotune cached entries");

            cache
        }

        #[cfg(not(persistence))]
        {
            TuneCache {
                in_memory_cache: HashMap::new(),
                id: alloc::format!("{device_id}/{name}"),
            }
        }
    }

    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        let Some(val) = self.in_memory_cache.get(key) else {
            return TuneCacheResult::Miss;
        };

        let CacheEntry::Done {
            checksum,
            fastest_index,
        } = val
        else {
            // Pending: clone the receiver so the caller can subscribe to the in-flight tune.
            let CacheEntry::Pending = val else {
                unreachable!()
            };
            return TuneCacheResult::Pending;
        };

        if cfg!(persistence) {
            match checksum {
                ChecksumState::ToBeVerified(..) => TuneCacheResult::Unchecked, // Don't know yet.
                ChecksumState::NoMatch => TuneCacheResult::Miss,               // Can't use this.
                ChecksumState::Match => TuneCacheResult::Hit {
                    fastest_index: *fastest_index,
                },
            }
        } else {
            // Clippy;
            let _ = checksum;
            TuneCacheResult::Hit {
                fastest_index: *fastest_index,
            }
        }
    }

    #[cfg(persistence)]
    pub fn validate_checksum(&mut self, key: &K, checksum: &str) -> TuneCacheResult {
        let Some(val) = self.in_memory_cache.get_mut(key) else {
            return TuneCacheResult::Miss;
        };

        if let CacheEntry::Done {
            checksum: checksum_state,
            ..
        } = val
            && let ChecksumState::ToBeVerified(checksum_expected) = checksum_state
        {
            if checksum_expected == checksum {
                *checksum_state = ChecksumState::Match;
            } else {
                *checksum_state = ChecksumState::NoMatch;
            }
        }

        self.fastest(key)
    }

    /// The index settled for `key`, reading only: `None` for anything but a verified result, and
    /// for every key once the environment has switched away from the one this cache was built
    /// under — the reset that switch owes is left to the next lookup that tunes.
    pub(crate) fn settled(&self, key: &K) -> Option<usize> {
        #[cfg(persistence)]
        if self.environment_switched() {
            return None;
        }

        match self.fastest(key) {
            TuneCacheResult::Hit { fastest_index } => Some(fastest_index),
            TuneCacheResult::Unchecked | TuneCacheResult::Pending | TuneCacheResult::Miss => None,
        }
    }

    /// Mark a key as being tuned. Used by [`Tuner::check_tune`] under the cache mutex so that
    /// concurrent callers see [`TuneCacheResult::Pending`] instead of starting a second job
    /// for the same key.
    pub(crate) fn mark_pending(&mut self, key: K) {
        self.in_memory_cache.insert(key, CacheEntry::Pending);
    }

    /// The pick seeded for `key`, now in this cache.
    pub(crate) fn seeded(&mut self, key: &K) -> Option<usize> {
        let index = *PICKS.lock().as_ref()?.get(&pick_key(&self.id, key))?;
        self.cache_insert(key.clone(), index);
        Some(index)
    }

    /// Drops a seeded pick that named no kernel of this device, so the key is measured.
    pub(crate) fn forget_seeded(&mut self, key: &K) {
        self.in_memory_cache.remove(key);
        if let Some(picks) = PICKS.lock().as_mut() {
            picks.remove(&pick_key(&self.id, key));
        }
    }

    /// Lists `key`'s measured pick among the picks [`autotune_picks`] exports.
    pub(crate) fn export(&self, key: &K, fastest_index: usize) {
        PICKS
            .lock()
            .get_or_insert_with(HashMap::new)
            .insert(pick_key(&self.id, key), fastest_index);
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize) {
        self.in_memory_cache.insert(
            key,
            CacheEntry::Done {
                checksum: ChecksumState::Match,
                fastest_index,
            },
        );
    }
}

#[cfg(persistence)]
impl<K: AutotuneKey> TuneCache<K> {
    /// Drops tuning state belonging to a previous environment, so a switch
    /// re-hydrates and re-tunes rather than serving the old environment's
    /// picks. One relaxed atomic load when nothing switched.
    ///
    /// In-flight tunes are dropped with everything else: their completion
    /// still records a hardware-valid result, so the whole cost of the race
    /// is one duplicate tune per switch.
    pub(crate) fn reset_if_environment_switched(&mut self) {
        if !self.environment_switched() {
            return;
        }

        log::debug!("Environment switched, resetting the autotune cache");
        self.generation = cubecl_environment::environment::generation();
        self.in_memory_cache.clear();
        self.hydrated = false;
    }

    /// Whether the environment switched since this cache was built. Never with persistence
    /// disabled: the tuning state is then process-local and unbound, like a store without a
    /// storage, and survives switches.
    fn environment_switched(&self) -> bool {
        self.persistent_cache.is_some()
            && cubecl_environment::environment::generation() != self.generation
    }

    /// Ingest everything the persistent store holds into the in-memory cache,
    /// as unverified entries.
    ///
    /// Runs at construction, and again whenever `hydrated` fell back to
    /// `false` after an environment switch. Once hydrated, a miss costs one
    /// bool check here — never a walk, and never a rescan of the database
    /// under the tuner mutex.
    ///
    /// Returns how many entries the store delivered.
    pub(crate) fn sync_persistent(&mut self) -> usize {
        if self.hydrated {
            return 0;
        }

        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return 0;
        };

        let mut delivered = 0;
        persistent_cache.scan(|key, value| {
            delivered += 1;
            self.in_memory_cache
                .entry(key.key)
                .or_insert(CacheEntry::Done {
                    checksum: ChecksumState::ToBeVerified(key.checksum),
                    fastest_index: value.fastest_index,
                });
        });
        self.hydrated = true;

        delivered
    }

    /// The namespace the table is stored under.
    pub(crate) fn table(&self) -> &str {
        &self.table
    }

    /// Store an answer in the table, and say whether it took it: never when
    /// the cache is disabled.
    pub(crate) fn persistent_cache_insert(
        &mut self,
        key: K,
        checksum: String,
        value: PersistentCacheValue,
    ) -> bool {
        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return false;
        };

        let Err(err) = persistent_cache.insert(PersistentCacheKey { key, checksum }, value) else {
            return true;
        };
        match err {
            StoreError::DuplicatedKey {
                key,
                value_previous,
                value_updated,
            } => log::warn!(
                "Autotune the same function multiple times for key {key:?} => old {value_previous:?}, new {value_updated:?}"
            ),
            // Another process sharing the cache root tuned this key first.
            // Routine with N training processes on a cold cache, and both
            // results are valid, so it stays quiet: warning here would
            // print a full result payload per key on every cold start.
            StoreError::KeyOutOfSync { key, .. } => {
                log::debug!("Autotune result for key {key:?} was already stored concurrently")
            }
            StoreError::Backend { key, error } => log::warn!(
                "Autotune result for key {key:?} could not be stored, it will be retuned: {error}"
            ),
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use core::fmt::Display;

    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, Debug)]
    struct Shape(u32);

    impl Display for Shape {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "shape {}", self.0)
        }
    }

    impl AutotuneKey for Shape {}

    #[test]
    fn a_pick_exported_seeds_a_later_cache_of_the_same_device_and_tuner() {
        let mut first = TuneCache::<Shape>::new("matmul", "gpu-0");
        first.export(&Shape(7), 3);
        let picks = autotune_picks();
        assert!(picks.contains(&(String::from("gpu-0/matmul|shape 7"), 3)));

        seed_autotune_picks(picks);
        let mut later = TuneCache::<Shape>::new("matmul", "gpu-0");
        assert_eq!(later.seeded(&Shape(7)), Some(3));
        assert_eq!(later.seeded(&Shape(8)), None);
        let mut other = TuneCache::<Shape>::new("conv", "gpu-0");
        assert_eq!(other.seeded(&Shape(7)), None);
    }
}
