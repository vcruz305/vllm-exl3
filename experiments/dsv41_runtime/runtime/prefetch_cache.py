"""Plan verified disk reads from actual routes before grouped GPU dispatch."""
from grouped_cache import GroupedExpertCache
from grouped_plan import cache_first_groups


class PrefetchExpertCache(GroupedExpertCache):
    def _prepare_dispatch(self,groups):
        # Routing is already known; no expert prediction, pruning, substitution
        # or extra device-to-host routing capture is introduced.
        groups[:]=cache_first_groups(groups,self.entries)
        self.store.plan(item.key for group in groups for item in group
                        if item.key not in self.entries)

    def apply(self,*args,**kwargs):
        try:
            return super().apply(*args,**kwargs)
        finally:
            self.stats.update({'prefetch_'+key:value for key,value in self.store.async_stats.items()})
            self.stats['prefetch_staging_bytes']=self.store.staging_bytes
            self.stats['prefetch_peak_staging_bytes']=self.store.peak_staging_bytes

    def close(self):
        try:
            self.store.plan([])
        finally:
            super().close()
