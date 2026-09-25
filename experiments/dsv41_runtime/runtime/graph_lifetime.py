"""One owner resets a graph before releasing any callback or cache storage."""
class GraphLease:
    def __init__(self, graph, cache, stages, retained):
        self.graph=graph;self.cache=cache;self.stages=tuple(stages)
        self.retained=retained;self.closed=False
        self.registered=[]

    def acquire(self):
        self.cache.lease_graph(self.graph);self.registered.append(self.cache.graph_leases)
        for stage in self.stages:
            stage.lease_graph(self.graph);self.registered.append(stage.graphs)

    def close(self):
        if self.closed:return
        # Fence/reset failures retain every lease. The bounded worker must stop.
        streams={int(x.stream.cuda_stream):x.stream for x in (self.cache,*self.stages)}
        for stream in streams.values():stream.synchronize()
        self.graph.reset()
        for stream in streams.values():stream.synchronize()
        key=id(self.graph)
        for mapping in self.registered:assert mapping.get(key) is self.graph
        for mapping in self.registered:del mapping[key]
        self.registered.clear()
        self.retained=None;self.closed=True
