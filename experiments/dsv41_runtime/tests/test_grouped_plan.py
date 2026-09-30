import unittest
from grouped_plan import routing_groups, packed_routing


def record(gate, up, down):
    return {'tensors': {p + '.trellis': {'shape': [320, 144, k*16]}
                       for p, k in zip(('w1', 'w3', 'w2'), (gate, up, down))}}


class GroupedPlanTests(unittest.TestCase):
    def test_mixed_projection_bits_and_ownership(self):
        records = {'0:1': record(2, 3, 2), '0:2': record(4, 6, 5), '0:3': record(2, 3, 2)}
        groups = routing_groups(0, [[1, 2, 3, 1, -1, 384]], records)
        self.assertEqual([[r.key for r in g] for g in groups], [['0:1', '0:3'], ['0:2']])
        self.assertEqual(groups[0][0].bits, (2, 3, 2))
        self.assertEqual(packed_routing(groups[0], tokens=1, topk=6),
                         ([2, 1, 3], [0]*6, [0, 3, 2, 0, 0, 0]))

    def test_oversized_duplicate_routes_are_not_dropped(self):
        groups = routing_groups(0, [[1, 1, 2]]*3000,
                                {'0:1': record(8, 7, 6), '0:2': record(8, 7, 6)})
        routes = [p for group in groups for expert in group for p in expert.positions]
        self.assertEqual(sorted(routes), list(range(9000)))
        self.assertTrue(all(len(e.positions) <= 2048 for g in groups for e in g))
        for group in groups:
            counts, tokens, offsets = packed_routing(group, tokens=3000, topk=3)
            self.assertEqual(sum(counts), 9000)
            self.assertEqual(len(tokens), len(offsets))

    def test_unowned_and_empty_routes(self):
        self.assertEqual(routing_groups(0, [[-1, 384, 7]], {'1:7': record(2, 2, 2)}), [])
        self.assertEqual(routing_groups(0, [], {}), [])
        with self.assertRaises(ValueError):
            routing_groups(0, [[1], [1, 2]], {})


if __name__ == '__main__':
    unittest.main()
