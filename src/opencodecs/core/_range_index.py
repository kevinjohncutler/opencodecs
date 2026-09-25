"""An index of cached byte ranges, independent of storage or eviction policy.

Group intervals by their length rounded up to a power of two. In a group
of width W, an interval covering offset O can start only in O's bucket
or the preceding bucket. Disjoint tiny ranges therefore do not require a
scan of every cached entry. Overlapping intervals in a bucket still need
an exact containment check; this is not a worst-case interval tree.

The owner supplies locking and removes entries when it evicts payloads.
Only keys are indexed, so payload ownership and byte accounting stay with
the cache. No range is discarded merely to accelerate lookup.
"""


class RangeIndex:
    def __init__(self):
        # Most sparse buckets contain one key. Avoid a dictionary per key.
        self._groups: dict[int, dict] = {}

    def add(self, key: tuple[int, int]) -> None:
        offset, length = key
        if length <= 0:
            return
        shift = (length - 1).bit_length()
        buckets = self._groups.setdefault(shift, {})
        bucket_id = offset >> shift
        bucket = buckets.get(bucket_id)
        if bucket is None:
            buckets[bucket_id] = key
        elif isinstance(bucket, tuple):
            if bucket != key:
                buckets[bucket_id] = {bucket: None, key: None}
        else:
            bucket[key] = None

    def discard(self, key: tuple[int, int]) -> None:
        offset, length = key
        if length <= 0:
            return
        shift = (length - 1).bit_length()
        buckets = self._groups.get(shift)
        if buckets is None:
            return
        bucket_id = offset >> shift
        bucket = buckets.get(bucket_id)
        if bucket is None:
            return
        if isinstance(bucket, tuple):
            if bucket != key:
                return
            del buckets[bucket_id]
        else:
            bucket.pop(key, None)
            if len(bucket) == 1:
                buckets[bucket_id] = next(iter(bucket))
        if not buckets:
            del self._groups[shift]

    def covering(self, offset: int, length: int) -> tuple[int, int] | None:
        end = offset + length
        for shift, buckets in self._groups.items():
            if length > 1 << shift:
                continue
            bucket_id = offset >> shift
            for candidate in (bucket_id, bucket_id - 1):
                bucket = buckets.get(candidate)
                if bucket:
                    candidates = (bucket,) if isinstance(bucket, tuple) else reversed(bucket)
                    for key in candidates:
                        start, size = key
                        if start <= offset and end <= start + size:
                            return key
        return None

    def clear(self) -> None:
        self._groups.clear()
