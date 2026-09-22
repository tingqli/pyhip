
class MemoryBlock:
    def __init__(self, start_addr, size, is_free=True):
        self.start_addr = start_addr
        self.size = size
        self.is_free = is_free
        self.next = None

class SimpleMemoryAllocator:
    def __init__(self, total_size=1024):
        self.total_size = total_size
        self.used_size = 0
        self.head = MemoryBlock(0, total_size, True)
        self.addr_bound = 0

    def upper_bound(self):
        return self.addr_bound

    def malloc(self, size, align=1):
        assert size > 0
        assert size <= self.total_size - self.used_size

        current = self.head
        prev = None

        while current:
            if current.is_free and current.size >= size:
                if current.size > size:
                    # cut extra free space into a separate block
                    new_block = MemoryBlock(
                        current.start_addr + size,
                        current.size - size,
                        True
                    )
                    new_block.next = current.next
                    current.next = new_block
                    current.size = size
                
                current.is_free = False
                self.used_size += current.size
                self.addr_bound = max(self.addr_bound, current.start_addr + size)
                assert (current.start_addr % align) == 0
                return current.start_addr
            prev = current
            current = current.next
        assert 0, f'Failed to find big enough block to hold {size} bytes (total free {self.total_size-self.used_size} bytes)'
    
    def free(self, start_addr):
        current = self.head
        while current:
            if current.start_addr == start_addr:
                assert not current.is_free
                current.is_free = True
                self.used_size -= current.size
                self._coalesce_fragments()
                return
            current = current.next
        assert 0, f"Failed to free, no block at addr {start_addr}"

    def _coalesce_fragments(self):
        current = self.head
        while current and current.next:
            if current.is_free and current.next.is_free:
                current.size += current.next.size
                current.next = current.next.next
                continue
            current = current.next


def test_basic():
    import pytest

    mem = SimpleMemoryAllocator()
    addr = mem.malloc(mem.total_size)
    assert addr == 0
    with pytest.raises(AssertionError) as exc_info:
        addr = mem.malloc(1)
    mem.free(addr)

    addr = mem.malloc(mem.total_size)
    mem.free(addr)

    addr0 = mem.malloc(mem.total_size-1)
    addr1 = mem.malloc(1)
    assert addr0 == 0
    assert addr1 == mem.total_size-1
    mem.free(addr1)
    mem.free(addr0)


    addr0 = mem.malloc(mem.total_size//4)
    addr1 = mem.malloc(mem.total_size//2)
    addr2 = mem.malloc(mem.total_size//4)
    mem.free(addr1)
    addr1b = mem.malloc(mem.total_size//2)
    assert addr1 == addr1b
    mem.free(addr1b)

    addr1b = mem.malloc(mem.total_size//4)
    addr1c = mem.malloc(mem.total_size//4)

    mem.free(addr0)
    addr0b = mem.malloc(mem.total_size//8)
    addr0c = mem.malloc(mem.total_size//8)
    assert addr0 == addr0b
    mem.free(addr0b)
    mem.free(addr0c)

    with pytest.raises(AssertionError) as exc_info:
        addr = mem.free(addr1b+1)

    mem.free(addr1b)
    mem.free(addr1c)
    mem.free(addr2)

    addr = mem.malloc(mem.total_size)
    mem.free(addr)

if __name__ == "__main__":
    test_basic()