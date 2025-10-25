

import numpy as np

# about LDS & swizzle
# https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/pipeline-descriptions.html#local-data-share-lds
# https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html

# ds_write_b128 for writing data into LDS:
#   - according to buffer_load_dwordx4's requiements,
#     the data is loaded using dwordx4 & as contingous as possible in cols
#   - bank-conflict happens in lane_group_w
#
# ds_read_b128 for reading data from LDS:
#   - according to MFMA's input layout requiements
#   - bank-conflict happens in lane_group_r
#
# when a 2D-tile of shape=(M,K) dtype=b128 in LDS is to be accessed, both instructions calculate LDS offsets using following logic:
#
'''
   logical_row, logical_col = get_logical_coord(lane)
   physical_row, physical_col = swizzle(logical_row, logical_col)
   LDS_offsets = (physical_row * tile_K + physical_col) * sizeof(b128)
'''
#
#  get_logical_coord() was determined by buffer_load_dwordx4(ds_write_b128) / mfma(ds_read_b128)
#  swizzle() shuffles the elements within the tiles by mapping logical coordinates to different physical coordinates.
#  so no bank-conflict may happen for both read turns & write turns
#
# this script just simulates all ds_read_b128 & ds_write_b128 instructions's behaviour and report conflict
#

WARP_SIZE = 64


# a write/read is invoked with LDS offsets (in bytes)
# check bank-conflict according to offsets / lane-groups / bank settings
def check_bank_conflict(lane_groups, offsets):
    conflicts = 0
    for group in lane_groups:
        banks = []
        for lane in group:
            b = bank_of(offsets[lane])
            conf = b in banks
            conflicts += 1 if conf else 0
            # print(f"{'conflict' if conf else '        '} ({logical_row},{logical_col}) => ({physical_row},{physical_col}) {lane} bank{b}")
            banks.append(b)
    return conflicts


class LDSTile2D:
    def __init__(self, M, K, itemsize):
        self.M = M
        self.K = K
        self.itemsize = itemsize
        self.conflicts = 0
        self.NUM_BANKS = 32
        self.BANK_WIDTH = np.dtype(np.int32).itemsize
        print(f"===== LDSTile2D(rows={self.M}, cols={self.K},  itemsize={self.itemsize} bytes) =====")

    def reset(self):
        self.access_map[:] = 0

    def swizzle(self, row, col):
        #return row, (col^(row//2))&(self.K-1)
        return row, col

    def show_swizzle(self):
        reverse_swizzle = {}
        for row in range(self.M):
            for col in range(self.K):
                phy_row, phy_col = self.swizzle(row, col)
                reverse_swizzle[(phy_row, phy_col)] = (row, col)

        print("=" * 80)
        print("Physical layout, with logical (row, col) at each position")
        for phy_row in range(self.M):
            print(f"{phy_row}:\t", end="")
            for phy_col in range(self.K):
                r,c = reverse_swizzle[(phy_row, phy_col)]
                if r == phy_row:
                    print(f"(__,{c:2})", end=",")
                else:
                    print(f"({r:2},{c:2})", end=",")
            print("     | banks: ", end="")
            for phy_col in range(self.K):
                bank = self.bank_of(phy_row, phy_col)
                print(f"{bank:2}", end=",")

            print()
        print(f"total conflicts = {self.conflicts}")

    # an item will occupy multiple banks when itemsize > BANK_WIDTH
    # but that's not a problem since the ID of first bank can represent whole bank-group
    # because both ds_read_b128/ds_write_b128 will access with same itemsize thus same bank-group
    def bank_of(self, physical_row, physical_col):
        bank = ((physical_row * self.K + physical_col)*self.itemsize//self.BANK_WIDTH) % self.NUM_BANKS
        return bank

    def access(self, lane_groups, lane_rows, lane_cols):
        # access will show the data being accessed
        info_map = {}
        access_map = np.zeros((3, self.M, self.K), dtype=np.uint32)
        lane_group_map = np.zeros((self.M, self.K), dtype=np.uint32)
        bank_map = np.zeros((self.M, self.K), dtype=np.uint32)
        lane_map = np.zeros((self.M, self.K), dtype=np.uint32)
        for group_id, group in enumerate(lane_groups):
            banks = []
            for lane in group:
                logical_row, logical_col = lane_rows[lane], lane_cols[lane]

                assert logical_row >=0 and logical_row < self.M, f"logical_row {logical_row} not in [0, {self.M})"
                assert logical_col >=0 and logical_col < self.K, f"logical_col {logical_col} not in [0, {self.K})"
                physical_row, physical_col = self.swizzle(logical_row, logical_col)
                assert physical_row >=0 and physical_row < self.M
                assert physical_col >=0 and physical_col < self.K

                b = self.bank_of(physical_row, physical_col)
                conf = b in banks
                self.conflicts += 1 if conf else 0
                #if conf: print(f"{'conflict' if conf else '        '} ({logical_row},{logical_col}) => ({physical_row},{physical_col}) lane{lane} bank{b}")
                banks.append(b)

                flag = "  " if conf == 0 else "\033[0;33m=>\033[0m"
                color0 = f"\033[0;{100+(group_id % 8)}m"
                color1 = f"\033[0m"
                info = f"{flag}{color0} L{lane:02} B{b:02} {color1}"
                info_map[(0, logical_row, logical_col)] = info
                info_map[(1, physical_row, physical_col)] = info

        for row in range(self.M):
            found = False
            for col in range(self.K):
                if (0, row, col) in info_map or (1, row, col) in info_map:
                    found = True
                    break
            if not found: continue

            print(f"{row:2}:\t", end="")
            # logical
            for col in range(self.K):
                info = info_map[(0,row,col)] if (0,row,col) in info_map else " "*9
                print(info, end=" ")
            print("\t|\t", end="")
            # physical
            for col in range(self.K):
                info = info_map[(1,row,col)] if (1,row,col) in info_map else " "*9
                print(info, end=" ")
            print()

    def ds_write_b128(self, lane_rows, lane_cols):
        if not hasattr(self, "lane_groups_wb128"):
            lane_groups=[]
            for lane in range(0, WARP_SIZE, 8):
                lane_groups.append([lane + i for i in range(8)])
            self.lane_groups_wb128 = lane_groups

        conflicts0 = self.conflicts
        self.access(self.lane_groups_wb128, lane_rows, lane_cols)
        print(f"ds_write_b128(). total conflicts += {self.conflicts-conflicts0}")

    def ds_read_b128(self, lane_rows, lane_cols):
        if not hasattr(self, "lane_groups_rb128"):
            def lg(lane0, lane1):
                return [lane0, lane0+1, lane0+2, lane0+3] + [lane1, lane1+1, lane1+2, lane1+3]
            lane_groups=[
                lg(0,20),
                lg(4,16),
                lg(8,28),
                lg(12,24),
                lg(32+0,32+20),
                lg(32+4,32+16),
                lg(32+8,32+28),
                lg(32+12,32+24),
            ]
            self.lane_groups_rb128 = lane_groups

        conflicts0 = self.conflicts
        self.access(self.lane_groups_rb128, lane_rows, lane_cols)
        print(f"ds_read_b128(). total conflicts += {self.conflicts-conflicts0}")


lds = LDSTile2D(32, 4, 16)

def write_lane_coord(row0=0, col0=0):
    row = []
    col = []
    for lane in range(0, WARP_SIZE):
        row.append((lane // 4) + row0)
        col.append((lane % 4) + col0)
    return row, col

print("================= ds_write_b128 ")
# buffer_load_dwordx4
lane = np.arange(0, WARP_SIZE)
lane_write_row = lane // 4
lane_write_col = lane % 4

lds.ds_write_b128(lane_write_row, lane_write_col)
lds.ds_write_b128(lane_write_row + 16, lane_write_col)

print("================= ds_read_b128 ")
# VMFMA 32x32x8
lane_read_row = lane % 32
lane_read_col = lane // 32
lds.ds_read_b128(lane_read_row, lane_read_col)
lds.ds_read_b128(lane_read_row, lane_read_col + 2)



lds.show_swizzle()