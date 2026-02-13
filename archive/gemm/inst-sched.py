


'''
虚拟指令调度机

输入模块负责生成指令序列，功能上完成目标任务
生成过程类似于传统计算的描述，但是其实for循环结构被完全展开
这些指令序列之间有数据依赖，而这些指令被根据数据依赖关系被缓冲起来

但是LDS的数据依赖无法用寄存器表达, 需要创建一个lds对象，对该对象
可只能先发起写，然后一次sync之后只能发起多次读，也就是一个具有状态的
对象，写状态下只能继续写或者sync，sync之后只能读

或者使用弱约束，分组，相同组内的指令必须按照生成次序连续发起，而lds指令
包含sync指令自动形成一个组。除此之外的其他指令完全靠寄存器依赖DAG约束

分组有序还有一个好处，可以防止随机调度过远的无关指令，比如同时执行iteration0和iteration9
的指令没有太大价值，稍加约束还是有好处的，就是同组的指令按顺序issue

数据依赖和原始局部次序是两个指导方针

sync()是一个
   v0 = buffer_load()
   v1 = buffer_load()
   v2 = buffer_load()
   v3 = buffer_load()

   lds1 = alloc_lds()

   lds_write(lds1, v0)
   lds_write(lds1, v1)
   lds_write(lds1, v2)
   lds_write(lds1, v3)
   
   sync(lds1);

   v4 = lds_read(lds1, offsets)
   v5 = lds_read(lds1, offsets)
   v6 = lds_read(lds1, offsets)
   v7 = lds_read(lds1, offsets)
   MFMA(v4,v6)
   MFMA(v4,v7)
   MFMA(v5,v6)
   MFMA(v5,v7)

Var是连接一个生产者和多个消费者的边的集合   
   
'''


class GCNVar:
    def __init__(self, var_type, id, inst):
        self.var_type = var_type
        self.id = id
        self.inst = inst
        self.user = list()

    def add_user(self, user):
        self.user.append(user)

    def __repr__(self):
        return f"{self.var_type}{self.id}"

class GCNInst:
    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self.target = "?"
        self.vars_args = list()
        for arg in args:
            if isinstance(arg, GCNVar):
                arg.add_user(self)
                self.vars_args.append(arg)

    def __repr__(self):
        return f"{self.target} = {self.name} {self.args}"

class ExecPipe:
    def __init__(self, name, fifo_size, latency, throughput, issue_gap=0, visible=False):
        self.fifo_size = fifo_size
        # latency in cycles: how many cycles spent on the road
        self.latency = latency
        # throughput in cycles: how many cycles another results is ready
        self.throughput = throughput
        # so final instruction latency ~= latency + [throughput if too often]
        self.issue_gap = issue_gap
        self.fifo = []
        self.visible = visible
        self.last_finish_clock = -1e9
        self.last_issue_clock = -1e9
        self.to_be_issue = None
        self.name = name

    def issue(self, inst):
        self.to_be_issue = inst

    def can_issue(self, clock):
        return len(self.fifo) < self.fifo_size and (clock - self.last_issue_clock) > self.issue_gap

    def clock_step(self, clock):
        # throughput constraint
        if clock - self.last_finish_clock >= self.throughput:
            for f in self.fifo:
                # only need to check first on in the fifo
                used_cycles = clock - f.start_clock + 1
                if used_cycles >= self.latency:
                    # latency constraint
                    finished_inst = self.fifo.pop(0)
                    if finished_inst.target is not None:
                        finished_inst.target.ready = 0 # 
                    self.last_finish_clock = clock
                break
        # after previous instruction done, enqueue new one
        if self.to_be_issue is not None:
            assert len(self.fifo) < self.fifo_size
            # enter fifo means start execution
            self.to_be_issue.start_clock = clock
            self.fifo.append(self.to_be_issue)
            self.to_be_issue = None
            self.last_issue_clock = clock

    def status(self):
        ret = ""
        if self.visible:
            if len(self.fifo):
                ret=f"{self.name}{self.fifo[0].start_clock:7}({len(self.fifo)}/{self.fifo_size})"
            else:
                ret=f"{self.name}{'-'*7}({len(self.fifo)}/{self.fifo_size})"
        return ret

class GCN:
    def __init__(self, exec_pipes, resources):
        self.var_ids = {}
        self.instructions = list()
        self.vars = list()
        self.exec_pipes = exec_pipes

        # tracking usage of each type of resources
        self.res_usage = {}
        self.res_limits = resources
        for vt in self.res_limits:
            self.res_usage[vt] = 0

        self.issue_latency = 4

    def get_ret_var_type(self, inst_name):
        # alloc_xxx 是创建非寄存器资源的特殊指令,不同类型资源有不同个数上限限制
        if inst_name.startswith("alloc_"):
            var_type = inst_name[6:]
        else:
            var_type = "v"
        return var_type

    def inst(self, name, *args):
        i = GCNInst(name, *args)
        var_type = self.get_ret_var_type(name)
        self.instructions.append(i)
        v = GCNVar(var_type, self._get_var_id(var_type), i)
        self.vars.append(v)
        i.target = v
        return v

    def _get_var_id(self, var_type):
        if var_type in self.var_ids:
            self.var_ids[var_type] += 1
        else:
            self.var_ids[var_type] = 0
        return self.var_ids[var_type]

    def __getattr__(self, func_name):
        def inst_attr(*args):
            return self.inst(func_name, *args)
        return inst_attr

    def __repr__(self):
        ret = ""
        for inst in self.instructions:
            ret += "\t" + repr(inst) + "\n"
        return ret

    def prepare(self):
        # -1 means not generated
        #  0 generated, not used
        #  1 used by one user
        #  2 used by two users
        #  ....
        remove_var = []
        for v in self.vars:
            v.ready = -1
            # remove not-used target register
            if len(v.user) == 0:
                v.inst.target = None
                remove_var.append(v)
        for v in remove_var:
            self.vars.remove(v)

        self.last_issue_cycle = 0

    def clock_step(self, clock_cycle):
        if clock_cycle == 0:
            self.prepare()

        inst = ""
        if clock_cycle - self.last_issue_cycle >= 4:
            # check instruction finish
            assert len(self.instructions) > 0
            # instruction will be removed once being issued
            candidates = []
            for inst in self.instructions:
                ready = True
                # skip instructions requires too many resources
                target_vtype = self.get_ret_var_type(inst.name)
                if self.res_usage[target_vtype] >= self.res_limits[target_vtype]:
                    continue
                # skip instruction with arg not ready
                for v in inst.vars_args:
                    if v.ready < 0:
                        ready = False
                        break

                if ready:
                    candidates.append(inst)

            #for i,c in enumerate(candidates):
            #    print(f"[{i}] {c}")
            issued = False
            if len(candidates):
                # select from candidates
                for inst in candidates:
                    p = self.exec_pipes[inst.name]
                    if p.can_issue(clock_cycle):
                        p.issue(inst)
                        issued = True
                        self.instructions.remove(inst)
                        self.last_issue_cycle = clock_cycle
                        for v in inst.vars_args:
                            v.ready += 1 # 变量被使用的次数增加
                            #if v.var_type == "lds": print(f">>>>>>>> {v.ready} {len(v.user)}")
                            if v.ready == len(v.user):
                                # 变量已经被全部使用完毕，可以释放了
                                self.res_usage[v.var_type] -= 1
                        # issue 指令时，该指令的目标变量资源也被分配
                        if inst.target is not None:
                            target_vtype = self.get_ret_var_type(inst.name)
                            self.res_usage[target_vtype] += 1
                            assert(self.res_usage[target_vtype] <= self.res_limits[target_vtype])
                        break
            if not issued:
                inst = f"... ...{inst}"
                inst = f""

        # execution pipeline clock stepping
        pipe_status = ""
        for k, p in self.exec_pipes.items():
            p.clock_step(clock_cycle)
            pipe_status += " | " + p.status()

        res_status = ""
        for tp in self.res_usage:
            res_status += " | " + f"{tp}:{self.res_usage[tp]}/{self.res_limits[tp]}"

        #if inst != "" :
        prefix = "    " if inst == "" else ""
        txt = f"{prefix}[{clock_cycle}] {inst} "
        txt += " " * max(50-len(txt), 0)
        print(f"{txt} {pipe_status} {res_status}")


'''
program 执行之后就构造了全展开的一个指令序列存放在gcn对象里面

从profiler结果来看，buffer_load会有500~1000个cycle左右的延迟，而MFMA的计算需要的时间为 4*4*4*(16 or 32)=1024 or 2048 个cycle
因此buffer-load需要提前一个轮次发起，从而需要消耗一份A/B tile大小的临时寄存器用来接收异步访问外存的结果。

本轮计算中就会等待数据就位，写入LDS，同时发起下一轮的预取。

而LDS读入大概有100个cycle左右的延迟，到MFMA计算则也需要一个提前量，但是不需要整个一轮那么多的提前量，只需要在末段提前即可

因此得到下面的pipeline:

1.MFMA 计算上一轮末尾加载的部分A0*B0
    并行写入上一轮预取到寄存器中的数据到LDS,并且发起下一轮数据的预取
2.MFMA 读取LDS刚才写入的数据 同时完成计算A1*B1
3.MFMA 基于读到的数据计算，并行读取下一轮stage1需要的数据

从流水来看可以拆分为两个pipeline:
p1.是外存预取数据到寄存器，寄存器写入LDS
p2.读取LDS, 计算MFMA

但是二者有同步点，就是p1写入LDS完毕之后，可以给p2读取用，p2读取完毕之后可以给p1写入用。
也即是p1的输出会传递给p2. p2使用完毕后的buffer再释放还给p1，

p1和p2内部是小pipe，加上同步点组成协同pipe，交织起来合成指令流

PF,PF,PF, .... .... LDW,LDW,LDW | PF,PF,PF, .... .... LDW,LDW,LDW,

单独考虑LD读取，MFMA计算的pipe:

LDR,LDR,LDR .... MFMA, MFMA, MFMA | LDR,LDR,LDR .... MFMA, MFMA, MFMA

由于LDR的总数只有 16, 远远小于MFMA 个数 64

如果假定LDR读的延迟是~100cycle的话，那么对于32-cycle的MFMA_32x32x8指令，只要提前4个MFMA指令发起第一个数据的读取，到第一个MFMA开始的时候，4个a/b数据已经就位就不会block,
而MFMA开始之后，如果按照每个MFMA配比一个LDR的方式，只需要占用MFMA的12个指令的延迟时间就能发起剩余的其他LDR读，加上再来4个MFMA指令延迟保证数据都读完就位。

这个pipeline里面需要调节的只有提前多少开始下一轮的读入，因为毕竟LDR延迟是估计值。
还需要调节的是，LDR读入和MFMA指令的次序，以及寄存器分配


整个pipe构造的关键指导思路是：
 1. 为了不会block计算，数据加载按照估计的加载延迟提前到上一个轮次，甚至上上个轮次完成
 2. 为了不会block issue，数据加载指令尽可能均匀的分布在可以使用的MFMA指令缝隙中
    尤其是外存访问指令

剩余的MFMA的issue空隙可以用来插入PF和LDW，方式是LDW之后PF

        0x0200 DS write     ds_write_b128  x 8
        0x0100 DS read      ds_read2_b64 x 16
        0x0008 MFMA         v_mfma_f32_32x32x8_f16 x 64
        0x0020 VMEM read    buffer_load_dwordx4 x 8

'''

common_exec_pipe = ExecPipe("comm", fifo_size=50, latency=1, throughput=1,visible=False)
lds_exec_pipe = ExecPipe("lds" ,fifo_size=5, latency=90, throughput=1, issue_gap=32, visible=True)
gcn = GCN(exec_pipes={
    "sync":common_exec_pipe,
    "alloc_lds":common_exec_pipe,
    "zero":common_exec_pipe,
    "buffer_load": ExecPipe("vmem", fifo_size=50000, latency=32*64, throughput=1, issue_gap=32*4, visible=True),
    "lds_write" : lds_exec_pipe,
    "lds_read" : lds_exec_pipe,
    "mfma" : ExecPipe("mfma", fifo_size=1, latency=32, throughput=32,visible=True),
}, resources={
    "v":8+16+16,
    "lds":2
})

def Program(gcn):
    c = []
    for m in range(16):
        c.append(gcn.zero())

    for out_k in range(2):
        lds_deps = []
        lds = gcn.alloc_lds()
        for m in range(8):
            v = gcn.buffer_load(out_k, m)
            lds_deps.append(gcn.lds_write(out_k, lds, v, m))

        lds_sync = gcn.sync(lds, *lds_deps)

        a = []
        b = []
        for i in range(8):
            # 8 elements, each have A/B 2 blocks, 
            a.append(gcn.lds_read(out_k, 2*i, lds_sync,lds))
            b.append(gcn.lds_read(out_k, 2*i+1, lds_sync,lds))

        for k in range(4):
            for m in range(4):
                for n in range(4):
                    cid = m*4 + n
                    gcn.mfma(out_k, (k, m, n), a[(k*4+m)//2], b[(k*4+n)//2], c[cid])

def Program_1(gcn):
    c = []
    for m in range(16):
        c.append(gcn.zero())

    for out_k in range(2):
        a = []
        b = []
        for i in range(8):
            # 8 elements, each have A/B 2 blocks, 
            a.append(gcn.lds_read(out_k, 2*i))
            b.append(gcn.lds_read(out_k, 2*i+1))
        for k in range(4):
            for m in range(4):
                for n in range(4):
                    cid = m*4 + n
                    gcn.mfma(out_k, (k, m, n), a[(k*4+m)//2], b[(k*4+n)//2], c[cid])


Program_1(gcn)
#Program(gcn)
print(gcn)

'''
虚拟执行这个程序，每个指令有一个估计耗时
假设issue 4个cycle开销，每个issue时间窗口选择数据依赖满足的指令进行issue
虚拟执行管线，内置FIFO, FIFO满的状态下issue会引起store，但是不满的状态下可以连续issue
指令执行完毕之后，FIFO就会出现空缺可以继续issue。

执行管线在issue的同时逐个cycle运行管理FIFO中指令的执行进度。
指令进入执行管线是按照指令类型分类的。

哪怕满足数据依赖，执行过远的代码会导致资源不足，例如我们可以限制同时活跃的LDS数量不能超过某个上限
同时使用的寄存器不能超过某个上限来限制过远的，过于无关的代码被提前调度执行。

但是如何track资源数量呢, 需要一个活跃的资源分配跟踪机制，尤其是未来代码不再需要某个资源时能够立刻发现

资源的创建指令返回寄存器或者lds引用，这时资源被分配，如果此时发现资源超出上限，则应该等待，避免issue此指令
从而依赖此资源的后继指令也不会被issue。

而最后一个引用某个资源的指令被执行完毕，这个资源就可以被释放，问题是，如何知道后继指令不再引用某个资源呢？
这需要搜索未来全部指令才能知道（reservation station不用这样因为某个命名的寄存器被重新赋值时，就保证了未来代码
不会再引用旧的寄存器数据内容）；因此我们需要一个实现搜索全部指令，确定最后使用某个资源的指令并标记，执行期该指令
就会附带完成释放资源的功能。

或者从另一个角度而言，这个步骤其实已经完成了物理资源到虚拟资源的映射，但是是在（模拟）运行期动态完成而非编译期静态完成，
但是我们希望从模拟运行中总结的一些东西指导编译期完成类似的调度


'''


gcn.prepare()
for clock in range(10000):
    gcn.clock_step(clock)
