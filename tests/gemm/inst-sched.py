


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

   lds1 = new_lds()

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
    def __init__(self, id, inst):
        self.id = id
        self.inst = inst
        self.user = list()

    def add_user(self, user):
        self.user.append(user)

    def __repr__(self):
        return f"v{self.id}"

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
        self.cycles = 4
        if name == "buffer_load":
            self.cycles = 100
        if name == "lds_read":
            self.cycles = 16
        if name == "lds_write":
            self.cycles = 16
        if name == "mfma":
            self.cycles = 16

    def __repr__(self):
        return f"{self.target} = {self.name} {self.args}"

class ExecPipe:
    def __init__(self, name, fifo_size, latency, throughput, visible=False):
        self.fifo_size = fifo_size
        # latency in cycles: how many cycles spent on the road
        self.latency = latency
        # throughput in cycles: how many cycles another results is ready
        self.throughput = throughput
        # so final instruction latency ~= latency + [throughput if too often]
        self.fifo = []
        self.busy_cycles = 0
        self.visible = visible
        self.last_finish_clock = 0
        self.to_be_issue = None
        self.name = name

    def issue(self, inst):
        self.to_be_issue = inst

    def can_issue(self):
        return len(self.fifo) < self.fifo_size

    def clock_step(self, clock):
        if self.to_be_issue is not None:
            assert len(self.fifo) < self.fifo_size
            # enter fifo means start execution
            self.to_be_issue.start_clock = clock
            self.fifo.append(self.to_be_issue)
            self.to_be_issue = None

        # throughput constraint
        if clock - self.last_finish_clock >= self.throughput:
            for f in self.fifo:
                # only need to check first on in the fifo
                pass_cycles = clock - f.start_clock
                if pass_cycles >= self.latency:
                    # latency constraint
                    finished_inst = self.fifo.pop(0)
                    finished_inst.target.ready = 0 # 
                    self.last_finish_clock = clock
                break

    def status(self):
        ret = ""
        if self.visible:
            if len(self.fifo):
                ret=f"{self.name}{self.fifo[0].start_clock:4}"
            else:
                ret=f"{self.name}----"
        return ret

class GCN:
    def __init__(self, exec_pipes):
        self.var_id = 0
        self.instructions = list()
        self.vars = list()
        self.exec_pipes = exec_pipes
        self.issue_latency = 4

    def inst(self, name, *args):
        i = GCNInst(name, *args)
        self.instructions.append(i)
        v = GCNVar(self._get_var_id(), i)
        self.vars.append(v)
        i.target = v
        return v

    def _get_var_id(self):
        self.var_id += 1
        return self.var_id

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
        for v in self.vars:
            v.ready = -1
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
                    if p.can_issue():
                        p.issue(inst)
                        issued = True
                        self.instructions.remove(inst)
                        self.last_issue_cycle = clock_cycle
                        break
            if not issued:
                inst = f"... ...{inst}"

        # execution pipeline clock stepping
        pipe_status = ""
        for k, p in self.exec_pipes.items():
            p.clock_step(clock_cycle)
            pipe_status += " | " + p.status()

        txt = f"[{clock_cycle}] {inst}"
        txt += " " * max(50-len(txt), 0)
        print(f"{txt} {pipe_status}")
        

'''
program 执行之后就构造了全展开的一个指令序列存放在gcn对象里面
'''
def Program(gcn):
    c = []
    for m in range(4):
        c.append(gcn.zero())

    for k in range(40):
        lds_deps = []
        lds = gcn.new_lds()
        for m in range(2):
            v = gcn.buffer_load(m, k)
            lds_deps.append(gcn.lds_write(lds, v, m))

        lds_sync = gcn.sync(lds, *lds_deps)

        a = []
        b = []
        for i in range(2):
            a.append(gcn.lds_read(lds_sync, i))
            b.append(gcn.lds_read(lds_sync, 2+i))

        for m in range(2):
            for n in range(2):
                cid = m*2 + n
                gcn.mfma(a[m], b[n], c[cid])

common_exec_pipe = ExecPipe("comm", fifo_size=50, latency=1, throughput=1,visible=True)
lds_exec_pipe = ExecPipe("lds" ,fifo_size=5, latency=20, throughput=1,visible=True)
gcn = GCN(exec_pipes={
    "sync":common_exec_pipe,
    "new_lds":common_exec_pipe,
    "zero":common_exec_pipe,
    "buffer_load": ExecPipe("vmem", fifo_size=5, latency=50, throughput=2,visible=True),
    "lds_write" : lds_exec_pipe,
    "lds_read" : lds_exec_pipe,
    "mfma" : ExecPipe("mfma", fifo_size=1, latency=16, throughput=16,visible=True),
})
Program(gcn)
print(gcn)

'''
虚拟执行这个程序，每个指令有一个估计耗时
假设issue 4个cycle开销，每个issue时间窗口选择数据依赖满足的指令进行issue
虚拟执行管线，内置FIFO, FIFO满的状态下issue会引起store，但是不满的状态下可以连续issue
指令执行完毕之后，FIFO就会出现空缺可以继续issue。

执行管线在issue的同时逐个cycle运行管理FIFO中指令的执行进度。
指令进入执行管线是按照指令类型分类的。

'''


gcn.prepare()
for clock in range(1000):
    gcn.clock_step(clock)
