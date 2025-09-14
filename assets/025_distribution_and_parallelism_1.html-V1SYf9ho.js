import{_ as i,a as n,b as a,c as r,d,e as s}from"./025_barrier-Cle6-zl5.js";import{_ as t}from"./plugin-vue_export-helper-x3n3nnut.js";import{o as l,c,f as u,a as e,b as v,e as o}from"./app--klcE6vy.js";const m={},b=e("h1",{id:"分布式训练之二-并行编程-parallel-programming",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#分布式训练之二-并行编程-parallel-programming","aria-hidden":"true"},"#"),v(" 分布式训练之二：并行编程 Parallel Programming")],-1),h=e("ul",null,[e("li",null,"Broadcast"),e("li",null,"Reduce & AllReduce"),e("li",null,"Gather & AllGather"),e("li",null,"Scatter & ReduceScatter")],-1),p=o('<h2 id="_0-点对点通信与集体通信" tabindex="-1"><a class="header-anchor" href="#_0-点对点通信与集体通信" aria-hidden="true">#</a> 0. 点对点通信与集体通信</h2><p>点对点通信（Peer-to-Peer Communication）是一种网络通信模式，其中<strong>两个或多个计算机或设备之间直接进行通信，而不需要通过中央服务器或集中式系统</strong>。在点对点通信中，每个参与者都可以充当客户端和服务器，能够直接与其他节点通信、交换信息或共享资源。</p><p>集体通信（Collective Communication）是指<strong>一组计算节点或处理单元之间进行协作、交换信息或执行通信操作的过程</strong>。这种通信形式<strong>涉及到多个节点之间的集体参与，而不仅仅是点对点的通信</strong>。它可以用于并行计算、分布式系统和群集计算等领域，以便在多个节点之间协调和管理数据的传输、处理和同步操作。</p><h2 id="_1-集体通信概览" tabindex="-1"><a class="header-anchor" href="#_1-集体通信概览" aria-hidden="true">#</a> 1. 集体通信概览</h2><ul><li>Broadcast</li><li>Reduce</li><li>AllReduce</li><li>Gather</li><li>AllGather</li><li>Scatter</li><li>ReduceScatter</li><li>Barrier</li></ul><p>Note：root节点作为服务器，它是某些操作的目标或源</p><p>关系：AllReduce = ReduceScatter + AllGather</p><figure><img src="'+i+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h2 id="_2-broadcast" tabindex="-1"><a class="header-anchor" href="#_2-broadcast" aria-hidden="true">#</a> 2. Broadcast</h2><figure><img src="'+n+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h2 id="_3-reduce-allreduce" tabindex="-1"><a class="header-anchor" href="#_3-reduce-allreduce" aria-hidden="true">#</a> 3. Reduce &amp; AllReduce</h2><p>将每个节点的值通过函数合为一个值</p><p>函数f()常用的为求和（SUM）或求平均值（AVG）</p><ul><li>AVG is only available with the NCCL backend</li></ul><figure><img src="'+a+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><ul><li>Reduce: 结果只发送给root节点</li><li>AllReduce: 结果广播发送到每个节点（每个节点的值一样）</li></ul><h2 id="_4-gather-allgather" tabindex="-1"><a class="header-anchor" href="#_4-gather-allgather" aria-hidden="true">#</a> 4. Gather &amp; AllGather</h2><figure><img src="'+r+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h2 id="_5-scatter-reducescatter" tabindex="-1"><a class="header-anchor" href="#_5-scatter-reducescatter" aria-hidden="true">#</a> 5. Scatter &amp; ReduceScatter</h2><p><img src="'+d+'" alt="" loading="lazy"> Scatter</p><ul><li>scatter不同于broadcast，scatter是将数据分片发送，broadcast是将数据完整发送</li><li>scatter在逻辑上是gather的反向操作</li></ul><p>ReduceScatter</p><ul><li>把每个节点上的数据分片</li><li>每个分片的各节点的数据通过函数进行Reduce</li><li>把每个分片Reduce的结果Scatter到每个节点</li></ul><h2 id="_6-barrier" tabindex="-1"><a class="header-anchor" href="#_6-barrier" aria-hidden="true">#</a> 6. Barrier</h2><p>所有节点到达障碍Barrier之前，Barrier不会解除，都到达Barrier后才能进行后续的计算，用于同步节点</p><figure><img src="'+s+`" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h2 id="_7-pytorch代码实现" tabindex="-1"><a class="header-anchor" href="#_7-pytorch代码实现" aria-hidden="true">#</a> 7. PyTorch代码实现</h2><h3 id="_7-1-nccl是什么" tabindex="-1"><a class="header-anchor" href="#_7-1-nccl是什么" aria-hidden="true">#</a> 7.1. NCCL是什么</h3><p>NCCL</p><ul><li>NVIDIA Collective Communications Library</li><li>NVIDIA 集体通信库</li><li>NVIDIA的GPU间通信的优化原语</li><li>NCCL 专为高效的 GPU-GPU 通信设计</li></ul><h3 id="_7-2-broadcast" tabindex="-1"><a class="header-anchor" href="#_7-2-broadcast" aria-hidden="true">#</a> 7.2. Broadcast</h3><p>代码</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>import torch
import torch.distributed as dist

def init_process():
    dist.init_process_group(backend=&#39;nccl&#39;)
    torch.cuda.set_device(dist.get_rank())
    
def example_broadcast():
    if dist.get_rank() == 0:
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
    else:
        tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f&quot;Before broadcast on rank {dist.get_rank()}: {tensor}&quot;)
    dist.broadcast(tensor, src=0)
    print(f&quot;After broadcast on rank {dist.get_rank()}: {tensor}&quot;)
    
init_process()
example_broadcats()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>输出</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>Before broadcast on rank 0: tensor([1., 2., 3., 4., 5.], device=&#39;cuda:0&#39;)
Before broadcast on rank 1: tensor([0., 0., 0., 0., 0.], device=&#39;cuda:1&#39;)
Before broadcast on rank 2: tensor([0., 0., 0., 0., 0.], device=&#39;cuda:2&#39;)

After broadcast on rank 0: tensor([1., 2., 3., 4., 5.], device=&#39;cuda:0&#39;)
After broadcast on rank 1: tensor([1., 2., 3., 4., 5.], device=&#39;cuda:1&#39;)
After broadcast on rank 2: tensor([1., 2., 3., 4., 5.], device=&#39;cuda:2&#39;)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_7-3-reduce" tabindex="-1"><a class="header-anchor" href="#_7-3-reduce" aria-hidden="true">#</a> 7.3. Reduce</h3><p>代码</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>def example_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f&quot;Before reduce on rank {dist.get_rank()}: {tensor}&quot;)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print(f&quot;After reduce on rank {rank}: {tensor}&quot;)
    
init_process()
example_reduce()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>输出</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>Before reduce on rank 0: tensor([1., 1., 1., 1., 1.], device=&#39;cuda:0&#39;)
Before reduce on rank 1: tensor([2., 2., 2., 2., 2.], device=&#39;cuda:1&#39;)
Before reduce on rank 2: tensor([3., 3., 3., 3., 3.], device=&#39;cuda:2&#39;)

After reduce on rank 0: tensor([6., 6., 6., 6., 6.], device=&#39;cuda:0&#39;)
After reduce on rank 1: tensor([2., 2., 2., 2., 2.], device=&#39;cuda:1&#39;)
After reduce on rank 2: tensor([3., 3., 3., 3., 3.], device=&#39;cuda:2&#39;)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_7-4-allreduce" tabindex="-1"><a class="header-anchor" href="#_7-4-allreduce" aria-hidden="true">#</a> 7.4. AllReduce</h3><p>代码</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>def example_all_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f&quot;Before all_reduce on rank {dist.get_rank()}: {tensor}&quot;)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f&quot;After all_reduce on rank {dist.get_rank()}: {tensor}&quot;)
    
init_process()
example_all_reduce()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>输出</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>Before all_reduce on rank 0: tensor([1., 1., 1., 1., 1.], device=&#39;cuda:0&#39;)
Before all_reduce on rank 1: tensor([2., 2., 2., 2., 2.], device=&#39;cuda:1&#39;)
Before all_reduce on rank 2: tensor([3., 3., 3., 3., 3.], device=&#39;cuda:2&#39;)

After all_reduce on rank 0: tensor([6., 6., 6., 6., 6.], device=&#39;cuda:0&#39;)
After all_reduce on rank 1: tensor([6., 6., 6., 6., 6.], device=&#39;cuda:1&#39;)
After all_reduce on rank 2: tensor([6., 6., 6., 6., 6.], device=&#39;cuda:2&#39;)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_7-5-gather" tabindex="-1"><a class="header-anchor" href="#_7-5-gather" aria-hidden="true">#</a> 7.5. Gather</h3><p>代码</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>def example_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    if dist.get_rank() == 0:
        gather_list = [
            torch.zeros(5, dtype=torch.float32).cuda()
            for _ in range(dist.get_world_size())
            ]
    else:
        gather_list = None
    print(f&quot;Before gather on rank {dist.get_rank()}: {tensor}&quot;)
    dist.gather(tensor, gather_list, dst=0)
    if dist.get_rank() == 0:
        print(f&quot;After gather on rank 0: {gather_list}&quot;)
    
init_process()
example_gather()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>输出</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>Before gather on rank 0: tensor([1., 1., 1., 1., 1.], device=&#39;cuda:0&#39;)
Before gather on rank 1: tensor([2., 2., 2., 2., 2.], device=&#39;cuda:1&#39;)
Before gather on rank 2: tensor([3., 3., 3., 3., 3.], device=&#39;cuda:2&#39;)

After gather on rank 0: [tensor([1., 1., 1., 1., 1.], device=&#39;cuda:0&#39;),
                         tensor([2., 2., 2., 2., 2.], device=&#39;cuda:0&#39;),
                         tensor([3., 3., 3., 3., 3.], device=&#39;cuda:0&#39;)]
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_7-6-allgather" tabindex="-1"><a class="header-anchor" href="#_7-6-allgather" aria-hidden="true">#</a> 7.6. AllGather</h3><p>代码</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>def example_all_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    gather_list = [
        torch.zeros(5, dtype=torch.float32).cuda()
        for _ in range(dist.get_world_size())
        ]
    print(f&quot;Before all_gather on rank {dist.get_rank()}: {tensor}&quot;)
    dist.all_gather(gather_list, tensor)
    print(f&quot;After all_gather on rank {dist.get_rank()}: {gather_list}&quot;)
    
init_process()
example_all_gather()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>输出</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>Before all_gather on rank 0: tensor([1., 1., 1., 1., 1.], device=&#39;cuda:0&#39;)
Before all_gather on rank 1: tensor([2., 2., 2., 2., 2.], device=&#39;cuda:1&#39;)
Before all_gather on rank 2: tensor([3., 3., 3., 3., 3.], device=&#39;cuda:2&#39;)

After all_gather on rank 0: [tensor([1., 1., 1., 1., 1.], device=&#39;cuda:0&#39;),
                             tensor([2., 2., 2., 2., 2.], device=&#39;cuda:0&#39;),
                             tensor([3., 3., 3., 3., 3.], device=&#39;cuda:0&#39;)]
After all_gather on rank 1: [tensor([1., 1., 1., 1., 1.], device=&#39;cuda:1&#39;),
                             tensor([2., 2., 2., 2., 2.], device=&#39;cuda:0&#39;),
                             tensor([3., 3., 3., 3., 3.], device=&#39;cuda:0&#39;)]
After all_gather on rank 2: [tensor([1., 1., 1., 1., 1.], device=&#39;cuda:2&#39;),
                             tensor([2., 2., 2., 2., 2.], device=&#39;cuda:2&#39;),
                             tensor([3., 3., 3., 3., 3.], device=&#39;cuda:2&#39;)]
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_7-7-scatter" tabindex="-1"><a class="header-anchor" href="#_7-7-scatter" aria-hidden="true">#</a> 7.7. Scatter</h3><p>代码</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>def example_scatter():
    if dist.get_rank() == 0:
        scatter_list = [
            torch.tensor([i + 1] * 5, dtype=torch.float32).cuda()
            for i in range(dist.get_world_size())
            ]
        print(f&quot;Rank 0: Tensor to scatter: {scatter_list}&quot;)
    else:
        scatter_list = None
    tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f&quot;Before scatter on rank {dist.get_rank()}: {tensor}&quot;)
    dist.scatter(tensor, scatter_list, src=0)
    print(f&quot;After scatter on rank {dist.get_rank()}: {tensor}&quot;)
    
init_process()
example_scatter()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>输出</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>Rank 0: Tensor to scatter: [tensor([1., 1., 1., 1., 1.], device=&#39;cuda:0&#39;),
                            tensor([2., 2., 2., 2., 2.], device=&#39;cuda:0&#39;),
                            tensor([3., 3., 3., 3., 3.], device=&#39;cuda:0&#39;)]
Before scatter on rank 0: tensor([0., 0., 0., 0., 0.], device=&#39;cuda:0&#39;)
Before scatter on rank 1: tensor([0., 0., 0., 0., 0.], device=&#39;cuda:1&#39;)
Before scatter on rank 2: tensor([0., 0., 0., 0., 0.], device=&#39;cuda:2&#39;)

After scatter on rank 0: tensor([1., 1., 1., 1., 1.], device=&#39;cuda:0&#39;)
After scatter on rank 1: tensor([2., 2., 2., 2., 2.], device=&#39;cuda:1&#39;)
After scatter on rank 2: tensor([3., 3., 3., 3., 3.], device=&#39;cuda:2&#39;)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_7-8-reducescatter" tabindex="-1"><a class="header-anchor" href="#_7-8-reducescatter" aria-hidden="true">#</a> 7.8. ReduceScatter</h3><p>代码</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>def example_reduce_scatter():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    input_tensor = [
        torch.tensor([(rank + 1) * i for i in range(1, 3)], dtype=torch.float32).cuda()**(j+1) 
        for j in range(world_size)
        ]
    output_tensor = torch.zeros(2, dtype=torch.float32).cuda()
    print(f&quot;Before ReduceScatter on rank {rank}: {input_tensor}&quot;)
    dist.reduce_scatter(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    print(f&quot;After ReduceScatter on rank {rank}: {output_tensor}&quot;)    
    
init_process()
example_reduce_scatter()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>输出</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>Before ReduceScatter on rank 0: [tensor([1., 2.], device=&#39;cuda:0&#39;),
											 tensor([1., 4.], device=&#39;cuda:0&#39;),
											 tensor([1., 8.], device=&#39;cuda:0&#39;)]
Before ReduceScatter on rank 1: [tensor([2., 4.], device=&#39;cuda:1&#39;),
                                 tensor([ 4., 16.], device=&#39;cuda:1&#39;),
                                 tensor([ 8., 64.], device=&#39;cuda:1&#39;)]
Before ReduceScatter on rank 2: [tensor([3., 6.], device=&#39;cuda:2&#39;),
                                 tensor([ 9., 36.], device=&#39;cuda:2&#39;),
                                 tensor([ 27., 216.], device=&#39;cuda:2&#39;)]

After ReduceScatter on rank 0: tensor([ 6., 12.], device=&#39;cuda:0&#39;)
After ReduceScatter on rank 1: tensor([14., 56.], device=&#39;cuda:1&#39;)
After ReduceScatter on rank 2: tensor([ 36., 288.], device=&#39;cuda:2&#39;)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_7-9-barrier" tabindex="-1"><a class="header-anchor" href="#_7-9-barrier" aria-hidden="true">#</a> 7.9. Barrier</h3><p>代码</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>def example_barrier():
    rank = dist.get_rank()
    t_start = time.time()
    print(f&quot;Rank {rank} sleeps {rank} seconds.&quot;)
    time.sleep(rank)  # Simulate different processing times
    dist.barrier()
    print(f&quot;Rank {rank} after barrier time delta: {time.time()-t_start:.4f}&quot;)
    
init_process()
example_barrier()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>输出</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>Rank 0 sleeps 0 seconds.
Rank 1 sleeps 1 seconds.
Rank 2 sleeps 2 seconds.

Rank 0 after barrier time delta: 2.0025
Rank 1 after barrier time delta: 2.0025
Rank 2 after barrier time delta: 2.0024
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,70);function _(g,f){return l(),c("div",null,[b,h,u(" more "),p])}const B=t(m,[["render",_],["__file","025_distribution_and_parallelism_1.html.vue"]]);export{B as default};
