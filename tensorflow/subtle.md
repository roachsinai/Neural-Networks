# Packages

## tf.train

### tf.train.GradientDescentOptimizer.minimize

内部调用了`compute_gradients, apply_gradients`，如果想在`apply_gradients`之前执行一些对梯队的其他操作，就分开调用这两个函数。

### tf.train.ExponentialMovingAverage

滑动平均可以在一定程度上提高最终模型在测试集上的表现。

![](./img/Exponential_smoothing.svg)

**Why is it “exponential”?**

![](./img/Exponential_smoothing2.svg)

+ param decay: 指定好的，通常是`1-alpha=0.999, 0.9999`
+ param num_updates: optional 通常每次传入已经执行的迭代次数，指定的情况下用来动态更改衰减因子，min{decay, (1+num_updates)/(10+num_updates)}

### tf.train.exponential_decay

```{.python .input}
decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
```

+ param global_step: 已迭代次数
+ param decay_steps: 每迭代多少次，学习率衰减一次
+ param staircase=False: 连续性衰减或阶梯形衰减
