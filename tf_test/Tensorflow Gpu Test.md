### Tensorflow Gpu Test

`Tensorflow_gpu_test.py` can check Tensorflow can use gpu or not.

Cuda 10.0

Tensorflow 1.13.0

```
 Created TensorFlow device (/device:GPU:0 with 4716 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 13993218817512968774
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 4945621811
locality {
  bus_id: 1
  links {
  }
}
incarnation: 11112430543432122253
physical_device_desc: "device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1"
]
```



