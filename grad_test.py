import tinygrad
from tinygrad import Tensor, nn, Device

def round_straight_through(x: Tensor) -> Tensor:
    x_det = x.detach()                   
    rounded_val = x_det.round()         
    # Combine to form output: (rounded_val - x_det) has no grad, x has grad.
    return x + (rounded_val - x_det)

if __name__ == "__main__":
    # Device.DEFAULT = "CPU"
    for i in range(10):
        with Tensor.train():
            x = Tensor([2.9], requires_grad=True)
            # y = x.round()
            y = x.nround().round()
            # x = x + Tensor.randint(x.shape, low=0, high=2).sub(0.5)
            # x.gradient(x, gradient=x+0.5)
            # x = x.max()
            # y = x.cos().sum()
            y.backward(gradient=Tensor([0.0]))
            # y.backward()

        # print(y.grad.tolist())
        print(f"y = {y.numpy()}, x grad = {x.grad.tolist()}")
        # print(y.grad.tolist())