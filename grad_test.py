import tinygrad
from tinygrad import Tensor, nn

def round_straight_through(x: Tensor) -> Tensor:
    x_det = x.detach()                   
    rounded_val = x_det.round()         
    # Combine to form output: (rounded_val - x_det) has no grad, x has grad.
    return x + (rounded_val - x_det)

if __name__ == "__main__":
    x = Tensor([2.9], requires_grad=True).to("CPU")
    # x = x.round()
    x = x.nround()
    # x = x + Tensor.randint(x.shape, low=0, high=2).sub(0.5)
    # x.gradient(x, gradient=x+0.5)
    # x = x.max()
    # y = x.cos().sum()
    # x[0].backward()
    # y.backward()

    # print(y.grad.tolist())
    print(x.grad.tolist())