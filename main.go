package main

import (
	"fmt"
	"nnfs/nn"

	"github.com/jnaraujo/numgo/ng"
)

func main() {
	fmt.Println(nn.MSELoss(ng.NewArray[float64](1, 0, 0, 1), ng.NewArray[float64](0, 0, 0, 0)))
}

func SimpleNeuron() {
	weights := ng.NewArray(0, 1.0)
	bias := 0.0
	n1 := nn.NewNeuron(weights, bias)
	h1 := n1.FeedForward(ng.NewArray(2., 3.))
	h2 := h1
	o1 := n1.FeedForward(ng.NewArray(h1, h2))

	fmt.Println(o1)
}

func SimpleNeuralNetwork() {
	network := nn.NewSimpleNeuralNetwork()
	fmt.Println(network.FeedForward(ng.NewArray(2., 3.)))
}
