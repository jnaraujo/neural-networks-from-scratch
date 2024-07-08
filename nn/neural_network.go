package nn

import "github.com/jnaraujo/numgo/ng"

type NeuralNetwork struct {
	h1, h2, o1 *Neuron
}

func NewSimpleNeuralNetwork() *NeuralNetwork {
	weights := ng.NewArray(0., 1.)
	bias := 0.

	return &NeuralNetwork{
		h1: NewNeuron(weights, bias),
		h2: NewNeuron(weights, bias),
		o1: NewNeuron(weights, bias),
	}
}

func (nn *NeuralNetwork) FeedForward(x ng.Array[float64]) float64 {
	outH1 := nn.h1.FeedForward(x)
	outH2 := nn.h2.FeedForward(x)
	outO1 := nn.o1.FeedForward(ng.NewArray(outH1, outH2))
	return outO1
}
